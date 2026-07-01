#include "compute_vp9/decoder.h"
#include "decoder/vp9_bitstream.h"
#include "decoder/vp9_parsed_frame.h"
#include "decoder/vp9_entropy.h"
#include "decoder/vp9_tile.h"
#include "decoder/vp9_reconstruct.h"
#include "decoder/vp9_frame.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#ifdef ENABLE_VULKAN
#include "backend/vulkan/vulkan_backend.h"
#endif
#ifdef ENABLE_OPENCL
#include "backend/opencl/opencl_backend.h"
#endif

/* ── Internal context ───────────────────────────────────────────────────── */

struct cvp9_ctx {
    cvp9_backend_t   backend;
    vp9_frame_header_t last_frame_hdr;

    /* Entropy probabilities and parsed metadata context */
    vp9_entropy_probs_t probs;
    vp9_parsed_frame_t *parsed_frame;

    /* Reference frame pool (8 slots) */
    cvp9_frame_info_t  ref_frames[8];

    /* Frame output queue (ring buffer, capacity=4) */
    cvp9_frame_info_t  frames[4];
    int                frame_rd;
    int                frame_wr;
    int                frame_count;

    /* Backend-specific state */
    void              *backend_ctx;
};

/* ── Helpers ────────────────────────────────────────────────────────────── */

static cvp9_backend_t select_backend(cvp9_backend_t preferred)
{
#ifdef ENABLE_VULKAN
    if (preferred == CVP9_BACKEND_AUTO || preferred == CVP9_BACKEND_VULKAN)
        return CVP9_BACKEND_VULKAN;
#endif
#ifdef ENABLE_OPENCL
    if (preferred == CVP9_BACKEND_AUTO || preferred == CVP9_BACKEND_OPENCL)
        return CVP9_BACKEND_OPENCL;
#endif
    return CVP9_BACKEND_CPU;
}

/* ── Public API ─────────────────────────────────────────────────────────── */

cvp9_err_t cvp9_create(const cvp9_config_t *cfg, cvp9_ctx_t **out)
{
    cvp9_ctx_t *ctx = calloc(1, sizeof(*ctx));
    if (!ctx) return CVP9_ERR_NOMEM;

    cvp9_backend_t preferred = cfg ? cfg->backend : CVP9_BACKEND_AUTO;
    ctx->backend = select_backend(preferred);

    cvp9_err_t err = CVP9_OK;

#ifdef ENABLE_VULKAN
    if (ctx->backend == CVP9_BACKEND_VULKAN) {
        err = vulkan_backend_init(&ctx->backend_ctx);
        if (err != CVP9_OK) {
            fprintf(stderr, "[compute-vp9] Vulkan init failed (%s), falling back to CPU\n",
                    cvp9_err_str(err));
            ctx->backend = CVP9_BACKEND_CPU;
        }
    }
#endif

    if (ctx->backend == CVP9_BACKEND_CPU)
        fprintf(stderr, "[compute-vp9] WARNING: using CPU software fallback\n");

    *out = ctx;
    return CVP9_OK;
}

void cvp9_destroy(cvp9_ctx_t *ctx)
{
    if (!ctx) return;
#ifdef ENABLE_VULKAN
    if (ctx->backend == CVP9_BACKEND_VULKAN && ctx->backend_ctx)
        vulkan_backend_destroy(ctx->backend_ctx);
#endif
    /* Free parsed frame structure */
    vp9_parsed_frame_free(ctx->parsed_frame);

    /* Free reference frames pool */
    for (int i = 0; i < 8; i++) {
        cvp9_frame_free(&ctx->ref_frames[i]);
    }

    /* Free any queued frame buffers */
    for (int i = 0; i < 4; i++) {
        cvp9_frame_free(&ctx->frames[i]);
    }
    free(ctx);
}

cvp9_err_t cvp9_decode(cvp9_ctx_t *ctx,
                        const uint8_t *data, size_t size,
                        int64_t pts)
{
    if (!ctx || !data || size == 0) return CVP9_ERR_INVALID_DATA;

    vp9_bitreader_t br;
    vp9_bitreader_init(&br, data, size);

    int rc = vp9_parse_frame_header(&br, &ctx->last_frame_hdr);
    if (rc != 0) return CVP9_ERR_INVALID_DATA;

    vp9_frame_header_t *hdr = &ctx->last_frame_hdr;

    /* 1. Allocate or resize parsed_frame if dimensions changed */
    if (!ctx->parsed_frame || ctx->parsed_frame->hdr.width != hdr->width || ctx->parsed_frame->hdr.height != hdr->height) {
        if (ctx->parsed_frame) vp9_parsed_frame_free(ctx->parsed_frame);
        ctx->parsed_frame = vp9_parsed_frame_alloc(hdr->width, hdr->height);
        if (!ctx->parsed_frame) return CVP9_ERR_NOMEM;
        
        /* Initialize default entropy probabilities */
        vp9_entropy_probs_init(&ctx->probs);
    }

    /* 2. Determine compressed header and tile locations */
    size_t header_start_pos = br.pos;
    if (br.bit < 7) header_start_pos++;
    if (header_start_pos + 2 > size) return CVP9_ERR_INVALID_DATA;
    
    uint32_t header_size = ((uint32_t)data[header_start_pos] << 8) | data[header_start_pos + 1];
    size_t comp_header_start = header_start_pos + 2;
    if (comp_header_start + header_size > size) return CVP9_ERR_INVALID_DATA;
    
    const uint8_t *tile_data = data + comp_header_start + header_size;
    size_t tile_size = size - comp_header_start - header_size;

    /* 3. Parse tiles and macroblocks/coefficients */
    int parse_rc = vp9_decode_tiles(hdr, tile_data, tile_size, &ctx->probs, ctx->parsed_frame);
    if (parse_rc != 0) return CVP9_ERR_INVALID_DATA;

    /* 4. Dispatch to compute backend or CPU fallback */
    cvp9_err_t err = CVP9_ERR_UNSUPPORTED;

#ifdef ENABLE_VULKAN
    if (ctx->backend == CVP9_BACKEND_VULKAN) {
        err = vulkan_decode_frame(ctx->backend_ctx, ctx->parsed_frame, pts);
    }
#endif

    if (err == CVP9_ERR_UNSUPPORTED) {
        /* CPU reference decode & reconstruction */
        cvp9_frame_info_t reconstructed_frame;
        memset(&reconstructed_frame, 0, sizeof(reconstructed_frame));
        
        const cvp9_frame_info_t *refs[8];
        for (int i = 0; i < 8; i++) {
            refs[i] = &ctx->ref_frames[i];
        }
        
        if (!vp9_reconstruct_frame(ctx->parsed_frame, refs, &reconstructed_frame)) {
            return CVP9_ERR_NOMEM;
        }
        reconstructed_frame.pts = pts;

        /* Update reference frame pool based on keyframe status */
        if (hdr->frame_type == VP9_FRAME_KEY) {
            for (int i = 0; i < 8; i++) {
                cvp9_frame_copy(&ctx->ref_frames[i], &reconstructed_frame);
            }
        } else {
            cvp9_frame_copy(&ctx->ref_frames[0], &reconstructed_frame);
        }

        /* Push to output queue (ring buffer, capacity 4) */
        if (ctx->frame_count < 4) {
            cvp9_frame_copy(&ctx->frames[ctx->frame_wr], &reconstructed_frame);
            ctx->frame_wr = (ctx->frame_wr + 1) % 4;
            ctx->frame_count++;
        } else {
            cvp9_frame_free(&ctx->frames[ctx->frame_rd]);
            cvp9_frame_copy(&ctx->frames[ctx->frame_rd], &reconstructed_frame);
            ctx->frame_rd = (ctx->frame_rd + 1) % 4;
            ctx->frame_wr = ctx->frame_rd;
        }
        
        cvp9_frame_free(&reconstructed_frame);
        err = CVP9_OK;
    }

    return err;
}

cvp9_err_t cvp9_get_frame(cvp9_ctx_t *ctx, cvp9_frame_info_t *info)
{
    if (!ctx || !info) return CVP9_ERR_INVALID_DATA;

#ifdef ENABLE_VULKAN
    if (ctx->backend == CVP9_BACKEND_VULKAN)
        return vulkan_get_frame(ctx->backend_ctx, info);
#endif

    if (ctx->frame_count > 0) {
        *info = ctx->frames[ctx->frame_rd];
        ctx->frame_rd = (ctx->frame_rd + 1) % 4;
        ctx->frame_count--;
        return CVP9_OK;
    }

    return CVP9_ERR_UNSUPPORTED;
}

cvp9_backend_t cvp9_active_backend(const cvp9_ctx_t *ctx)
{
    return ctx ? ctx->backend : CVP9_BACKEND_CPU;
}

const char *cvp9_err_str(cvp9_err_t err)
{
    switch (err) {
    case CVP9_OK:               return "OK";
    case CVP9_ERR_NOMEM:        return "Out of memory";
    case CVP9_ERR_NO_BACKEND:   return "No compute backend available";
    case CVP9_ERR_INVALID_DATA: return "Invalid bitstream data";
    case CVP9_ERR_UNSUPPORTED:  return "Unsupported operation";
    case CVP9_ERR_GPU:          return "GPU error";
    default:                    return "Unknown error";
    }
}
