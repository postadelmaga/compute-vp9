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

    uint32_t         width;
    uint32_t         height;

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

    if (hdr->width > 0 && hdr->height > 0) {
        ctx->width = hdr->width;
        ctx->height = hdr->height;
    } else {
        hdr->width = ctx->width;
        hdr->height = ctx->height;
    }

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

    /* Adapt entropy probabilities for the next frame */
    vp9_adapt_probabilities(&ctx->probs, ctx->parsed_frame);

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

static cvp9_err_t get_frame_common(cvp9_ctx_t *ctx, cvp9_frame_info_t *info, int wait)
{
    if (!ctx || !info) return CVP9_ERR_INVALID_DATA;

#ifdef ENABLE_VULKAN
    if (ctx->backend == CVP9_BACKEND_VULKAN)
        return vulkan_get_frame(ctx->backend_ctx, info, wait);
#endif
    (void)wait;  /* CPU frames are always ready once queued */

    if (ctx->frame_count > 0) {
        *info = ctx->frames[ctx->frame_rd];
        ctx->frame_rd = (ctx->frame_rd + 1) % 4;
        ctx->frame_count--;
        return CVP9_OK;
    }

    return CVP9_ERR_UNSUPPORTED;
}

cvp9_err_t cvp9_get_frame(cvp9_ctx_t *ctx, cvp9_frame_info_t *info)
{
    return get_frame_common(ctx, info, 0);
}

cvp9_err_t cvp9_get_frame_sync(cvp9_ctx_t *ctx, cvp9_frame_info_t *info)
{
    return get_frame_common(ctx, info, 1);
}

cvp9_err_t cvp9_get_frame_dmabuf(cvp9_ctx_t *ctx, cvp9_dmabuf_frame_t *out)
{
    if (!ctx || !out) return CVP9_ERR_INVALID_DATA;

#ifdef ENABLE_VULKAN
    if (ctx->backend == CVP9_BACKEND_VULKAN)
        return vulkan_get_frame_dmabuf(ctx->backend_ctx, out);
#endif

    return CVP9_ERR_UNSUPPORTED;
}

cvp9_err_t cvp9_export_buffer_alloc(cvp9_ctx_t *ctx, uint64_t size, cvp9_export_buffer_t *out)
{
    if (!ctx || !out) return CVP9_ERR_INVALID_DATA;

#ifdef ENABLE_VULKAN
    if (ctx->backend == CVP9_BACKEND_VULKAN)
        return vulkan_export_buffer_alloc(ctx->backend_ctx, size, out);
#endif

    return CVP9_ERR_UNSUPPORTED;
}

void cvp9_export_buffer_free(cvp9_ctx_t *ctx, cvp9_export_buffer_t *buf)
{
    if (!ctx || !buf) return;

#ifdef ENABLE_VULKAN
    if (ctx->backend == CVP9_BACKEND_VULKAN)
        vulkan_export_buffer_free(ctx->backend_ctx, buf);
#endif
}

cvp9_err_t cvp9_set_render_target(cvp9_ctx_t *ctx, const cvp9_export_buffer_t *target)
{
    if (!ctx) return CVP9_ERR_INVALID_DATA;

#ifdef ENABLE_VULKAN
    if (ctx->backend == CVP9_BACKEND_VULKAN)
        return vulkan_set_render_target(ctx->backend_ctx, target);
#endif

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
    case CVP9_ERR_AGAIN:        return "Frame not ready yet (try again)";
    default:                    return "Unknown error";
    }
}

#ifdef ENABLE_VAAPI
cvp9_err_t cvp9_decode_vaapi(cvp9_ctx_t *ctx,
                             const uint8_t *data, size_t size,
                             int64_t pts,
                             const VADecPictureParameterBufferVP9 *pic_param,
                             const VASliceParameterBufferVP9 *slice_param)
{
    if (!ctx || !data || size == 0 || !pic_param || !slice_param)
        return CVP9_ERR_INVALID_DATA;

    /* 1. Populate ctx->last_frame_hdr from VA-API picture parameters */
    vp9_frame_header_t *hdr = &ctx->last_frame_hdr;
    memset(hdr, 0, sizeof(*hdr));

    hdr->frame_type = pic_param->pic_fields.bits.frame_type ? VP9_FRAME_NON_KEY : VP9_FRAME_KEY;
    hdr->width = pic_param->frame_width;
    hdr->height = pic_param->frame_height;

    if (hdr->width > 0 && hdr->height > 0) {
        ctx->width = hdr->width;
        ctx->height = hdr->height;
    } else {
        hdr->width = ctx->width;
        hdr->height = ctx->height;
    }
    hdr->profile = pic_param->profile;
    hdr->bit_depth = pic_param->bit_depth;
    hdr->show_frame = pic_param->pic_fields.bits.show_frame;
    hdr->error_resilient = pic_param->pic_fields.bits.error_resilient_mode;
    hdr->log2_tile_cols = pic_param->log2_tile_columns;
    hdr->log2_tile_rows = pic_param->log2_tile_rows;

    hdr->filter_level = pic_param->filter_level;
    hdr->sharpness_level = pic_param->sharpness_level;
    hdr->segmentation_enabled = pic_param->pic_fields.bits.segmentation_enabled;

    /* Active reference slots (LAST/GOLDEN/ALTREF). Refresh flags are not in
     * the VA parameters (the client manages the DPB via surfaces); with our
     * internal reference pool we approximate by refreshing the LAST slot. */
    hdr->ref_frame_idx[0] = pic_param->pic_fields.bits.last_ref_frame;
    hdr->ref_frame_idx[1] = pic_param->pic_fields.bits.golden_ref_frame;
    hdr->ref_frame_idx[2] = pic_param->pic_fields.bits.alt_ref_frame;
    hdr->refresh_frame_flags = (hdr->frame_type == VP9_FRAME_KEY)
        ? 0xFF
        : (uint8_t)(1u << (pic_param->pic_fields.bits.last_ref_frame & 7));

    /* base_qindex is not part of the VA-API picture parameters — it lives in
     * the uncompressed frame header, so parse it from the bitstream */
    {
        vp9_bitreader_t br;
        vp9_frame_header_t parsed;
        vp9_bitreader_init(&br, data, size);
        if (vp9_parse_frame_header(&br, &parsed) == 0) {
            hdr->base_qindex = parsed.base_qindex;
        }
    }

    /* 2. Allocate or resize parsed_frame if dimensions changed */
    if (!ctx->parsed_frame || ctx->parsed_frame->hdr.width != hdr->width || ctx->parsed_frame->hdr.height != hdr->height) {
        if (ctx->parsed_frame) vp9_parsed_frame_free(ctx->parsed_frame);
        ctx->parsed_frame = vp9_parsed_frame_alloc(hdr->width, hdr->height);
        if (!ctx->parsed_frame) return CVP9_ERR_NOMEM;
        
        /* Initialize default entropy probabilities */
        vp9_entropy_probs_init(&ctx->probs);
    }

    /* 3. Parse tiles from the bitstream using the range coder/entropy parser */
    size_t tile_data_offset = slice_param->slice_data_offset + pic_param->first_partition_size;
    if (tile_data_offset >= size) {
        tile_data_offset = 0;
    }
    const uint8_t *tile_data = data + tile_data_offset;
    size_t tile_size = size - tile_data_offset;

    int parse_rc = vp9_decode_tiles(hdr, tile_data, tile_size, &ctx->probs, ctx->parsed_frame);
    if (parse_rc != 0) return CVP9_ERR_INVALID_DATA;

    /* Adapt entropy probabilities for the next frame */
    vp9_adapt_probabilities(&ctx->probs, ctx->parsed_frame);

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
#endif
