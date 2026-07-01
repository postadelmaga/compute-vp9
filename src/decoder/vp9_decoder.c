#include "compute_vp9/decoder.h"
#include "decoder/vp9_bitstream.h"
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
    /* Free any queued frame buffers */
    for (int i = 0; i < 4; i++) {
        free(ctx->frames[i].plane_y);
        ctx->frames[i].plane_y = NULL;
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

    /* Dispatch to compute backend */
    cvp9_err_t err = CVP9_ERR_UNSUPPORTED;

#ifdef ENABLE_VULKAN
    if (ctx->backend == CVP9_BACKEND_VULKAN)
        err = vulkan_decode_frame(ctx->backend_ctx, hdr, data, size, pts);
#endif

    if (err == CVP9_ERR_UNSUPPORTED) {
        /* TODO: CPU reference decode */
        (void)pts;
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
