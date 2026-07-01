/**
 * compute-vp9 — VP9 decode via GPU compute (Vulkan/OpenCL) + VA-API
 *
 * Public decoder API
 */
#pragma once

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Opaque handles ──────────────────────────────────────────────────────── */
typedef struct cvp9_ctx    cvp9_ctx_t;
typedef struct cvp9_frame  cvp9_frame_t;
typedef struct cvp9_packet cvp9_packet_t;

/* ── Backend selection ───────────────────────────────────────────────────── */
typedef enum {
    CVP9_BACKEND_AUTO    = 0,   /**< Auto-select best available backend      */
    CVP9_BACKEND_VULKAN  = 1,   /**< Vulkan 1.1+ compute shaders             */
    CVP9_BACKEND_OPENCL  = 2,   /**< OpenCL 2.0+ compute kernels             */
    CVP9_BACKEND_CPU     = 3,   /**< Software fallback (reference)           */
} cvp9_backend_t;

/* ── Return codes ────────────────────────────────────────────────────────── */
typedef enum {
    CVP9_OK               =  0,
    CVP9_ERR_NOMEM        = -1,
    CVP9_ERR_NO_BACKEND   = -2,
    CVP9_ERR_INVALID_DATA = -3,
    CVP9_ERR_UNSUPPORTED  = -4,
    CVP9_ERR_GPU          = -5,
} cvp9_err_t;

/* ── Context configuration ───────────────────────────────────────────────── */
typedef struct {
    cvp9_backend_t  backend;        /**< Preferred compute backend           */
    uint32_t        thread_count;   /**< CPU threads for entropy decode (0=auto) */
    int             prefer_low_power; /**< Prefer iGPU over dGPU            */
} cvp9_config_t;

/* ── Frame description ───────────────────────────────────────────────────── */
typedef struct {
    uint32_t    width;
    uint32_t    height;
    uint32_t    stride_y;
    uint32_t    stride_uv;
    uint8_t    *plane_y;
    uint8_t    *plane_u;
    uint8_t    *plane_v;
    int64_t     pts;
} cvp9_frame_info_t;

/* ── API ─────────────────────────────────────────────────────────────────── */

/**
 * Create a decoder context.
 * @param cfg  Configuration (may be NULL for defaults)
 * @param ctx  Output context handle
 * @return     CVP9_OK on success
 */
cvp9_err_t cvp9_create(const cvp9_config_t *cfg, cvp9_ctx_t **ctx);

/**
 * Destroy a decoder context and release all GPU/CPU resources.
 */
void cvp9_destroy(cvp9_ctx_t *ctx);

/**
 * Submit a VP9 bitstream packet for decoding.
 * @param ctx   Decoder context
 * @param data  Raw VP9 bitstream bytes
 * @param size  Byte count
 * @param pts   Presentation timestamp (optional, pass 0)
 * @return      CVP9_OK on success
 */
cvp9_err_t cvp9_decode(cvp9_ctx_t *ctx,
                        const uint8_t *data, size_t size,
                        int64_t pts);

#ifdef ENABLE_VAAPI
#include <va/va.h>
#include <va/va_dec_vp9.h>
cvp9_err_t cvp9_decode_vaapi(cvp9_ctx_t *ctx,
                             const uint8_t *data, size_t size,
                             int64_t pts,
                             const VADecPictureParameterBufferVP9 *pic_param,
                             const VASliceParameterBufferVP9 *slice_param);
#endif

/**
 * Retrieve the next decoded frame, if available.
 * @param ctx    Decoder context
 * @param info   Output frame info (caller must not free plane pointers)
 * @return       CVP9_OK if a frame is ready, CVP9_ERR_UNSUPPORTED if queue empty
 */
cvp9_err_t cvp9_get_frame(cvp9_ctx_t *ctx, cvp9_frame_info_t *info);

/**
 * Return a human-readable string for an error code.
 */
const char *cvp9_err_str(cvp9_err_t err);

/**
 * Query the active backend of a context.
 */
cvp9_backend_t cvp9_active_backend(const cvp9_ctx_t *ctx);

#ifdef __cplusplus
}
#endif
