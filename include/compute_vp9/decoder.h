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
    CVP9_ERR_AGAIN        = -6,  /**< Frame in flight but not ready yet       */
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
 * Retrieve the next decoded frame, if available (non-blocking).
 *
 * Decoded frames are delivered in submission order. On GPU backends decoding
 * is pipelined: up to a few frames may be in flight simultaneously so the CPU
 * bitstream parse of frame N+1 overlaps GPU reconstruction of frame N.
 *
 * @param ctx    Decoder context
 * @param info   Output frame info (plane pointers are owned by the decoder
 *               and remain valid until the corresponding pipeline slot is
 *               reused, i.e. for at least the next 2 cvp9_decode() calls)
 * @return       CVP9_OK if a frame is ready,
 *               CVP9_ERR_AGAIN if the oldest frame is still decoding,
 *               CVP9_ERR_UNSUPPORTED if no frame is queued
 */
cvp9_err_t cvp9_get_frame(cvp9_ctx_t *ctx, cvp9_frame_info_t *info);

/**
 * Retrieve the next decoded frame, blocking until it is ready.
 * Same delivery semantics as cvp9_get_frame(), but never returns
 * CVP9_ERR_AGAIN. Returns CVP9_ERR_UNSUPPORTED when no frame is queued.
 */
cvp9_err_t cvp9_get_frame_sync(cvp9_ctx_t *ctx, cvp9_frame_info_t *info);

/* ── Zero-copy cross-GPU export (DMA-BUF) ────────────────────────────────── */

/**
 * A decoded frame exported as a Linux DMA-BUF for zero-copy sharing with
 * another GPU/API (Vulkan external memory, EGL, KMS, ...).
 * Layout is planar I420 inside a single buffer object.
 */
typedef struct {
    int         fd;          /**< dup()'d DMA-BUF fd — caller must close()   */
    uint64_t    size;        /**< total size of the buffer object            */
    uint32_t    width;
    uint32_t    height;
    uint32_t    offsets[3];  /**< Y/U/V plane offsets inside the buffer      */
    uint32_t    pitches[3];  /**< Y/U/V plane pitches                        */
    int64_t     pts;
} cvp9_dmabuf_frame_t;

/**
 * Like cvp9_get_frame_sync(), but instead of CPU plane pointers returns a
 * DMA-BUF fd referencing the decoded frame in GPU-shareable memory. The
 * importing GPU should copy/consume the frame before the pipeline slot is
 * reused (at least the next 2 cvp9_decode() calls).
 *
 * @return CVP9_ERR_UNSUPPORTED if the backend has no DMA-BUF export support
 *         (missing VK_EXT_external_memory_dma_buf) or no frame is queued.
 */
cvp9_err_t cvp9_get_frame_dmabuf(cvp9_ctx_t *ctx, cvp9_dmabuf_frame_t *out);

/**
 * A generic CPU-mappable, DMA-BUF-exportable GPU buffer. Used by the VA-API
 * driver to back render-target surfaces so clients (Chrome) can import them
 * zero-copy via vaExportSurfaceHandle.
 */
typedef struct {
    int         fd;        /**< DMA-BUF fd, owned by the buffer             */
    void       *mapped;    /**< persistent CPU mapping                      */
    uint64_t    size;
    void       *priv_buf;  /**< backend handle (opaque)                     */
    void       *priv_mem;  /**< backend handle (opaque)                     */
} cvp9_export_buffer_t;

/**
 * Allocate a DMA-BUF-exportable, CPU-mappable buffer on the decode GPU.
 * @return CVP9_ERR_UNSUPPORTED if the backend cannot export DMA-BUFs.
 */
cvp9_err_t cvp9_export_buffer_alloc(cvp9_ctx_t *ctx, uint64_t size,
                                    cvp9_export_buffer_t *out);

void cvp9_export_buffer_free(cvp9_ctx_t *ctx, cvp9_export_buffer_t *buf);

/**
 * Arm a one-shot NV12 render target for the NEXT cvp9_decode*() call: the
 * GPU packs the decoded frame as NV12 directly into `target` (a buffer from
 * cvp9_export_buffer_alloc) as part of the decode submission — no CPU copy.
 * Pass NULL to disarm. The target must stay alive until the frame is
 * retrieved with cvp9_get_frame*().
 *
 * @return CVP9_ERR_UNSUPPORTED if the backend cannot write NV12 directly.
 */
cvp9_err_t cvp9_set_render_target(cvp9_ctx_t *ctx, const cvp9_export_buffer_t *target);

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
