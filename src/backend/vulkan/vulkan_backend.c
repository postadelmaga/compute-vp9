/**
 * compute-vp9 — Vulkan backend implementation stubs
 */
#include "vulkan_backend.h"
#include <stdlib.h>

#ifdef ENABLE_VULKAN

cvp9_err_t vulkan_backend_init(void **ctx)
{
    vulkan_ctx_t *vk = calloc(1, sizeof(*vk));
    if (!vk) return CVP9_ERR_NOMEM;
    
    // Stub implementation: actual Vulkan instance/device initialization goes here.
    *ctx = vk;
    return CVP9_OK;
}

void vulkan_backend_destroy(void *ctx)
{
    if (!ctx) return;
    vulkan_ctx_t *vk = (vulkan_ctx_t *)ctx;
    free(vk);
}

cvp9_err_t vulkan_decode_frame(void *ctx,
                                const vp9_frame_header_t *hdr,
                                const uint8_t *data, size_t size,
                                int64_t pts)
{
    (void)ctx; (void)hdr; (void)data; (void)size; (void)pts;
    return CVP9_OK;
}

cvp9_err_t vulkan_get_frame(void *ctx, cvp9_frame_info_t *info)
{
    (void)ctx; (void)info;
    return CVP9_ERR_UNSUPPORTED; // Frame queue not implemented in stub
}

#endif // ENABLE_VULKAN
