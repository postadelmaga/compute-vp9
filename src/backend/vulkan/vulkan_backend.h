/**
 * compute-vp9 — Vulkan compute backend
 *
 * One VkPipeline per VP9 decode stage:
 *   1. vp9_idct        — inverse DCT transform (4x4, 8x8, 16x16, 32x32)
 *   2. vp9_mc          — motion compensation (inter prediction)
 *   3. vp9_intra_pred  — intra prediction modes
 *   4. vp9_loopfilter  — deblocking loop filter
 */
#pragma once

#include "compute_vp9/decoder.h"
#include "decoder/vp9_bitstream.h"

#ifdef ENABLE_VULKAN

#include <vulkan/vulkan.h>

typedef struct {
    VkInstance       instance;
    VkPhysicalDevice phys_device;
    VkDevice         device;
    VkQueue          compute_queue;
    uint32_t         compute_family;

    /* Pipelines */
    VkPipeline       pipe_idct;
    VkPipeline       pipe_mc;
    VkPipeline       pipe_intra;
    VkPipeline       pipe_loopfilter;
    VkPipelineLayout pipe_layout;
    VkDescriptorPool desc_pool;

    /* Output buffer (host-visible) */
    VkBuffer         output_buf;
    VkDeviceMemory   output_mem;
    size_t           output_size;
    void            *output_mapped;

    /* Command infrastructure */
    VkCommandPool    cmd_pool;
    VkFence          fence;
} vulkan_ctx_t;

cvp9_err_t vulkan_backend_init(void **ctx);
void       vulkan_backend_destroy(void *ctx);

cvp9_err_t vulkan_decode_frame(void *ctx,
                                const vp9_frame_header_t *hdr,
                                const uint8_t *data, size_t size,
                                int64_t pts);

cvp9_err_t vulkan_get_frame(void *ctx, cvp9_frame_info_t *info);

#endif /* ENABLE_VULKAN */
