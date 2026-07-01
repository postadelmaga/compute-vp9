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
    VkDescriptorSetLayout desc_layout;

    /* Frame buffers (YUV format) */
    VkBuffer         ref_bufs[8];
    VkDeviceMemory   ref_mems[8];
    size_t           ref_sizes[8];
    
    VkBuffer         dst_buf;
    VkDeviceMemory   dst_mem;
    size_t           dst_size;
    
    /* Input buffers */
    VkBuffer         coeff_buf;
    VkDeviceMemory   coeff_mem;
    VkBuffer         mv_buf;
    VkDeviceMemory   mv_mem;

    /* Temporary buffers for neighbor copy */
    VkBuffer         above_buf;
    VkDeviceMemory   above_mem;
    VkBuffer         left_buf;
    VkDeviceMemory   left_mem;

    /* Current frame dimensions */
    uint32_t         width;
    uint32_t         height;
    int64_t          pts;
    int              has_frame;

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

#include "decoder/vp9_parsed_frame.h"

cvp9_err_t vulkan_decode_frame(void *ctx,
                                const vp9_parsed_frame_t *pf,
                                int64_t pts);

cvp9_err_t vulkan_get_frame(void *ctx, cvp9_frame_info_t *info);

#endif /* ENABLE_VULKAN */
