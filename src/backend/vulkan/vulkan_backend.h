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
    uint32_t is_intra;
    uint32_t skip;
    uint32_t block_size;
    uint32_t tx_size;
    
    uint32_t pred_mode;
    int32_t  qstep;
    uint32_t coeff_offset;
    uint32_t dst_stride;
    
    uint32_t dst_offset;
    uint32_t pad1;
    uint32_t pad2;
    uint32_t pad3;
} gpu_block_data_t;

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
    VkDescriptorSetLayout desc_layout_mc;
    VkDescriptorSetLayout desc_layout_buffer;
    VkDescriptorSet  desc_mc;
    VkDescriptorSet  desc_intra;
    VkDescriptorSet  desc_idct;
    VkDescriptorSet  desc_lf;

    /* Frame buffers (YUV format) */
    VkImage          ref_images[8];
    VkDeviceMemory   ref_image_mems[8];
    VkImageView      ref_views[8];
    VkSampler        ref_sampler;
    
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
    
    /* Batch block data buffer */
    VkBuffer         block_buf;
    VkDeviceMemory   block_mem;
    size_t           block_buf_size;

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
    VkCommandBuffer  active_cmd;
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
