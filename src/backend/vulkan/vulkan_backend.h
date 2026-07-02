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
    int32_t  qstep;       /* AC quantizer step */
    uint32_t coeff_offset;
    uint32_t dst_stride;

    uint32_t dst_offset;
    int32_t  qstep_dc;    /* DC quantizer step (coefficient 0) */
    uint32_t pad2;
    uint32_t pad3;
} gpu_block_data_t;

/* Number of frames that may be in flight on the GPU simultaneously.
 * The CPU entropy decode of frame N+1..N+2 overlaps GPU reconstruction of
 * frame N; delivery latency is at most CVP9_INFLIGHT-1 frames. */
#define CVP9_INFLIGHT 3

/* Per in-flight-frame resources: everything the CPU touches while the GPU
 * may still be working on another frame must be per-slot. */
typedef struct {
    VkCommandBuffer  cmd;        /* persistent, implicitly reset on begin */
    VkFence          fence;
    int              pending;    /* submitted, not yet delivered */
    int64_t          pts;

    /* Host-visible output (I420), persistently mapped, optionally
     * exported as DMA-BUF for zero-copy cross-GPU consumption */
    VkBuffer         output_buf;
    VkDeviceMemory   output_mem;
    void            *output_mapped;
    int              dmabuf_fd;  /* -1 when export unavailable */

    /* Upload staging (host-visible, persistently mapped, grow-only) */
    VkBuffer         coeff_buf;
    VkDeviceMemory   coeff_mem;
    size_t           coeff_cap;
    void            *coeff_mapped;
    VkBuffer         mv_buf;
    VkDeviceMemory   mv_mem;
    size_t           mv_cap;
    void            *mv_mapped;
    VkBuffer         block_buf;
    VkDeviceMemory   block_mem;
    size_t           block_cap;
    void            *block_mapped;

    /* Descriptor sets bound to this slot's buffers */
    VkDescriptorSet  desc_mc;
    VkDescriptorSet  desc_intra;
    VkDescriptorSet  desc_idct;
    VkDescriptorSet  desc_lf;
    VkDescriptorSet  desc_nv12;

    /* Optional caller-owned NV12 buffer written by the GPU at the end of
     * the decode (direct-to-surface, no CPU repack) */
    VkBuffer         nv12_target;
    VkDeviceSize     nv12_target_size;
} vk_frame_slot_t;

typedef struct {
    VkInstance       instance;
    VkPhysicalDevice phys_device;
    VkDevice         device;
    VkQueue          compute_queue;
    uint32_t         compute_family;

    /* External memory (DMA-BUF) export support */
    int              ext_dmabuf;
    PFN_vkGetMemoryFdKHR pfn_get_memory_fd;

    /* Pipelines */
    VkPipeline       pipe_idct;
    VkPipeline       pipe_mc;
    VkPipeline       pipe_intra;
    VkPipeline       pipe_loopfilter;
    VkPipeline       pipe_nv12;
    VkPipelineLayout pipe_layout;
    VkDescriptorPool desc_pool;
    VkDescriptorSetLayout desc_layout_mc;
    VkDescriptorSetLayout desc_layout_buffer;

    /* Frame buffers (YUV format), shared across slots — GPU work is
     * serialized on a single queue so cross-frame hazards are handled with
     * barriers at command-buffer boundaries */
    VkImage          ref_images[8];
    VkDeviceMemory   ref_image_mems[8];
    VkImageView      ref_views[8];
    VkSampler        ref_sampler;

    VkBuffer         dst_buf;
    VkDeviceMemory   dst_mem;
    size_t           dst_size;

    /* Temporary buffers for neighbor copy */
    VkBuffer         above_buf;
    VkDeviceMemory   above_mem;
    VkBuffer         left_buf;
    VkDeviceMemory   left_mem;

    /* Current frame dimensions */
    uint32_t         width;
    uint32_t         height;
    size_t           output_size;

    /* In-flight frame ring */
    vk_frame_slot_t  slots[CVP9_INFLIGHT];
    uint32_t         ring_head;   /* oldest pending slot */
    uint32_t         ring_count;  /* number of pending slots */

    /* One-shot NV12 render target for the next decode */
    VkBuffer         pending_target;
    VkDeviceSize     pending_target_size;

    /* Command infrastructure */
    VkCommandPool    cmd_pool;
} vulkan_ctx_t;

cvp9_err_t vulkan_backend_init(void **ctx);
void       vulkan_backend_destroy(void *ctx);

#include "decoder/vp9_parsed_frame.h"

cvp9_err_t vulkan_decode_frame(void *ctx,
                                const vp9_parsed_frame_t *pf,
                                int64_t pts);

/* wait=0: returns CVP9_ERR_AGAIN if the oldest frame is still on the GPU
 * (unless the pipeline is full, in which case it blocks to guarantee
 * progress). wait=1: always blocks until the oldest frame is done. */
cvp9_err_t vulkan_get_frame(void *ctx, cvp9_frame_info_t *info, int wait);

cvp9_err_t vulkan_get_frame_dmabuf(void *ctx, cvp9_dmabuf_frame_t *out);

cvp9_err_t vulkan_export_buffer_alloc(void *ctx, uint64_t size, cvp9_export_buffer_t *out);
void       vulkan_export_buffer_free(void *ctx, cvp9_export_buffer_t *buf);

/* Arm a one-shot NV12 render target (an export buffer from
 * vulkan_export_buffer_alloc) for the next vulkan_decode_frame: the GPU
 * packs the decoded frame into it directly, no CPU repack needed. */
cvp9_err_t vulkan_set_render_target(void *ctx, const cvp9_export_buffer_t *target);

#endif /* ENABLE_VULKAN */
