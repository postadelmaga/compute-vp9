#ifdef ENABLE_VULKAN
/**
 * compute-vp9 — Vulkan backend implementation
 */
#include "vulkan_backend.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "decoder/vp9_frame.h"

/* Helper: load SPIR-V shader module from the build directory */
static VkShaderModule load_shader_module(VkDevice device, const char *filename)
{
    char path[512];
    snprintf(path, sizeof(path), "%s/%s", SPIRV_OUTPUT_DIR, filename);

    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "[compute-vp9] Failed to open shader: %s\n", path);
        return VK_NULL_HANDLE;
    }

    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);

    uint32_t *code = malloc(size);
    if (!code) {
        fclose(f);
        return VK_NULL_HANDLE;
    }

    if (fread(code, 1, size, f) != (size_t)size) {
        free(code);
        fclose(f);
        return VK_NULL_HANDLE;
    }
    fclose(f);

    VkShaderModuleCreateInfo create_info = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = size,
        .pCode = code
    };

    VkShaderModule module;
    VkResult res = vkCreateShaderModule(device, &create_info, NULL, &module);
    free(code);

    if (res != VK_SUCCESS) {
        fprintf(stderr, "[compute-vp9] Failed to create shader module for: %s (error %d)\n", filename, res);
        return VK_NULL_HANDLE;
    }
    return module;
}

/* Helper: find Vulkan memory type */
static uint32_t find_memory_type(VkPhysicalDevice phys, uint32_t type_filter, VkMemoryPropertyFlags properties)
{
    VkPhysicalDeviceMemoryProperties mem_props;
    vkGetPhysicalDeviceMemoryProperties(phys, &mem_props);
    for (uint32_t i = 0; i < mem_props.memoryTypeCount; i++) {
        if ((type_filter & (1 << i)) && (mem_props.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
    return (uint32_t)-1;
}

/* Helper: create Vulkan buffer and allocate memory */
static VkResult create_buffer(VkDevice device, VkPhysicalDevice phys, VkDeviceSize size,
                             VkBufferUsageFlags usage, VkMemoryPropertyFlags properties,
                             VkBuffer *buffer, VkDeviceMemory *memory)
{
    VkBufferCreateInfo buffer_info = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = size,
        .usage = usage,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE
    };
    
    VkResult res = vkCreateBuffer(device, &buffer_info, NULL, buffer);
    if (res != VK_SUCCESS) return res;
    
    VkMemoryRequirements mem_reqs;
    vkGetBufferMemoryRequirements(device, *buffer, &mem_reqs);
    
    VkMemoryAllocateInfo alloc_info = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = mem_reqs.size,
        .memoryTypeIndex = find_memory_type(phys, mem_reqs.memoryTypeBits, properties)
    };
    
    res = vkAllocateMemory(device, &alloc_info, NULL, memory);
    if (res != VK_SUCCESS) return res;
    
    return vkBindBufferMemory(device, *buffer, *memory, 0);
}

/* Helper: allocate a descriptor set */
static VkDescriptorSet allocate_desc_set(VkDevice device, VkDescriptorPool pool, VkDescriptorSetLayout layout)
{
    VkDescriptorSetAllocateInfo alloc_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &layout
    };
    VkDescriptorSet set;
    if (vkAllocateDescriptorSets(device, &alloc_info, &set) != VK_SUCCESS) {
        return VK_NULL_HANDLE;
    }
    return set;
}

/* Helper: bind buffer to descriptor set */
static void bind_buffer_to_descriptor(VkDevice device, VkDescriptorSet set, uint32_t binding, VkBuffer buffer, VkDeviceSize size)
{
    VkDescriptorBufferInfo buffer_info = {
        .buffer = buffer,
        .offset = 0,
        .range = size
    };
    VkWriteDescriptorSet write = {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = set,
        .dstBinding = binding,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .pBufferInfo = &buffer_info
    };
    vkUpdateDescriptorSets(device, 1, &write, 0, NULL);
}

/* Ensure buffers are allocated and of the correct size */
static void ensure_buffers(vulkan_ctx_t *vk, uint32_t width, uint32_t height)
{
    if (vk->width == width && vk->height == height && vk->dst_buf != VK_NULL_HANDLE) {
        return;
    }
    
    /* Clean up old buffers */
    if (vk->dst_buf != VK_NULL_HANDLE) {
        vkDestroyBuffer(vk->device, vk->dst_buf, NULL);
        vkFreeMemory(vk->device, vk->dst_mem, NULL);
    }
    for (int i = 0; i < 8; i++) {
        if (vk->ref_bufs[i] != VK_NULL_HANDLE) {
            vkDestroyBuffer(vk->device, vk->ref_bufs[i], NULL);
            vkFreeMemory(vk->device, vk->ref_mems[i], NULL);
        }
    }
    if (vk->above_buf != VK_NULL_HANDLE) {
        vkDestroyBuffer(vk->device, vk->above_buf, NULL);
        vkFreeMemory(vk->device, vk->above_mem, NULL);
    }
    if (vk->left_buf != VK_NULL_HANDLE) {
        vkDestroyBuffer(vk->device, vk->left_buf, NULL);
        vkFreeMemory(vk->device, vk->left_mem, NULL);
    }
    
    vk->width = width;
    vk->height = height;
    
    size_t yuv_size = width * height + 2 * ((width + 1) / 2) * ((height + 1) / 2);
    vk->dst_size = yuv_size;

    create_buffer(vk->device, vk->phys_device, yuv_size,
                  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                  VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                  &vk->dst_buf, &vk->dst_mem);
                  
    for (int i = 0; i < 8; i++) {
        vk->ref_sizes[i] = yuv_size;
        create_buffer(vk->device, vk->phys_device, yuv_size,
                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                      &vk->ref_bufs[i], &vk->ref_mems[i]);
    }

    create_buffer(vk->device, vk->phys_device, 64, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                  VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                  &vk->above_buf, &vk->above_mem);
    create_buffer(vk->device, vk->phys_device, 64, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                  VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                  &vk->left_buf, &vk->left_mem);
    
    if (vk->output_buf != VK_NULL_HANDLE) {
        vkUnmapMemory(vk->device, vk->output_mem);
        vkDestroyBuffer(vk->device, vk->output_buf, NULL);
        vkFreeMemory(vk->device, vk->output_mem, NULL);
    }
    vk->output_size = yuv_size;
    create_buffer(vk->device, vk->phys_device, yuv_size,
                  VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                  &vk->output_buf, &vk->output_mem);
    vkMapMemory(vk->device, vk->output_mem, 0, yuv_size, 0, &vk->output_mapped);
}

cvp9_err_t vulkan_backend_init(void **ctx)
{
    vulkan_ctx_t *vk = calloc(1, sizeof(*vk));
    if (!vk) return CVP9_ERR_NOMEM;

    VkApplicationInfo app_info = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "compute-vp9",
        .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
        .pEngineName = "No Engine",
        .engineVersion = VK_MAKE_VERSION(1, 0, 0),
        .apiVersion = VK_API_VERSION_1_0
    };

    VkInstanceCreateInfo create_info = {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &app_info
    };

    VkResult res = vkCreateInstance(&create_info, NULL, &vk->instance);
    if (res != VK_SUCCESS) {
        free(vk);
        return CVP9_ERR_GPU;
    }

    uint32_t device_count = 0;
    vkEnumeratePhysicalDevices(vk->instance, &device_count, NULL);
    if (device_count == 0) {
        vkDestroyInstance(vk->instance, NULL);
        free(vk);
        return CVP9_ERR_GPU;
    }

    VkPhysicalDevice *devices = malloc(sizeof(VkPhysicalDevice) * device_count);
    vkEnumeratePhysicalDevices(vk->instance, &device_count, devices);
    vk->phys_device = devices[0];
    free(devices);

    uint32_t queue_family_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(vk->phys_device, &queue_family_count, NULL);
    VkQueueFamilyProperties *queue_families = malloc(sizeof(VkQueueFamilyProperties) * queue_family_count);
    vkGetPhysicalDeviceQueueFamilyProperties(vk->phys_device, &queue_family_count, queue_families);
    
    vk->compute_family = (uint32_t)-1;
    for (uint32_t i = 0; i < queue_family_count; i++) {
        if (queue_families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            vk->compute_family = i;
            break;
        }
    }
    free(queue_families);
    if (vk->compute_family == (uint32_t)-1) {
        vkDestroyInstance(vk->instance, NULL);
        free(vk);
        return CVP9_ERR_GPU;
    }

    float queue_priority = 1.0f;
    VkDeviceQueueCreateInfo queue_create_info = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = vk->compute_family,
        .queueCount = 1,
        .pQueuePriorities = &queue_priority
    };

    VkDeviceCreateInfo device_create_info = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &queue_create_info
    };

    res = vkCreateDevice(vk->phys_device, &device_create_info, NULL, &vk->device);
    if (res != VK_SUCCESS) {
        vkDestroyInstance(vk->instance, NULL);
        free(vk);
        return CVP9_ERR_GPU;
    }

    vkGetDeviceQueue(vk->device, vk->compute_family, 0, &vk->compute_queue);

    /* Setup Descriptor Layout */
    VkDescriptorSetLayoutBinding bindings[3] = {
        {
            .binding = 0,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT
        },
        {
            .binding = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT
        },
        {
            .binding = 2,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT
        }
    };
    
    VkDescriptorSetLayoutCreateInfo layout_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 3,
        .pBindings = bindings
    };
    
    vkCreateDescriptorSetLayout(vk->device, &layout_info, NULL, &vk->desc_layout);

    /* Setup Pipeline Layout */
    VkPushConstantRange push_constant_range = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = 32
    };

    VkPipelineLayoutCreateInfo pipeline_layout_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &vk->desc_layout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &push_constant_range
    };

    vkCreatePipelineLayout(vk->device, &pipeline_layout_info, NULL, &vk->pipe_layout);

    /* Create Compute Pipelines */
    VkShaderModule mod_idct = load_shader_module(vk->device, "vp9_idct.comp.glsl.spv");
    VkShaderModule mod_mc = load_shader_module(vk->device, "vp9_mc.comp.glsl.spv");
    VkShaderModule mod_intra = load_shader_module(vk->device, "vp9_intra_pred.comp.glsl.spv");
    VkShaderModule mod_loopfilter = load_shader_module(vk->device, "vp9_loopfilter.comp.glsl.spv");

    if (mod_idct == VK_NULL_HANDLE || mod_mc == VK_NULL_HANDLE ||
        mod_intra == VK_NULL_HANDLE || mod_loopfilter == VK_NULL_HANDLE) {
        if (mod_idct) vkDestroyShaderModule(vk->device, mod_idct, NULL);
        if (mod_mc) vkDestroyShaderModule(vk->device, mod_mc, NULL);
        if (mod_intra) vkDestroyShaderModule(vk->device, mod_intra, NULL);
        if (mod_loopfilter) vkDestroyShaderModule(vk->device, mod_loopfilter, NULL);
        vkDestroyPipelineLayout(vk->device, vk->pipe_layout, NULL);
        vkDestroyDescriptorSetLayout(vk->device, vk->desc_layout, NULL);
        vkDestroyDevice(vk->device, NULL);
        vkDestroyInstance(vk->instance, NULL);
        free(vk);
        return CVP9_ERR_GPU;
    }

    VkComputePipelineCreateInfo pipeline_create_info = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .layout = vk->pipe_layout,
        .stage = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT,
            .pName = "main"
        }
    };

    pipeline_create_info.stage.module = mod_idct;
    vkCreateComputePipelines(vk->device, VK_NULL_HANDLE, 1, &pipeline_create_info, NULL, &vk->pipe_idct);

    pipeline_create_info.stage.module = mod_mc;
    vkCreateComputePipelines(vk->device, VK_NULL_HANDLE, 1, &pipeline_create_info, NULL, &vk->pipe_mc);

    pipeline_create_info.stage.module = mod_intra;
    vkCreateComputePipelines(vk->device, VK_NULL_HANDLE, 1, &pipeline_create_info, NULL, &vk->pipe_intra);

    pipeline_create_info.stage.module = mod_loopfilter;
    vkCreateComputePipelines(vk->device, VK_NULL_HANDLE, 1, &pipeline_create_info, NULL, &vk->pipe_loopfilter);
    
    vkDestroyShaderModule(vk->device, mod_idct, NULL);
    vkDestroyShaderModule(vk->device, mod_mc, NULL);
    vkDestroyShaderModule(vk->device, mod_intra, NULL);
    vkDestroyShaderModule(vk->device, mod_loopfilter, NULL);

    /* Descriptor Pool */
    VkDescriptorPoolSize pool_size = {
        .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 64
    };

    VkDescriptorPoolCreateInfo pool_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = 32,
        .poolSizeCount = 1,
        .pPoolSizes = &pool_size
    };

    vkCreateDescriptorPool(vk->device, &pool_info, NULL, &vk->desc_pool);

    /* Allocate persistent descriptor sets */
    vk->desc_mc = allocate_desc_set(vk->device, vk->desc_pool, vk->desc_layout);
    vk->desc_intra = allocate_desc_set(vk->device, vk->desc_pool, vk->desc_layout);
    vk->desc_idct = allocate_desc_set(vk->device, vk->desc_pool, vk->desc_layout);
    vk->desc_lf = allocate_desc_set(vk->device, vk->desc_pool, vk->desc_layout);

    if (vk->desc_mc == VK_NULL_HANDLE || vk->desc_intra == VK_NULL_HANDLE ||
        vk->desc_idct == VK_NULL_HANDLE || vk->desc_lf == VK_NULL_HANDLE) {
        vulkan_backend_destroy(vk);
        return CVP9_ERR_GPU;
    }

    /* Infrastructure */
    VkCommandPoolCreateInfo command_pool_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .queueFamilyIndex = vk->compute_family,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT
    };

    vkCreateCommandPool(vk->device, &command_pool_info, NULL, &vk->cmd_pool);

    VkFenceCreateInfo fence_info = {
        .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO
    };

    vkCreateFence(vk->device, &fence_info, NULL, &vk->fence);

    *ctx = vk;
    return CVP9_OK;
}

void vulkan_backend_destroy(void *ctx)
{
    if (!ctx) return;
    vulkan_ctx_t *vk = (vulkan_ctx_t *)ctx;

    if (vk->device) {
        vkDestroyFence(vk->device, vk->fence, NULL);
        vkDestroyCommandPool(vk->device, vk->cmd_pool, NULL);
        vkDestroyDescriptorPool(vk->device, vk->desc_pool, NULL);
        
        vkDestroyPipeline(vk->device, vk->pipe_idct, NULL);
        vkDestroyPipeline(vk->device, vk->pipe_mc, NULL);
        vkDestroyPipeline(vk->device, vk->pipe_intra, NULL);
        vkDestroyPipeline(vk->device, vk->pipe_loopfilter, NULL);
        vkDestroyPipelineLayout(vk->device, vk->pipe_layout, NULL);
        
        if (vk->desc_layout != VK_NULL_HANDLE) {
            vkDestroyDescriptorSetLayout(vk->device, vk->desc_layout, NULL);
        }

        if (vk->output_mapped) {
            vkUnmapMemory(vk->device, vk->output_mem);
        }
        if (vk->output_buf) {
            vkDestroyBuffer(vk->device, vk->output_buf, NULL);
            vkFreeMemory(vk->device, vk->output_mem, NULL);
        }

        if (vk->above_buf != VK_NULL_HANDLE) {
            vkDestroyBuffer(vk->device, vk->above_buf, NULL);
            vkFreeMemory(vk->device, vk->above_mem, NULL);
        }
        if (vk->left_buf != VK_NULL_HANDLE) {
            vkDestroyBuffer(vk->device, vk->left_buf, NULL);
            vkFreeMemory(vk->device, vk->left_mem, NULL);
        }
        if (vk->coeff_buf != VK_NULL_HANDLE) {
            vkDestroyBuffer(vk->device, vk->coeff_buf, NULL);
            vkFreeMemory(vk->device, vk->coeff_mem, NULL);
        }
        if (vk->mv_buf != VK_NULL_HANDLE) {
            vkDestroyBuffer(vk->device, vk->mv_buf, NULL);
            vkFreeMemory(vk->device, vk->mv_mem, NULL);
        }

        for (int i = 0; i < 8; i++) {
            if (vk->ref_bufs[i] != VK_NULL_HANDLE) {
                vkDestroyBuffer(vk->device, vk->ref_bufs[i], NULL);
                vkFreeMemory(vk->device, vk->ref_mems[i], NULL);
            }
        }
        if (vk->dst_buf != VK_NULL_HANDLE) {
            vkDestroyBuffer(vk->device, vk->dst_buf, NULL);
            vkFreeMemory(vk->device, vk->dst_mem, NULL);
        }

        vkDestroyDevice(vk->device, NULL);
    }
    if (vk->instance) {
        vkDestroyInstance(vk->instance, NULL);
    }
    free(vk);
}

cvp9_err_t vulkan_decode_frame(void *ctx,
                                const vp9_parsed_frame_t *pf,
                                int64_t pts)
{
    vulkan_ctx_t *vk = (vulkan_ctx_t *)ctx;
    if (!vk || !pf) return CVP9_ERR_INVALID_DATA;

    /* Wait for previous frame's command buffer to finish if it hasn't already */
    if (vk->active_cmd != VK_NULL_HANDLE) {
        vkWaitForFences(vk->device, 1, &vk->fence, VK_TRUE, UINT64_MAX);
        vkFreeCommandBuffers(vk->device, vk->cmd_pool, 1, &vk->active_cmd);
        vk->active_cmd = VK_NULL_HANDLE;
    }

    ensure_buffers(vk, pf->hdr.width, pf->hdr.height);

    vk->pts = pts;
    vk->has_frame = 1;

    /* Upload coefficients */
    size_t coeff_size = pf->num_coeffs * sizeof(int16_t);
    if (coeff_size == 0) coeff_size = 256;
    if (vk->coeff_buf != VK_NULL_HANDLE) {
        vkDestroyBuffer(vk->device, vk->coeff_buf, NULL);
        vkFreeMemory(vk->device, vk->coeff_mem, NULL);
    }
    create_buffer(vk->device, vk->phys_device, coeff_size,
                  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                  &vk->coeff_buf, &vk->coeff_mem);
    if (pf->num_coeffs > 0) {
        void *mapped;
        vkMapMemory(vk->device, vk->coeff_mem, 0, coeff_size, 0, &mapped);
        memcpy(mapped, pf->coeffs, coeff_size);
        vkUnmapMemory(vk->device, vk->coeff_mem);
    }

    /* Upload motion vectors */
    size_t mv_size = pf->mv_grid_width * pf->mv_grid_height * sizeof(cvp9_mv_t);
    if (mv_size == 0) mv_size = 256;
    if (vk->mv_buf != VK_NULL_HANDLE) {
        vkDestroyBuffer(vk->device, vk->mv_buf, NULL);
        vkFreeMemory(vk->device, vk->mv_mem, NULL);
    }
    create_buffer(vk->device, vk->phys_device, mv_size,
                  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                  &vk->mv_buf, &vk->mv_mem);
    if (pf->mv_grid_width > 0 && pf->mv_grid_height > 0) {
        void *mapped;
        vkMapMemory(vk->device, vk->mv_mem, 0, mv_size, 0, &mapped);
        memcpy(mapped, pf->mv_grid, mv_size);
        vkUnmapMemory(vk->device, vk->mv_mem);
    }

    /* Update descriptor sets once per frame */
    bind_buffer_to_descriptor(vk->device, vk->desc_mc, 0, vk->ref_bufs[0], vk->dst_size);
    bind_buffer_to_descriptor(vk->device, vk->desc_mc, 1, vk->dst_buf, vk->dst_size);
    bind_buffer_to_descriptor(vk->device, vk->desc_mc, 2, vk->mv_buf, mv_size);

    bind_buffer_to_descriptor(vk->device, vk->desc_intra, 0, vk->dst_buf, vk->dst_size);
    bind_buffer_to_descriptor(vk->device, vk->desc_intra, 1, vk->dst_buf, vk->dst_size);
    bind_buffer_to_descriptor(vk->device, vk->desc_intra, 2, vk->dst_buf, vk->dst_size);

    bind_buffer_to_descriptor(vk->device, vk->desc_idct, 0, vk->coeff_buf, coeff_size);
    bind_buffer_to_descriptor(vk->device, vk->desc_idct, 1, vk->dst_buf, vk->dst_size);
    bind_buffer_to_descriptor(vk->device, vk->desc_idct, 2, vk->dst_buf, vk->dst_size);

    bind_buffer_to_descriptor(vk->device, vk->desc_lf, 0, vk->dst_buf, vk->dst_size);
    bind_buffer_to_descriptor(vk->device, vk->desc_lf, 1, vk->dst_buf, vk->dst_size);
    bind_buffer_to_descriptor(vk->device, vk->desc_lf, 2, vk->dst_buf, vk->dst_size);

    /* Record command buffer */
    VkCommandBufferAllocateInfo cmd_alloc = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = vk->cmd_pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1
    };
    VkCommandBuffer cmd;
    vkAllocateCommandBuffers(vk->device, &cmd_alloc, &cmd);
    
    VkCommandBufferBeginInfo begin_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
    };
    vkBeginCommandBuffer(cmd, &begin_info);

    VkMemoryBarrier mem_barrier = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT
    };

    /* 1. Motion Compensation (Inter prediction) */
    struct {
        uint32_t width;
        uint32_t height;
        uint32_t block_size;
        uint32_t filter_type;
    } pc_mc = {
        .width = vk->width,
        .height = vk->height,
        .block_size = 4,
        .filter_type = 0
    };

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, vk->pipe_mc);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, vk->pipe_layout, 0, 1, &vk->desc_mc, 0, NULL);
    vkCmdPushConstants(cmd, vk->pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc_mc), &pc_mc);
    vkCmdDispatch(cmd, (vk->width + 7) / 8, (vk->height + 7) / 8, 1);

    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 1, &mem_barrier, 0, NULL, 0, NULL);

    /* 2. Block-by-block Intra Prediction and IDCT Residual Addition */
    int need_barrier_before_intra = 0;
    for (uint32_t i = 0; i < pf->num_blocks; i++) {
        const vp9_macroblock_info_t *block = &pf->blocks[i];
        uint32_t block_size = block->width;

        if (block->is_intra) {
            /* Synchronize if any IDCT ran since the last barrier, ensuring reconstructed neighbors are visible */
            if (need_barrier_before_intra) {
                vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                     0, 1, &mem_barrier, 0, NULL, 0, NULL);
                need_barrier_before_intra = 0;
            }

            /* Dispatch Intra prediction */
            struct {
                uint32_t block_size;
                uint32_t pred_mode;
                uint32_t dst_stride;
                uint32_t dst_offset;
            } pc_intra = {
                .block_size = block_size,
                .pred_mode = block->y_mode,
                .dst_stride = vk->width,
                .dst_offset = block->y * vk->width + block->x
            };

            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, vk->pipe_intra);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, vk->pipe_layout, 0, 1, &vk->desc_intra, 0, NULL);
            vkCmdPushConstants(cmd, vk->pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc_intra), &pc_intra);
            vkCmdDispatch(cmd, (block_size + 7) / 8, (block_size + 7) / 8, 1);
            
            /* Must synchronize intra pred write before IDCT reads it for the same block */
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 0, 1, &mem_barrier, 0, NULL, 0, NULL);
        }

        /* Residual Addition (IDCT) */
        uint32_t tx_size = 1 << (block->tx_size + 2);

        struct {
            uint32_t block_size;
            int      qstep;
            uint32_t block_offset;
            uint32_t dst_stride;
            uint32_t dst_offset;
        } pc_idct = {
            .block_size = tx_size,
            .qstep = 128, /* quantization scale step */
            .block_offset = block->coeff_offset,
            .dst_stride = vk->width,
            .dst_offset = block->y * vk->width + block->x
        };

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, vk->pipe_idct);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, vk->pipe_layout, 0, 1, &vk->desc_idct, 0, NULL);
        vkCmdPushConstants(cmd, vk->pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc_idct), &pc_idct);
        vkCmdDispatch(cmd, 1, 1, 1);

        /* Set flag to require barrier before any future intra block reads */
        need_barrier_before_intra = 1;
    }

    /* Ensure all block reconstruction writes are completed before starting loop filtering */
    if (need_barrier_before_intra) {
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0, 1, &mem_barrier, 0, NULL, 0, NULL);
    }

    /* 3. Deblocking Loop Filter */
    struct {
        uint32_t frame_width;
        uint32_t frame_height;
        uint32_t stride;
        uint32_t filter_level;
        uint32_t sharpness;
        uint32_t pass;
    } pc_lf = {
        .frame_width = vk->width,
        .frame_height = vk->height,
        .stride = vk->width,
        .filter_level = 32,
        .sharpness = 0,
        .pass = 0 /* Horizontal pass */
    };

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, vk->pipe_loopfilter);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, vk->pipe_layout, 0, 1, &vk->desc_lf, 0, NULL);
    
    /* Horizontal pass */
    vkCmdPushConstants(cmd, vk->pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc_lf), &pc_lf);
    vkCmdDispatch(cmd, (vk->width + 7) / 8, 1, 1);

    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 1, &mem_barrier, 0, NULL, 0, NULL);

    /* Vertical pass */
    pc_lf.pass = 1;
    vkCmdPushConstants(cmd, vk->pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc_lf), &pc_lf);
    vkCmdDispatch(cmd, (vk->height + 7) / 8, 1, 1);

    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 1, &mem_barrier, 0, NULL, 0, NULL);

    /* Copy destination back to staging output buffer */
    VkBufferCopy copy_region = {
        .srcOffset = 0,
        .dstOffset = 0,
        .size = vk->dst_size
    };
    vkCmdCopyBuffer(cmd, vk->dst_buf, vk->output_buf, 1, &copy_region);

    /* Update Reference frame slot 0 for subsequent inter frames */
    VkBufferCopy ref_copy = {
        .srcOffset = 0,
        .dstOffset = 0,
        .size = vk->dst_size
    };
    vkCmdCopyBuffer(cmd, vk->dst_buf, vk->ref_bufs[0], 1, &ref_copy);

    vkEndCommandBuffer(cmd);

    /* Submit asynchronously */
    VkSubmitInfo submit_info = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &cmd
    };
    
    vkResetFences(vk->device, 1, &vk->fence);
    VkResult queue_res = vkQueueSubmit(vk->compute_queue, 1, &submit_info, vk->fence);
    if (queue_res != VK_SUCCESS) {
        vkFreeCommandBuffers(vk->device, vk->cmd_pool, 1, &cmd);
        return CVP9_ERR_GPU;
    }
    
    vk->active_cmd = cmd;
    return CVP9_OK;
}

cvp9_err_t vulkan_get_frame(void *ctx, cvp9_frame_info_t *info)
{
    vulkan_ctx_t *vk = (vulkan_ctx_t *)ctx;
    if (!vk || !vk->has_frame) return CVP9_ERR_UNSUPPORTED;
    
    /* Synchronize and clean up the active command buffer */
    if (vk->active_cmd != VK_NULL_HANDLE) {
        vkWaitForFences(vk->device, 1, &vk->fence, VK_TRUE, UINT64_MAX);
        vkFreeCommandBuffers(vk->device, vk->cmd_pool, 1, &vk->active_cmd);
        vk->active_cmd = VK_NULL_HANDLE;
    }

    cvp9_frame_alloc(info, vk->width, vk->height);
    info->pts = vk->pts;
    
    uint8_t *src = vk->output_mapped;
    uint32_t w = vk->width;
    uint32_t h = vk->height;
    uint32_t size_y = w * h;
    uint32_t size_uv = ((w + 1) / 2) * ((h + 1) / 2);
    
    /* Copy Y plane */
    for (uint32_t r = 0; r < h; r++) {
        memcpy(info->plane_y + r * info->stride_y, src + r * w, w);
    }
    /* Copy U plane */
    uint32_t ch = (h + 1) / 2;
    uint32_t cw = (w + 1) / 2;
    for (uint32_t r = 0; r < ch; r++) {
        memcpy(info->plane_u + r * info->stride_uv, src + size_y + r * cw, cw);
    }
    /* Copy V plane */
    for (uint32_t r = 0; r < ch; r++) {
        memcpy(info->plane_v + r * info->stride_uv, src + size_y + size_uv + r * cw, cw);
    }
    
    vk->has_frame = 0;
    return CVP9_OK;
}

#endif /* ENABLE_VULKAN */
