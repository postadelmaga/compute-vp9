#ifdef ENABLE_VULKAN
/**
 * compute-vp9 — Vulkan backend implementation
 */
#include "vulkan_backend.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
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


static VkResult create_image(VkDevice device, VkPhysicalDevice phys, uint32_t width, uint32_t height, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage *image, VkDeviceMemory *imageMemory) {
    VkImageCreateInfo imageInfo = {0};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.extent.width = width;
    imageInfo.extent.height = height;
    imageInfo.extent.depth = 1;
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.format = format;
    imageInfo.tiling = tiling;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageInfo.usage = usage;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateImage(device, &imageInfo, NULL, image) != VK_SUCCESS) return VK_ERROR_OUT_OF_DEVICE_MEMORY;

    VkMemoryRequirements memRequirements;
    vkGetImageMemoryRequirements(device, *image, &memRequirements);

    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(phys, &memProperties);
    uint32_t memoryTypeIndex = (uint32_t)-1;
    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((memRequirements.memoryTypeBits & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            memoryTypeIndex = i;
            break;
        }
    }

    VkMemoryAllocateInfo allocInfo = {0};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = memoryTypeIndex;

    if (vkAllocateMemory(device, &allocInfo, NULL, imageMemory) != VK_SUCCESS) return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    vkBindImageMemory(device, *image, *imageMemory, 0);
    return VK_SUCCESS;
}

static VkImageView create_image_view(VkDevice device, VkImage image, VkFormat format) {
    VkImageViewCreateInfo viewInfo = {0};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = image;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = format;
    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = 1;

    VkImageView imageView;
    if (vkCreateImageView(device, &viewInfo, NULL, &imageView) != VK_SUCCESS) return VK_NULL_HANDLE;
    return imageView;
}

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
    
    uint32_t mem_type = find_memory_type(phys, mem_reqs.memoryTypeBits, properties);
    if (mem_type == (uint32_t)-1) {
        fprintf(stderr, "[compute-vp9] find_memory_type failed! size: %lu, filter: 0x%X, properties: 0x%X\n",
                (unsigned long)size, mem_reqs.memoryTypeBits, properties);
        vkDestroyBuffer(device, *buffer, NULL);
        return VK_ERROR_FEATURE_NOT_PRESENT;
    }
    
    VkMemoryAllocateInfo alloc_info = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = mem_reqs.size,
        .memoryTypeIndex = mem_type
    };
    
    res = vkAllocateMemory(device, &alloc_info, NULL, memory);
    if (res != VK_SUCCESS) {
        fprintf(stderr, "[compute-vp9] vkAllocateMemory failed! res: %d, size: %lu, req_size: %lu, alignment: %lu, mem_type: %u\n",
                res, (unsigned long)size, (unsigned long)mem_reqs.size, (unsigned long)mem_reqs.alignment, mem_type);
        vkDestroyBuffer(device, *buffer, NULL);
        return res;
    }
    
    res = vkBindBufferMemory(device, *buffer, *memory, 0);
    if (res != VK_SUCCESS) {
        vkFreeMemory(device, *memory, NULL);
        vkDestroyBuffer(device, *buffer, NULL);
        return res;
    }
    return VK_SUCCESS;
}

/* Helper: allocate a descriptor set */
static void bind_image_to_descriptor(VkDevice device, VkDescriptorSet set, uint32_t binding, VkImageView view, VkSampler sampler)
{
    VkDescriptorImageInfo image_info = {
        .sampler = sampler,
        .imageView = view,
        .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
    };
    VkWriteDescriptorSet write = {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = set,
        .dstBinding = binding,
        .dstArrayElement = 0,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        .pImageInfo = &image_info
    };
    vkUpdateDescriptorSets(device, 1, &write, 0, NULL);
}

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
static VkResult ensure_buffers(vulkan_ctx_t *vk, uint32_t width, uint32_t height)
{
    if (vk->width == width && vk->height == height && vk->dst_buf != VK_NULL_HANDLE) {
        return VK_SUCCESS;
    }
    
    /* Clean up old buffers */
    if (vk->dst_buf != VK_NULL_HANDLE) {
        vkDestroyBuffer(vk->device, vk->dst_buf, NULL);
        vkFreeMemory(vk->device, vk->dst_mem, NULL);
        vk->dst_buf = VK_NULL_HANDLE;
    }
    for (int i = 0; i < 8; i++) {
        if (vk->ref_images[i] != VK_NULL_HANDLE) {
            vkDestroyImageView(vk->device, vk->ref_views[i], NULL);
            vkDestroyImage(vk->device, vk->ref_images[i], NULL);
            vkFreeMemory(vk->device, vk->ref_image_mems[i], NULL);
            vk->ref_images[i] = VK_NULL_HANDLE;
        }
    }
    if (vk->ref_sampler != VK_NULL_HANDLE) {
        vkDestroySampler(vk->device, vk->ref_sampler, NULL);
        vk->ref_sampler = VK_NULL_HANDLE;
    }
    
    if (vk->above_buf != VK_NULL_HANDLE) {
        vkDestroyBuffer(vk->device, vk->above_buf, NULL);
        vkFreeMemory(vk->device, vk->above_mem, NULL);
        vk->above_buf = VK_NULL_HANDLE;
    }
    if (vk->left_buf != VK_NULL_HANDLE) {
        vkDestroyBuffer(vk->device, vk->left_buf, NULL);
        vkFreeMemory(vk->device, vk->left_mem, NULL);
        vk->left_buf = VK_NULL_HANDLE;
    }
    
    vk->width = width;
    vk->height = height;
    
    size_t yuv_size = width * height + 2 * ((width + 1) / 2) * ((height + 1) / 2);
    vk->dst_size = yuv_size;

    VkResult res = create_buffer(vk->device, vk->phys_device, yuv_size,
                  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                  VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                  &vk->dst_buf, &vk->dst_mem);
    if (res != VK_SUCCESS) {
        fprintf(stderr, "[compute-vp9] Failed to create dst_buf: %d\n", res);
        return res;
    }
                  
    for (int i = 0; i < 8; i++) {
        res = create_image(vk->device, vk->phys_device, vk->width, vk->height, VK_FORMAT_R8_UNORM, VK_IMAGE_TILING_OPTIMAL,
                           VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                           VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                           &vk->ref_images[i], &vk->ref_image_mems[i]);
        if (res != VK_SUCCESS) {
            fprintf(stderr, "Failed to create ref image %d: %d\n", i, res);
            return CVP9_ERR_GPU;
        }
        vk->ref_views[i] = create_image_view(vk->device, vk->ref_images[i], VK_FORMAT_R8_UNORM);
    }
    VkSamplerCreateInfo samplerInfo = {0};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.anisotropyEnable = VK_FALSE;
    samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
    samplerInfo.unnormalizedCoordinates = VK_FALSE;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    VkResult samplerRes = vkCreateSampler(vk->device, &samplerInfo, NULL, &vk->ref_sampler);
    if (samplerRes != VK_SUCCESS) {
        fprintf(stderr, "Failed to create sampler: %d\n", samplerRes);
        return CVP9_ERR_GPU;
    }

    res = create_buffer(vk->device, vk->phys_device, 64, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                  VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                  &vk->above_buf, &vk->above_mem);
    if (res != VK_SUCCESS) {
        fprintf(stderr, "[compute-vp9] Failed to create above_buf: %d\n", res);
        return res;
    }

    res = create_buffer(vk->device, vk->phys_device, 64, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                  VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                  &vk->left_buf, &vk->left_mem);
    if (res != VK_SUCCESS) {
        fprintf(stderr, "[compute-vp9] Failed to create left_buf: %d\n", res);
        return res;
    }
    
    if (vk->output_buf != VK_NULL_HANDLE) {
        vkUnmapMemory(vk->device, vk->output_mem);
        vkDestroyBuffer(vk->device, vk->output_buf, NULL);
        vkFreeMemory(vk->device, vk->output_mem, NULL);
        vk->output_buf = VK_NULL_HANDLE;
    }
    vk->output_size = yuv_size;
    res = create_buffer(vk->device, vk->phys_device, yuv_size,
                  VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                  &vk->output_buf, &vk->output_mem);
    if (res != VK_SUCCESS) {
        fprintf(stderr, "[compute-vp9] Failed to create output_buf: %d\n", res);
        return res;
    }
    res = vkMapMemory(vk->device, vk->output_mem, 0, yuv_size, 0, &vk->output_mapped);
    if (res != VK_SUCCESS) {
        fprintf(stderr, "[compute-vp9] Failed to map output_mem: %d\n", res);
        return res;
    }

    return VK_SUCCESS;
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

    const char *preferred_vendor = getenv("CVP9_GPU_VENDOR");
    uint32_t selected_idx = 0;

    printf("[compute-vp9] Available Vulkan GPUs:\n");
    for (uint32_t i = 0; i < device_count; i++) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(devices[i], &props);

        const char *vendor_name = "Unknown";
        if (props.vendorID == 0x8086) vendor_name = "Intel";
        else if (props.vendorID == 0x10DE) vendor_name = "NVIDIA";
        else if (props.vendorID == 0x1002) vendor_name = "AMD";

        printf("  [%u] %s (Vendor: %s, ID: 0x%04X, Type: %d)\n",
               i, props.deviceName, vendor_name, props.vendorID, props.deviceType);

        if (preferred_vendor) {
            if (strcasecmp(preferred_vendor, "intel") == 0 && props.vendorID == 0x8086) {
                selected_idx = i;
            } else if (strcasecmp(preferred_vendor, "nvidia") == 0 && props.vendorID == 0x10DE) {
                selected_idx = i;
            } else if (strcasecmp(preferred_vendor, "amd") == 0 && props.vendorID == 0x1002) {
                selected_idx = i;
            }
        }
    }

    vk->phys_device = devices[selected_idx];
    VkPhysicalDeviceProperties selected_props;
    vkGetPhysicalDeviceProperties(vk->phys_device, &selected_props);
    printf("[compute-vp9] Selected Vulkan GPU: %s\n", selected_props.deviceName);

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

    VkPhysicalDevice16BitStorageFeatures features16 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES,
        .storageBuffer16BitAccess = VK_TRUE
    };

    VkPhysicalDevice8BitStorageFeatures features8 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES,
        .pNext = &features16,
        .storageBuffer8BitAccess = VK_TRUE
    };

    VkDeviceCreateInfo device_create_info = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .pNext = &features8,
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

    /* Descriptor Set Layout for MC (binding 0 is Image) */
    VkDescriptorSetLayoutBinding bindings_mc[3] = {
        {
            .binding = 0,
            .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
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
    
    VkDescriptorSetLayoutCreateInfo layout_info_mc = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 3,
        .pBindings = bindings_mc
    };
    
    vkCreateDescriptorSetLayout(vk->device, &layout_info_mc, NULL, &vk->desc_layout_mc);

    /* Descriptor Set Layout for others (binding 0 is Buffer) */
    VkDescriptorSetLayoutBinding bindings_buf[3] = {
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
    
    VkDescriptorSetLayoutCreateInfo layout_info_buf = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 3,
        .pBindings = bindings_buf
    };
    
    vkCreateDescriptorSetLayout(vk->device, &layout_info_buf, NULL, &vk->desc_layout_buffer);

    /* Setup Pipeline Layout */
    VkPushConstantRange push_constant_range = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = 32
    };

    /* Setup Pipeline Layout for MC */
    VkPipelineLayoutCreateInfo pipeline_layout_info_mc = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &vk->desc_layout_mc,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &push_constant_range
    };
    
    VkPipelineLayout pipe_layout_mc;
    vkCreatePipelineLayout(vk->device, &pipeline_layout_info_mc, NULL, &pipe_layout_mc);

    /* Setup Pipeline Layout for Buffers (Intra, IDCT, LF) */
    VkPipelineLayoutCreateInfo pipeline_layout_info_buf = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &vk->desc_layout_buffer,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &push_constant_range
    };
    vkCreatePipelineLayout(vk->device, &pipeline_layout_info_buf, NULL, &vk->pipe_layout);

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
        vkDestroyDescriptorSetLayout(vk->device, vk->desc_layout_mc, NULL);
        vkDestroyDescriptorSetLayout(vk->device, vk->desc_layout_buffer, NULL);
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
    res = vkCreateComputePipelines(vk->device, VK_NULL_HANDLE, 1, &pipeline_create_info, NULL, &vk->pipe_idct);
    if (res != VK_SUCCESS) fprintf(stderr, "[compute-vp9] Failed to create pipe_idct: %d\n", res);
 
    pipeline_create_info.layout = pipe_layout_mc;
    pipeline_create_info.stage.module = mod_mc;
    res = vkCreateComputePipelines(vk->device, VK_NULL_HANDLE, 1, &pipeline_create_info, NULL, &vk->pipe_mc);
    if (res != VK_SUCCESS) fprintf(stderr, "[compute-vp9] Failed to create pipe_mc: %d\n", res);
 
    pipeline_create_info.layout = vk->pipe_layout;
    pipeline_create_info.stage.module = mod_intra;
    res = vkCreateComputePipelines(vk->device, VK_NULL_HANDLE, 1, &pipeline_create_info, NULL, &vk->pipe_intra);
    if (res != VK_SUCCESS) fprintf(stderr, "[compute-vp9] Failed to create pipe_intra: %d\n", res);
 
    pipeline_create_info.stage.module = mod_loopfilter;
    res = vkCreateComputePipelines(vk->device, VK_NULL_HANDLE, 1, &pipeline_create_info, NULL, &vk->pipe_loopfilter);
    if (res != VK_SUCCESS) fprintf(stderr, "[compute-vp9] Failed to create pipe_loopfilter: %d\n", res);
    
    vkDestroyShaderModule(vk->device, mod_idct, NULL);
    vkDestroyShaderModule(vk->device, mod_mc, NULL);
    vkDestroyShaderModule(vk->device, mod_intra, NULL);
    vkDestroyShaderModule(vk->device, mod_loopfilter, NULL);

    /* Descriptor Pool */
    VkDescriptorPoolSize pool_size[2] = {
        { .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 64 },
        { .type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = 32 }
    };

    VkDescriptorPoolCreateInfo pool_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = 32,
        .poolSizeCount = 2,
        .pPoolSizes = pool_size
    };

    vkCreateDescriptorPool(vk->device, &pool_info, NULL, &vk->desc_pool);

    /* Allocate persistent descriptor sets */
    vk->desc_mc = allocate_desc_set(vk->device, vk->desc_pool, vk->desc_layout_mc);
    vk->desc_intra = allocate_desc_set(vk->device, vk->desc_pool, vk->desc_layout_buffer);
    vk->desc_idct = allocate_desc_set(vk->device, vk->desc_pool, vk->desc_layout_buffer);
    vk->desc_lf = allocate_desc_set(vk->device, vk->desc_pool, vk->desc_layout_buffer);

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
        
        if (vk->desc_layout_mc != VK_NULL_HANDLE) {
            vkDestroyDescriptorSetLayout(vk->device, vk->desc_layout_mc, NULL);
        }
        if (vk->desc_layout_buffer != VK_NULL_HANDLE) {
            vkDestroyDescriptorSetLayout(vk->device, vk->desc_layout_buffer, NULL);
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
        if (vk->block_buf != VK_NULL_HANDLE) {
            vkDestroyBuffer(vk->device, vk->block_buf, NULL);
            vkFreeMemory(vk->device, vk->block_mem, NULL);
        }

        for (int i = 0; i < 8; i++) {
            if (vk->ref_images[i] != VK_NULL_HANDLE) {
                vkDestroyImageView(vk->device, vk->ref_views[i], NULL);
                vkDestroyImage(vk->device, vk->ref_images[i], NULL);
                vkFreeMemory(vk->device, vk->ref_image_mems[i], NULL);
            }
        }
        if (vk->ref_sampler != VK_NULL_HANDLE) {
            vkDestroySampler(vk->device, vk->ref_sampler, NULL);
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
        VkResult fence_res = vkWaitForFences(vk->device, 1, &vk->fence, VK_TRUE, UINT64_MAX);
        if (fence_res != VK_SUCCESS) {
            fprintf(stderr, "[compute-vp9] vkWaitForFences failed: %d\n", fence_res);
            return CVP9_ERR_GPU;
        }
        vkFreeCommandBuffers(vk->device, vk->cmd_pool, 1, &vk->active_cmd);
        vk->active_cmd = VK_NULL_HANDLE;
    }

    VkResult res = ensure_buffers(vk, pf->hdr.width, pf->hdr.height);
    if (res != VK_SUCCESS) {
        return CVP9_ERR_GPU;
    }

    vk->pts = pts;
    vk->has_frame = 1;
    
    if (pts == 1) {
        printf("[compute-vp9] Frame %ld: num_blocks = %u\n", (long)pts, pf->num_blocks);
    }

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

    /* Batch Command Submission: Upload block data */
    size_t block_buf_size = pf->num_blocks * sizeof(gpu_block_data_t);
    if (block_buf_size == 0) block_buf_size = sizeof(gpu_block_data_t);
    if (vk->block_buf_size < block_buf_size) {
        if (vk->block_buf != VK_NULL_HANDLE) {
            vkDestroyBuffer(vk->device, vk->block_buf, NULL);
            vkFreeMemory(vk->device, vk->block_mem, NULL);
        }
        create_buffer(vk->device, vk->phys_device, block_buf_size,
                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                      &vk->block_buf, &vk->block_mem);
        vk->block_buf_size = block_buf_size;
    }
    if (pf->num_blocks > 0) {
        gpu_block_data_t *mapped;
        vkMapMemory(vk->device, vk->block_mem, 0, block_buf_size, 0, (void **)&mapped);
        for (uint32_t i = 0; i < pf->num_blocks; i++) {
            const vp9_macroblock_info_t *block = &pf->blocks[i];
            mapped[i].is_intra = block->is_intra;
            mapped[i].skip = block->skip;
            mapped[i].block_size = block->width;
            mapped[i].tx_size = 1 << (block->tx_size + 2);
            mapped[i].pred_mode = block->y_mode;
            mapped[i].qstep = 128; /* constant for now */
            mapped[i].coeff_offset = block->coeff_offset;
            mapped[i].dst_stride = vk->width;
            mapped[i].dst_offset = block->y * vk->width + block->x;
            mapped[i].pad1 = 0;
            mapped[i].pad2 = 0;
            mapped[i].pad3 = 0;
        }
        vkUnmapMemory(vk->device, vk->block_mem);
    }

    /* Update descriptor sets once per frame */
    bind_image_to_descriptor(vk->device, vk->desc_mc, 0, vk->ref_views[0], vk->ref_sampler);
    bind_buffer_to_descriptor(vk->device, vk->desc_mc, 1, vk->dst_buf, vk->dst_size);
    bind_buffer_to_descriptor(vk->device, vk->desc_mc, 2, vk->mv_buf, mv_size);

    bind_buffer_to_descriptor(vk->device, vk->desc_intra, 0, vk->dst_buf, vk->dst_size);
    bind_buffer_to_descriptor(vk->device, vk->desc_intra, 1, vk->dst_buf, vk->dst_size);
    bind_buffer_to_descriptor(vk->device, vk->desc_intra, 2, vk->block_buf, block_buf_size);

    bind_buffer_to_descriptor(vk->device, vk->desc_idct, 0, vk->coeff_buf, coeff_size);
    bind_buffer_to_descriptor(vk->device, vk->desc_idct, 1, vk->dst_buf, vk->dst_size);
    bind_buffer_to_descriptor(vk->device, vk->desc_idct, 2, vk->block_buf, block_buf_size);

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

    /* 2. Block-by-block Batched Submission (Intra then IDCT) */
    if (pf->num_blocks > 0) {
        /* Dispatch all Intra Prediction blocks */
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, vk->pipe_intra);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, vk->pipe_layout, 0, 1, &vk->desc_intra, 0, NULL);
        vkCmdDispatch(cmd, pf->num_blocks, 1, 1);

        /* Barrier to ensure Intra finishes before IDCT adds residual */
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0, 1, &mem_barrier, 0, NULL, 0, NULL);

        /* Dispatch all IDCT/Residual blocks */
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, vk->pipe_idct);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, vk->pipe_layout, 0, 1, &vk->desc_idct, 0, NULL);
        vkCmdDispatch(cmd, pf->num_blocks, 1, 1);
        
        /* Barrier to ensure IDCT writes complete before Loop Filter */
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
        .pass = 0 
    };

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, vk->pipe_loopfilter);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, vk->pipe_layout, 0, 1, &vk->desc_lf, 0, NULL);
    
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

    /* Pipeline barrier: compute write -> transfer read */
    VkMemoryBarrier transfer_barrier = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT
    };
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0, 1, &transfer_barrier, 0, NULL, 0, NULL);

    /* Copy destination back to staging output buffer */
    VkBufferCopy copy_region = {
        .srcOffset = 0,
        .dstOffset = 0,
        .size = vk->dst_size
    };
    vkCmdCopyBuffer(cmd, vk->dst_buf, vk->output_buf, 1, &copy_region);

    /* Update Reference frame slot 0 for subsequent inter frames */
    /* Transition ref_image to TRANSFER_DST */
    VkImageMemoryBarrier barrier = {0};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = vk->ref_images[0];
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;
    barrier.srcAccessMask = 0;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0, 0, NULL, 0, NULL, 1, &barrier);

    VkBufferImageCopy region = {0};
    region.bufferOffset = 0;
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;
    region.imageOffset = (VkOffset3D){0, 0, 0};
    region.imageExtent = (VkExtent3D){vk->width, vk->height, 1};

    vkCmdCopyBufferToImage(cmd, vk->dst_buf, vk->ref_images[0], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    /* Transition to SHADER_READ_ONLY */
    barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 0, NULL, 0, NULL, 1, &barrier);

    /* Pipeline barrier: transfer write -> host read */
    VkMemoryBarrier host_barrier = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_HOST_READ_BIT
    };
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
                         0, 1, &host_barrier, 0, NULL, 0, NULL);

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
        fprintf(stderr, "[compute-vp9] vkQueueSubmit failed: %d\n", queue_res);
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
        VkResult fence_res = vkWaitForFences(vk->device, 1, &vk->fence, VK_TRUE, UINT64_MAX);
        if (fence_res != VK_SUCCESS) {
            fprintf(stderr, "[compute-vp9] vkWaitForFences in get_frame failed: %d\n", fence_res);
            return CVP9_ERR_GPU;
        }
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
