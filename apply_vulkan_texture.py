import re
import sys

def patch_vulkan_backend():
    with open('src/backend/vulkan/vulkan_backend.c', 'r') as f:
        src = f.read()

    # 1. Add create_image helper
    create_image_func = """
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
"""
    # Insert helper before create_buffer
    src = src.replace('static VkResult create_buffer(', create_image_func + '\nstatic VkResult create_buffer(')

    # 2. Change desc_mc binding 0 to COMBINED_IMAGE_SAMPLER
    src = re.sub(
        r'VkDescriptorSetLayoutBinding bindings\[3\] = \{.*?\};',
        r'''VkDescriptorSetLayoutBinding bindings[3] = {
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
    };''',
        src, flags=re.DOTALL
    )

    # 3. Add COMBINED_IMAGE_SAMPLER to Descriptor Pool
    src = src.replace(
        'VkDescriptorPoolSize pool_size = {',
        'VkDescriptorPoolSize pool_size[2] = {\n        { .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 64 },\n        { .type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = 32 }\n    };\n    /* '
    )
    src = src.replace(
        '.poolSizeCount = 1,\n        .pPoolSizes = &pool_size',
        '.poolSizeCount = 2,\n        .pPoolSizes = pool_size'
    )

    # 4. In reallocate_buffers, allocate images instead of buffers
    old_alloc = """for (int i = 0; i < 8; i++) {
        if (vk->ref_bufs[i] != VK_NULL_HANDLE) {
            vkDestroyBuffer(vk->device, vk->ref_bufs[i], NULL);
            vkFreeMemory(vk->device, vk->ref_mems[i], NULL);
            vk->ref_bufs[i] = VK_NULL_HANDLE;
        }
    }"""
    
    new_alloc = """for (int i = 0; i < 8; i++) {
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
    """
    src = src.replace(old_alloc, new_alloc)

    old_alloc2 = """for (int i = 0; i < 8; i++) {
        vk->ref_sizes[i] = yuv_size;
        res = create_buffer(vk->device, vk->phys_device, yuv_size,
                            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                            &vk->ref_bufs[i], &vk->ref_mems[i]);
        if (res != VK_SUCCESS) {
            fprintf(stderr, "[compute-vp9] Failed to create ref_buf[%d]: %d\\n", i, res);
            return CVP9_ERR_GPU;
        }
    }"""
    new_alloc2 = """for (int i = 0; i < 8; i++) {
        res = create_image(vk->device, vk->phys_device, vk->width, vk->height, VK_FORMAT_R8_UNORM, VK_IMAGE_TILING_OPTIMAL,
                           VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                           VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                           &vk->ref_images[i], &vk->ref_image_mems[i]);
        if (res != VK_SUCCESS) return CVP9_ERR_GPU;
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
    if (vkCreateSampler(vk->device, &samplerInfo, NULL, &vk->ref_sampler) != VK_SUCCESS) return CVP9_ERR_GPU;
    """
    src = src.replace(old_alloc2, new_alloc2)

    # 5. In vulkan_backend_destroy
    old_dest = """for (int i = 0; i < 8; i++) {
            if (vk->ref_bufs[i] != VK_NULL_HANDLE) {
                vkDestroyBuffer(vk->device, vk->ref_bufs[i], NULL);
                vkFreeMemory(vk->device, vk->ref_mems[i], NULL);
            }
        }"""
    new_dest = """for (int i = 0; i < 8; i++) {
            if (vk->ref_images[i] != VK_NULL_HANDLE) {
                vkDestroyImageView(vk->device, vk->ref_views[i], NULL);
                vkDestroyImage(vk->device, vk->ref_images[i], NULL);
                vkFreeMemory(vk->device, vk->ref_image_mems[i], NULL);
            }
        }
        if (vk->ref_sampler != VK_NULL_HANDLE) {
            vkDestroySampler(vk->device, vk->ref_sampler, NULL);
        }"""
    src = src.replace(old_dest, new_dest)

    # 6. Bind texture instead of buffer
    old_bind = """void bind_buffer_to_descriptor(VkDevice device, VkDescriptorSet set, uint32_t binding, VkBuffer buffer, VkDeviceSize size)
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
        .dstArrayElement = 0,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .pBufferInfo = &buffer_info
    };
    vkUpdateDescriptorSets(device, 1, &write, 0, NULL);
}"""
    new_bind = old_bind + """
void bind_image_to_descriptor(VkDevice device, VkDescriptorSet set, uint32_t binding, VkImageView view, VkSampler sampler)
{
    VkDescriptorImageInfo image_info = {
        .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        .imageView = view,
        .sampler = sampler
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
"""
    src = src.replace(old_bind, new_bind)

    src = src.replace(
        "bind_buffer_to_descriptor(vk->device, vk->desc_mc, 0, vk->ref_bufs[0], vk->dst_size);",
        "bind_image_to_descriptor(vk->device, vk->desc_mc, 0, vk->ref_views[0], vk->ref_sampler);"
    )

    # 7. vkCmdCopyBufferToImage
    # First we need to transition image layout!
    copy_logic = """
    // Transition ref_image to TRANSFER_DST
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

    // Transition to SHADER_READ_ONLY
    barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 0, NULL, 0, NULL, 1, &barrier);
"""
    old_copy = """VkBufferCopy ref_copy = {
        .srcOffset = 0,
        .dstOffset = 0,
        .size = vk->dst_size
    };
    
    vkCmdCopyBuffer(cmd, vk->dst_buf, vk->ref_bufs[0], 1, &ref_copy);"""
    src = src.replace(old_copy, copy_logic)

    with open('src/backend/vulkan/vulkan_backend.c', 'w') as f:
        f.write(src)

patch_vulkan_backend()
