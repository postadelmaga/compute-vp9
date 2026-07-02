/**
 * I420 → NV12 pack — Vulkan compute shader
 *
 * Reads the decoded frame (planar I420 in the working buffer) and writes it
 * as NV12 directly into a VA surface's DMA-BUF-exported buffer, so the
 * client (Chrome's compositor) imports the frame with zero CPU copies.
 *
 * One thread per chroma pixel: writes a 2×2 luma quad plus one UV pair.
 */
#version 450
#extension GL_EXT_shader_8bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) readonly buffer SrcI420 {
    uint8_t src[];
};

layout(set = 0, binding = 1, std430) writeonly buffer DstNV12 {
    uint8_t dst[];
};

layout(set = 0, binding = 2, std430) readonly buffer Unused {
    uint8_t unused[];
};

layout(push_constant) uniform PushConst {
    uint width;
    uint height;
} pc;

void main()
{
    uint cw = (pc.width + 1u) / 2u;
    uint ch = (pc.height + 1u) / 2u;
    uint cx = gl_GlobalInvocationID.x;
    uint cy = gl_GlobalInvocationID.y;
    if (cx >= cw || cy >= ch) return;

    uint luma_size = pc.width * pc.height;
    uint chroma_size = cw * ch;

    /* 2×2 luma quad */
    for (uint dy = 0u; dy < 2u; dy++) {
        for (uint dx = 0u; dx < 2u; dx++) {
            uint x = 2u * cx + dx;
            uint y = 2u * cy + dy;
            if (x < pc.width && y < pc.height) {
                dst[y * pc.width + x] = src[y * pc.width + x];
            }
        }
    }

    /* Interleaved UV pair */
    uint uv_row = luma_size + cy * 2u * cw;
    dst[uv_row + 2u * cx]      = src[luma_size + cy * cw + cx];
    dst[uv_row + 2u * cx + 1u] = src[luma_size + chroma_size + cy * cw + cx];
}
