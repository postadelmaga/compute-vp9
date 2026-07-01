/**
 * VP9 Motion Compensation — Vulkan compute shader
 *
 * Performs inter-frame prediction by copying/blending blocks
 * from reference frames using VP9 motion vectors.
 *
 * Binding layout:
 *   binding=0 — ref_frame:    uint8  luma plane of reference frame (readonly)
 *   binding=1 — dst_frame:    uint8  output luma plane             (writeonly)
 *   binding=2 — motion_vecs:  ivec2  per-block motion vectors      (readonly)
 *
 * Push constants: frame dimensions, block grid size, filter type
 */
#version 450
#extension GL_EXT_shader_8bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0) uniform sampler2D ref_frame;

layout(set = 0, binding = 1, std430) writeonly buffer DstFrame {
    uint8_t dst_luma[];
};

layout(set = 0, binding = 2, std430) readonly buffer MotionVecs {
    ivec2 mv[];   /* 1/8-pixel precision, per 4x4 block */
};

layout(push_constant) uniform PushConst {
    uint frame_width;
    uint frame_height;
    uint block_size;
    uint filter_type;
} pc;

void main()
{
    uvec2 pixel = gl_GlobalInvocationID.xy;
    if (pixel.x >= pc.frame_width || pixel.y >= pc.frame_height) return;

    /* Lookup motion vector for the 4x4 block containing this pixel */
    uint bx = pixel.x / 4u;
    uint by = pixel.y / 4u;
    uint blocks_per_row = (pc.frame_width + 3u) / 4u;
    ivec2 mv_val = mv[by * blocks_per_row + bx];

    /* Reference position in 1/8 pixel units */
    float ref_x = float(int(pixel.x * 8) + mv_val.x) / 8.0;
    float ref_y = float(int(pixel.y * 8) + mv_val.y) / 8.0;

    /* Normalize coordinates for sampler2D */
    vec2 uv = vec2(ref_x + 0.5, ref_y + 0.5) / vec2(pc.frame_width, pc.frame_height);

    /* Hardware bilinear interpolation */
    float sampled = texture(ref_frame, uv).r;
    
    /* Convert back to 8-bit [0, 255] */
    uint predicted = uint(clamp(sampled * 255.0, 0.0, 255.0));
    
    dst_luma[pixel.y * pc.frame_width + pixel.x] = uint8_t(predicted);
}
