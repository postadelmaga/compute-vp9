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

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) readonly buffer RefFrame {
    uint8_t ref_luma[];
};

layout(set = 0, binding = 1, std430) writeonly buffer DstFrame {
    uint8_t dst_luma[];
};

layout(set = 0, binding = 2, std430) readonly buffer MotionVecs {
    ivec2 mv[];   /* 1/8-pixel precision, per 4x4 block */
};

layout(push_constant) uniform PushConst {
    uint frame_width;
    uint frame_height;
    uint block_size;    /* 4, 8, 16, 32, 64 */
    uint filter_type;   /* 0=bilinear, 1=8-tap, 2=smooth */
} pc;

/* ── Bilinear interpolation helper ─────────────────────────────────────── */
uint bilinear(uint x, uint y, ivec2 sub)
{
    /* sub is in 1/8 pixel units */
    int fx = sub.x & 7;
    int fy = sub.y & 7;

    uint p00 = ref_luma[clamp(y,     0u, pc.frame_height-1u) * pc.frame_width
                       + clamp(x,     0u, pc.frame_width-1u)];
    uint p10 = ref_luma[clamp(y,     0u, pc.frame_height-1u) * pc.frame_width
                       + clamp(x+1u, 0u, pc.frame_width-1u)];
    uint p01 = ref_luma[clamp(y+1u, 0u, pc.frame_height-1u) * pc.frame_width
                       + clamp(x,     0u, pc.frame_width-1u)];
    uint p11 = ref_luma[clamp(y+1u, 0u, pc.frame_height-1u) * pc.frame_width
                       + clamp(x+1u, 0u, pc.frame_width-1u)];

    uint top = (p00 * (8u - uint(fx)) + p10 * uint(fx) + 4u) >> 3;
    uint bot = (p01 * (8u - uint(fx)) + p11 * uint(fx) + 4u) >> 3;
    return    (top  * (8u - uint(fy)) + bot  * uint(fy) + 4u) >> 3;
}

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
    int ref_x8 = int(pixel.x) * 8 + mv_val.x;
    int ref_y8 = int(pixel.y) * 8 + mv_val.y;
    uint ref_x  = uint(ref_x8 >> 3);
    uint ref_y  = uint(ref_y8 >> 3);
    ivec2 sub   = ivec2(ref_x8 & 7, ref_y8 & 7);

    uint predicted = bilinear(ref_x, ref_y, sub);
    dst_luma[pixel.y * pc.frame_width + pixel.x] = uint8_t(predicted);
}
