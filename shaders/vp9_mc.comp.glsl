/**
 * VP9 Motion Compensation — Vulkan compute shader
 *
 * Inter prediction with the VP9 8-tap (EIGHTTAP regular) separable subpel
 * filter at 1/16-pel phases (MVs carry 1/8-pel precision, mapped to the
 * even phases), reading from up to 3 active reference frames
 * (LAST / GOLDEN / ALTREF) selected per 4x4 MV-grid cell.
 *
 * Binding layout:
 *   binding=0 — ref_frames[3]: luma planes of the active references
 *   binding=1 — dst_frame:     uint8 output luma plane (writeonly)
 *   binding=2 — motion_vecs:   ivec4 {mv_x, mv_y, ref, pad} per 4x4 cell
 *
 * Push constants: frame dimensions, block grid size, filter type
 */
#version 450
#extension GL_EXT_shader_8bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0) uniform sampler2D ref_frames[3];

layout(set = 0, binding = 1, std430) writeonly buffer DstFrame {
    uint8_t dst_luma[];
};

layout(set = 0, binding = 2, std430) readonly buffer MotionVecs {
    ivec4 mv[];   /* {x, y, ref, pad} — 1/8-pixel precision, per 4x4 block */
};

layout(push_constant) uniform PushConst {
    uint frame_width;
    uint frame_height;
    uint block_size;
    uint filter_type;
} pc;

/* VP9 sub_pel_filters_8 (EIGHTTAP regular), 16 phases × 8 taps, sum = 128 */
const int SUBPEL_FILTER[16][8] = int[16][8](
    int[8](  0,  0,   0, 128,   0,   0,  0,  0),
    int[8](  0,  1,  -5, 126,   8,  -3,  1,  0),
    int[8]( -1,  3, -10, 122,  18,  -6,  2,  0),
    int[8]( -1,  4, -13, 118,  27,  -9,  3, -1),
    int[8]( -1,  4, -16, 112,  37, -11,  4, -1),
    int[8]( -1,  5, -18, 105,  48, -14,  4, -1),
    int[8]( -1,  5, -19,  97,  58, -16,  5, -1),
    int[8]( -1,  6, -19,  88,  68, -18,  6, -2),
    int[8]( -1,  6, -19,  78,  78, -19,  6, -1),
    int[8]( -2,  6, -18,  68,  88, -19,  6, -1),
    int[8]( -1,  5, -16,  58,  97, -19,  5, -1),
    int[8]( -1,  4, -14,  48, 105, -18,  5, -1),
    int[8]( -1,  4, -11,  37, 112, -16,  4, -1),
    int[8]( -1,  3,  -9,  27, 118, -13,  4, -1),
    int[8](  0,  2,  -6,  18, 122, -10,  3, -1),
    int[8](  0,  1,  -3,   8, 126,  -5,  1,  0)
);

/* Clamped texel fetch; constant sampler indices per branch so no dynamic
 * (non-uniform) descriptor indexing is required */
int fetch_ref(int r, int x, int y)
{
    ivec2 p = clamp(ivec2(x, y), ivec2(0),
                    ivec2(int(pc.frame_width) - 1, int(pc.frame_height) - 1));
    if (r == 1) return int(texelFetch(ref_frames[1], p, 0).r * 255.0 + 0.5);
    if (r == 2) return int(texelFetch(ref_frames[2], p, 0).r * 255.0 + 0.5);
    return int(texelFetch(ref_frames[0], p, 0).r * 255.0 + 0.5);
}

void main()
{
    uvec2 pixel = gl_GlobalInvocationID.xy;
    if (pixel.x >= pc.frame_width || pixel.y >= pc.frame_height) return;

    /* Lookup motion vector for the 4x4 block containing this pixel */
    uint bx = pixel.x / 4u;
    uint by = pixel.y / 4u;
    uint blocks_per_row = (pc.frame_width + 3u) / 4u;
    ivec4 m = mv[by * blocks_per_row + bx];
    int ref = clamp(m.z, 0, 2);

    /* Integer and fractional MV parts (1/8 pel → 16-phase filter index) */
    int base_x = int(pixel.x) + (m.x >> 3);
    int base_y = int(pixel.y) + (m.y >> 3);
    int fx = (m.x & 7) << 1;
    int fy = (m.y & 7) << 1;

    int pred;
    if (fx == 0 && fy == 0) {
        /* Full-pel: direct copy */
        pred = fetch_ref(ref, base_x, base_y);
    } else {
        /* Separable 8-tap: horizontal pass into 8 rows, then vertical */
        int rows[8];
        for (int r = 0; r < 8; r++) {
            int acc = 0;
            for (int t = 0; t < 8; t++) {
                acc += SUBPEL_FILTER[fx][t] * fetch_ref(ref, base_x + t - 3, base_y + r - 3);
            }
            rows[r] = clamp((acc + 64) >> 7, 0, 255);
        }
        int acc = 0;
        for (int t = 0; t < 8; t++) {
            acc += SUBPEL_FILTER[fy][t] * rows[t];
        }
        pred = clamp((acc + 64) >> 7, 0, 255);
    }

    dst_luma[pixel.y * pc.frame_width + pixel.x] = uint8_t(pred);
}
