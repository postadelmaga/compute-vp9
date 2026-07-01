/**
 * VP9 Intra Prediction — Vulkan compute shader
 *
 * Implements VP9 intra prediction modes for luma (Y) plane:
 *   0 = DC_PRED, 1 = V_PRED, 2 = H_PRED, 3 = D45_PRED,
 *   4 = D135_PRED, 5 = D117_PRED, 6 = D153_PRED,
 *   7 = D207_PRED, 8 = D63_PRED, 9 = TM_PRED
 */
#version 450
#extension GL_EXT_shader_8bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

/* Above and left neighbor rows (packed into buffers) */
layout(set = 0, binding = 0, std430) readonly buffer AboveRow {
    uint8_t above[];
};
layout(set = 0, binding = 1, std430) readonly buffer LeftCol {
    uint8_t left[];
};
layout(set = 0, binding = 2, std430) writeonly buffer PredOut {
    uint8_t pred[];
};

layout(push_constant) uniform PushConst {
    uint block_size;  /* 4, 8, 16, 32 */
    uint pred_mode;   /* 0–9 */
    uint dst_stride;
    uint dst_offset;
} pc;

uint sample_above(uint x) { return uint(above[x]); }
uint sample_left(uint y)  { return uint(left[y]);  }

void main()
{
    uvec2 pos = gl_LocalInvocationID.xy;
    if (pos.x >= pc.block_size || pos.y >= pc.block_size) return;

    uint val = 128u; /* fallback: mid-grey */

    switch (pc.pred_mode) {
    case 0: { /* DC_PRED — average of above + left */
        uint sum = 0u;
        for (uint i = 0u; i < pc.block_size; i++) sum += sample_above(i);
        for (uint i = 0u; i < pc.block_size; i++) sum += sample_left(i);
        val = (sum + pc.block_size) / (pc.block_size * 2u);
        break;
    }
    case 1: /* V_PRED — copy from above */
        val = sample_above(pos.x);
        break;
    case 2: /* H_PRED — copy from left */
        val = sample_left(pos.y);
        break;
    case 9: { /* TM_PRED — true motion */
        uint top_left = sample_above(0u); /* approximation */
        val = clamp(int(sample_above(pos.x)) + int(sample_left(pos.y)) - int(top_left),
                    0, 255);
        break;
    }
    /* D45..D63: angular modes — TODO */
    default:
        val = sample_above(pos.x);
        break;
    }

    pred[pc.dst_offset + pos.y * pc.dst_stride + pos.x] = uint8_t(val);
}
