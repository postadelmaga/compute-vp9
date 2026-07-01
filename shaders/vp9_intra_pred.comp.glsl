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

layout(set = 0, binding = 0, std430) buffer PredOut {
    uint8_t pred[];
};

layout(push_constant) uniform PushConst {
    uint block_size;  /* 4, 8, 16, 32 */
    uint pred_mode;   /* 0–9 */
    uint dst_stride;
    uint dst_offset;
} pc;

uint sample_above(uint x) {
    return uint(pred[pc.dst_offset - pc.dst_stride + x]);
}

uint sample_left(uint y) {
    return uint(pred[pc.dst_offset - 1u + y * pc.dst_stride]);
}

uint sample_top_left() {
    return uint(pred[pc.dst_offset - pc.dst_stride - 1u]);
}

void main()
{
    uvec2 pos = gl_GlobalInvocationID.xy;
    if (pos.x >= pc.block_size || pos.y >= pc.block_size) return;

    uint val = 128u; /* fallback: mid-grey */

    bool has_above = (pc.dst_offset >= pc.dst_stride);
    bool has_left = ((pc.dst_offset % pc.dst_stride) > 0u);

    switch (pc.pred_mode) {
    case 0: { /* DC_PRED — average of above + left */
        if (has_above && has_left) {
            uint sum = 0u;
            for (uint i = 0u; i < pc.block_size; i++) sum += sample_above(i);
            for (uint i = 0u; i < pc.block_size; i++) sum += sample_left(i);
            val = (sum + pc.block_size) / (pc.block_size * 2u);
        } else if (has_above) {
            uint sum = 0u;
            for (uint i = 0u; i < pc.block_size; i++) sum += sample_above(i);
            val = (sum + pc.block_size / 2u) / pc.block_size;
        } else if (has_left) {
            uint sum = 0u;
            for (uint i = 0u; i < pc.block_size; i++) sum += sample_left(i);
            val = (sum + pc.block_size / 2u) / pc.block_size;
        } else {
            val = 128u;
        }
        break;
    }
    case 1: /* V_PRED — copy from above */
        val = has_above ? sample_above(pos.x) : 128u;
        break;
    case 2: /* H_PRED — copy from left */
        val = has_left ? sample_left(pos.y) : 128u;
        break;
    case 9: { /* TM_PRED — true motion */
        uint top_left = (has_above && has_left) ? sample_top_left() : 128u;
        uint a = has_above ? sample_above(pos.x) : 128u;
        uint l = has_left ? sample_left(pos.y) : 128u;
        val = uint(clamp(int(a) + int(l) - int(top_left), 0, 255));
        break;
    }
    /* D45..D63: angular modes — TODO */
    default:
        val = has_above ? sample_above(pos.x) : 128u;
        break;
    }

    pred[pc.dst_offset + pos.y * pc.dst_stride + pos.x] = uint8_t(val);
}
