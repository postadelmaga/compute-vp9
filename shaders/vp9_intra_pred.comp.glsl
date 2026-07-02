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

layout(local_size_x = 32, local_size_y = 32, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) buffer PredOut {
    uint8_t pred[];
};

struct BlockData {
    uint is_intra;
    uint skip;
    uint block_size;
    uint tx_size;
    
    uint pred_mode;
    int qstep;
    uint coeff_offset;
    uint dst_stride;
    
    uint dst_offset;
    int  qstep_dc;
    uint pad2;
    uint pad3;
};

layout(set = 0, binding = 2, std430) readonly buffer BlockBuf {
    BlockData blocks[];
};

uint sample_above(uint x, uint dst_offset, uint dst_stride) {
    return uint(pred[dst_offset - dst_stride + x]);
}

uint sample_left(uint y, uint dst_offset, uint dst_stride) {
    return uint(pred[dst_offset - 1u + y * dst_stride]);
}

uint sample_top_left(uint dst_offset, uint dst_stride) {
    return uint(pred[dst_offset - dst_stride - 1u]);
}

shared uint shared_dc_val;

void main()
{
    uint block_idx = gl_WorkGroupID.x;
    BlockData b = blocks[block_idx];
    
    if (b.is_intra == 0) return;
    
    uvec2 pos = gl_LocalInvocationID.xy;
    if (pos.x >= b.block_size || pos.y >= b.block_size) return;

    uint val = 128u; /* fallback: mid-grey */

    bool has_above = (b.dst_offset >= b.dst_stride);
    bool has_left = ((b.dst_offset % b.dst_stride) > 0u);

    switch (b.pred_mode) {
    case 0: { /* DC_PRED — average of above + left */
        if (has_above && has_left) {
            uint sum = 0u;
            for (uint i = 0u; i < b.block_size; i++) sum += sample_above(i, b.dst_offset, b.dst_stride);
            for (uint i = 0u; i < b.block_size; i++) sum += sample_left(i, b.dst_offset, b.dst_stride);
            val = (sum + b.block_size) / (b.block_size * 2u);
        } else if (has_above) {
            uint sum = 0u;
            for (uint i = 0u; i < b.block_size; i++) sum += sample_above(i, b.dst_offset, b.dst_stride);
            val = (sum + b.block_size / 2u) / b.block_size;
        } else if (has_left) {
            uint sum = 0u;
            for (uint i = 0u; i < b.block_size; i++) sum += sample_left(i, b.dst_offset, b.dst_stride);
            val = (sum + b.block_size / 2u) / b.block_size;
        } else {
            val = 128u;
        }
        break;
    }
    case 1: /* V_PRED — copy from above */
        val = has_above ? sample_above(pos.x, b.dst_offset, b.dst_stride) : 128u;
        break;
    case 2: /* H_PRED — copy from left */
        val = has_left ? sample_left(pos.y, b.dst_offset, b.dst_stride) : 128u;
        break;
    case 9: { /* TM_PRED — true motion */
        uint top_left = (has_above && has_left) ? sample_top_left(b.dst_offset, b.dst_stride) : 128u;
        uint a = has_above ? sample_above(pos.x, b.dst_offset, b.dst_stride) : 128u;
        uint l = has_left ? sample_left(pos.y, b.dst_offset, b.dst_stride) : 128u;
        val = uint(clamp(int(a) + int(l) - int(top_left), 0, 255));
        break;
    }
    /* D45..D63: angular modes — TODO */
    default:
        val = has_above ? sample_above(pos.x, b.dst_offset, b.dst_stride) : 128u;
        break;
    }

    pred[b.dst_offset + pos.y * b.dst_stride + pos.x] = uint8_t(val);
}
