/**
 * VP9 Inverse DCT and Residual Addition — Vulkan compute shader
 *
 * Performs 2D IDCT and adds the residuals directly to the prediction frame.
 */
#version 450
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_8bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require

layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) readonly buffer CoeffIn {
    int16_t coeff[];
};

layout(set = 0, binding = 1, std430) buffer DstFrame {
    uint8_t dst_pixels[];
};

layout(push_constant) uniform PushConst {
    uint block_size;   /* 4, 8, 16 or 32 */
    int  qstep;        /* dequantization step */
    uint block_offset; /* offset into coeff buffer */
    uint dst_stride;
    uint dst_offset;   /* byte offset of block start in dst_pixels */
} pc;

#define M_PI 3.141592653589793

float get_matrix_val(uint i, uint j, uint size) {
    float c0, c1, angle_scale;
    if (size == 4) {
        c0 = 0.5;
        c1 = 0.70710678;
        angle_scale = 0.39269908;
    } else if (size == 8) {
        c0 = 0.35355339;
        c1 = 0.5;
        angle_scale = 0.19634954;
    } else if (size == 16) {
        c0 = 0.25;
        c1 = 0.35355339;
        angle_scale = 0.09817477;
    } else { // size == 32
        c0 = 0.17677669;
        c1 = 0.25;
        angle_scale = 0.049087385;
    }

    if (i == 0) {
        return c0;
    } else {
        return c1 * cos((2.0 * float(j) + 1.0) * float(i) * angle_scale);
    }
}

shared float shared_M[32][32];
shared float shared_in[32][32];
shared float shared_Y[32][32];

void main()
{
    uint tid = gl_LocalInvocationID.x;
    uint N = pc.block_size;

    // 1. Compute and store transform matrix M in shared memory
    if (tid < N) {
        for (uint j = 0; j < N; j++) {
            shared_M[tid][j] = get_matrix_val(tid, j, N);
        }
    }

    // 2. Load and dequantize input coefficients in coalesced manner
    if (tid < N) {
        for (uint row = 0; row < N; row++) {
            int coeff_val = int(coeff[pc.block_offset + row * N + tid]);
            shared_in[row][tid] = float(coeff_val) * float(pc.qstep);
        }
    }

    barrier();

    // 3. Compute Y = M^T * In
    if (tid < N) {
        for (uint col = 0; col < N; col++) {
            float sum = 0.0;
            for (uint k = 0; k < N; k++) {
                sum += shared_M[k][tid] * shared_in[k][col];
            }
            shared_Y[tid][col] = sum;
        }
    }

    barrier();

    // 4. Compute Out = Y * M and add to dst_pixels
    if (tid < N) {
        for (uint col = 0; col < N; col++) {
            float sum = 0.0;
            for (uint k = 0; k < N; k++) {
                sum += shared_Y[tid][k] * shared_M[k][col];
            }
            
            int rounded = int(round(sum));
            uint pixel_offset = pc.dst_offset + tid * pc.dst_stride + col;
            int pred_val = int(dst_pixels[pixel_offset]);
            
            dst_pixels[pixel_offset] = uint8_t(clamp(pred_val + rounded, 0, 255));
        }
    }
}
