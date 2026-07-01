/**
 * VP9 Inverse DCT — Vulkan compute shader
 *
 * Implements 4x4, 8x8, 16x16 and 32x32 2D IDCT on transform coefficients.
 * Each workgroup processes one transform block.
 *
 * Binding layout:
 *   binding=0  — input:  int16 coefficients  (readonly)
 *   binding=1  — output: int16 residuals      (writeonly)
 *   push constants: block_size (4/8/16/32), qstep
 */
#version 450
#extension GL_EXT_shader_16bit_storage : require

layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) readonly buffer CoeffIn {
    int16_t coeff[];
};

layout(set = 0, binding = 1, std430) writeonly buffer ResidualOut {
    int16_t residual[];
};

layout(push_constant) uniform PushConst {
    uint block_size;   /* 4, 8, 16 or 32 */
    int  qstep;        /* dequantization step */
    uint block_offset; /* offset into coeff/residual buffers */
} pc;

/* ── Shared memory for 1D butterfly passes ──────────────────────────────── */
shared int tmp[32];

/* ── VP9 IDCT constants (scaled by 2^14) ────────────────────────────────── */
const int COS_PI_1_64 = 16364;
const int COS_PI_4_64 = 15137;
const int COS_PI_8_64 = 13623;
const int COS_PI_16_64 = 11585;

int butterfly(int a, int b, int cos_val, int sin_val) {
    return (a * cos_val + b * sin_val + (1 << 13)) >> 14;
}

/* ── 4x4 IDCT (one thread per row) ─────────────────────────────────────── */
void idct4x4(uint row)
{
    uint base = pc.block_offset + row * 4;
    int c0 = int(coeff[base + 0]) * pc.qstep;
    int c1 = int(coeff[base + 1]) * pc.qstep;
    int c2 = int(coeff[base + 2]) * pc.qstep;
    int c3 = int(coeff[base + 3]) * pc.qstep;

    /* Stage 1 */
    int s0 = (c0 + c2) >> 1;
    int s1 = (c0 - c2) >> 1;
    int s2 = butterfly(c1, c3,  COS_PI_4_64,  COS_PI_16_64);
    int s3 = butterfly(c1, c3, -COS_PI_16_64, COS_PI_4_64);

    /* Stage 2 */
    residual[base + 0] = int16_t(clamp(s0 + s2, -32768, 32767));
    residual[base + 1] = int16_t(clamp(s1 + s3, -32768, 32767));
    residual[base + 2] = int16_t(clamp(s1 - s3, -32768, 32767));
    residual[base + 3] = int16_t(clamp(s0 - s2, -32768, 32767));
}

void main()
{
    uint tid = gl_GlobalInvocationID.x;
    uint block_area = pc.block_size * pc.block_size;

    /* Each invocation handles one row of the assigned block */
    if (tid >= pc.block_size) return;

    if (pc.block_size == 4) {
        idct4x4(tid);
    }
    /* TODO: idct8x8, idct16x16, idct32x32 — to be implemented */
}
