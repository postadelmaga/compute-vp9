/**
 * VP9 Inverse DCT and Residual Addition — Vulkan compute shader
 *
 * Performs 1D Fast IDCT (butterfly) and adds the residuals directly to the prediction frame.
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
    uint pad1;
    uint pad2;
    uint pad3;
};

layout(set = 0, binding = 2, std430) readonly buffer BlockBuf {
    BlockData blocks[];
};

#define ROUND_SHIFT(x) (((x) + 8192) >> 14)

shared int shared_step1[32][32];
shared int shared_step2[32][32];

void idct4_c(int input_arr[4], inout int output_arr[4]) {
    int temp1, temp2;
    
    // stage 1
    temp1 = (input_arr[0] + input_arr[2]) * 11585;
    temp2 = (input_arr[0] - input_arr[2]) * 11585;
    shared_step1[gl_LocalInvocationID.x][0] = ROUND_SHIFT(temp1);
    shared_step1[gl_LocalInvocationID.x][1] = ROUND_SHIFT(temp2);
    temp1 = input_arr[1] * 6270 - input_arr[3] * 15137;
    temp2 = input_arr[1] * 15137 + input_arr[3] * 6270;
    shared_step1[gl_LocalInvocationID.x][2] = ROUND_SHIFT(temp1);
    shared_step1[gl_LocalInvocationID.x][3] = ROUND_SHIFT(temp2);
    
    // stage 2
    output_arr[0] = (shared_step1[gl_LocalInvocationID.x][0] + shared_step1[gl_LocalInvocationID.x][3]);
    output_arr[1] = (shared_step1[gl_LocalInvocationID.x][1] + shared_step1[gl_LocalInvocationID.x][2]);
    output_arr[2] = (shared_step1[gl_LocalInvocationID.x][1] - shared_step1[gl_LocalInvocationID.x][2]);
    output_arr[3] = (shared_step1[gl_LocalInvocationID.x][0] - shared_step1[gl_LocalInvocationID.x][3]);
}

void idct8_c(int input_arr[8], inout int output_arr[8]) {
    int temp1, temp2;
    
    // stage 1
    shared_step1[gl_LocalInvocationID.x][0] = input_arr[0];
    shared_step1[gl_LocalInvocationID.x][2] = input_arr[4];
    shared_step1[gl_LocalInvocationID.x][1] = input_arr[2];
    shared_step1[gl_LocalInvocationID.x][3] = input_arr[6];
    temp1 = input_arr[1] * 3196 - input_arr[7] * 16069;
    temp2 = input_arr[1] * 16069 + input_arr[7] * 3196;
    shared_step1[gl_LocalInvocationID.x][4] = ROUND_SHIFT(temp1);
    shared_step1[gl_LocalInvocationID.x][7] = ROUND_SHIFT(temp2);
    temp1 = input_arr[5] * 13623 - input_arr[3] * 9102;
    temp2 = input_arr[5] * 9102 + input_arr[3] * 13623;
    shared_step1[gl_LocalInvocationID.x][5] = ROUND_SHIFT(temp1);
    shared_step1[gl_LocalInvocationID.x][6] = ROUND_SHIFT(temp2);
    
    // stage 2
    temp1 = (shared_step1[gl_LocalInvocationID.x][0] + shared_step1[gl_LocalInvocationID.x][2]) * 11585;
    temp2 = (shared_step1[gl_LocalInvocationID.x][0] - shared_step1[gl_LocalInvocationID.x][2]) * 11585;
    shared_step2[gl_LocalInvocationID.x][0] = ROUND_SHIFT(temp1);
    shared_step2[gl_LocalInvocationID.x][1] = ROUND_SHIFT(temp2);
    temp1 = shared_step1[gl_LocalInvocationID.x][1] * 6270 - shared_step1[gl_LocalInvocationID.x][3] * 15137;
    temp2 = shared_step1[gl_LocalInvocationID.x][1] * 15137 + shared_step1[gl_LocalInvocationID.x][3] * 6270;
    shared_step2[gl_LocalInvocationID.x][2] = ROUND_SHIFT(temp1);
    shared_step2[gl_LocalInvocationID.x][3] = ROUND_SHIFT(temp2);
    shared_step2[gl_LocalInvocationID.x][4] = (shared_step1[gl_LocalInvocationID.x][4] + shared_step1[gl_LocalInvocationID.x][5]);
    shared_step2[gl_LocalInvocationID.x][5] = (shared_step1[gl_LocalInvocationID.x][4] - shared_step1[gl_LocalInvocationID.x][5]);
    shared_step2[gl_LocalInvocationID.x][6] = (-shared_step1[gl_LocalInvocationID.x][6] + shared_step1[gl_LocalInvocationID.x][7]);
    shared_step2[gl_LocalInvocationID.x][7] = (shared_step1[gl_LocalInvocationID.x][6] + shared_step1[gl_LocalInvocationID.x][7]);
    
    // stage 3
    shared_step1[gl_LocalInvocationID.x][0] = (shared_step2[gl_LocalInvocationID.x][0] + shared_step2[gl_LocalInvocationID.x][3]);
    shared_step1[gl_LocalInvocationID.x][1] = (shared_step2[gl_LocalInvocationID.x][1] + shared_step2[gl_LocalInvocationID.x][2]);
    shared_step1[gl_LocalInvocationID.x][2] = (shared_step2[gl_LocalInvocationID.x][1] - shared_step2[gl_LocalInvocationID.x][2]);
    shared_step1[gl_LocalInvocationID.x][3] = (shared_step2[gl_LocalInvocationID.x][0] - shared_step2[gl_LocalInvocationID.x][3]);
    shared_step1[gl_LocalInvocationID.x][4] = shared_step2[gl_LocalInvocationID.x][4];
    temp1 = (shared_step2[gl_LocalInvocationID.x][6] - shared_step2[gl_LocalInvocationID.x][5]) * 11585;
    temp2 = (shared_step2[gl_LocalInvocationID.x][5] + shared_step2[gl_LocalInvocationID.x][6]) * 11585;
    shared_step1[gl_LocalInvocationID.x][5] = ROUND_SHIFT(temp1);
    shared_step1[gl_LocalInvocationID.x][6] = ROUND_SHIFT(temp2);
    shared_step1[gl_LocalInvocationID.x][7] = shared_step2[gl_LocalInvocationID.x][7];
    
    // stage 4
    output_arr[0] = (shared_step1[gl_LocalInvocationID.x][0] + shared_step1[gl_LocalInvocationID.x][7]);
    output_arr[1] = (shared_step1[gl_LocalInvocationID.x][1] + shared_step1[gl_LocalInvocationID.x][6]);
    output_arr[2] = (shared_step1[gl_LocalInvocationID.x][2] + shared_step1[gl_LocalInvocationID.x][5]);
    output_arr[3] = (shared_step1[gl_LocalInvocationID.x][3] + shared_step1[gl_LocalInvocationID.x][4]);
    output_arr[4] = (shared_step1[gl_LocalInvocationID.x][3] - shared_step1[gl_LocalInvocationID.x][4]);
    output_arr[5] = (shared_step1[gl_LocalInvocationID.x][2] - shared_step1[gl_LocalInvocationID.x][5]);
    output_arr[6] = (shared_step1[gl_LocalInvocationID.x][1] - shared_step1[gl_LocalInvocationID.x][6]);
    output_arr[7] = (shared_step1[gl_LocalInvocationID.x][0] - shared_step1[gl_LocalInvocationID.x][7]);
}

void idct16_c(int input_arr[16], inout int output_arr[16]) {
    int temp1, temp2;
    
    // stage 1
    shared_step1[gl_LocalInvocationID.x][0] = input_arr[0 / 2];
    shared_step1[gl_LocalInvocationID.x][1] = input_arr[16 / 2];
    shared_step1[gl_LocalInvocationID.x][2] = input_arr[8 / 2];
    shared_step1[gl_LocalInvocationID.x][3] = input_arr[24 / 2];
    shared_step1[gl_LocalInvocationID.x][4] = input_arr[4 / 2];
    shared_step1[gl_LocalInvocationID.x][5] = input_arr[20 / 2];
    shared_step1[gl_LocalInvocationID.x][6] = input_arr[12 / 2];
    shared_step1[gl_LocalInvocationID.x][7] = input_arr[28 / 2];
    shared_step1[gl_LocalInvocationID.x][8] = input_arr[2 / 2];
    shared_step1[gl_LocalInvocationID.x][9] = input_arr[18 / 2];
    shared_step1[gl_LocalInvocationID.x][10] = input_arr[10 / 2];
    shared_step1[gl_LocalInvocationID.x][11] = input_arr[26 / 2];
    shared_step1[gl_LocalInvocationID.x][12] = input_arr[6 / 2];
    shared_step1[gl_LocalInvocationID.x][13] = input_arr[22 / 2];
    shared_step1[gl_LocalInvocationID.x][14] = input_arr[14 / 2];
    shared_step1[gl_LocalInvocationID.x][15] = input_arr[30 / 2];
    
    // stage 2
    shared_step2[gl_LocalInvocationID.x][0] = shared_step1[gl_LocalInvocationID.x][0];
    shared_step2[gl_LocalInvocationID.x][1] = shared_step1[gl_LocalInvocationID.x][1];
    shared_step2[gl_LocalInvocationID.x][2] = shared_step1[gl_LocalInvocationID.x][2];
    shared_step2[gl_LocalInvocationID.x][3] = shared_step1[gl_LocalInvocationID.x][3];
    shared_step2[gl_LocalInvocationID.x][4] = shared_step1[gl_LocalInvocationID.x][4];
    shared_step2[gl_LocalInvocationID.x][5] = shared_step1[gl_LocalInvocationID.x][5];
    shared_step2[gl_LocalInvocationID.x][6] = shared_step1[gl_LocalInvocationID.x][6];
    shared_step2[gl_LocalInvocationID.x][7] = shared_step1[gl_LocalInvocationID.x][7];
    
    temp1 = shared_step1[gl_LocalInvocationID.x][8] * 1606 - shared_step1[gl_LocalInvocationID.x][15] * 16305;
    temp2 = shared_step1[gl_LocalInvocationID.x][8] * 16305 + shared_step1[gl_LocalInvocationID.x][15] * 1606;
    shared_step2[gl_LocalInvocationID.x][8] = ROUND_SHIFT(temp1);
    shared_step2[gl_LocalInvocationID.x][15] = ROUND_SHIFT(temp2);
    
    temp1 = shared_step1[gl_LocalInvocationID.x][9] * 12665 - shared_step1[gl_LocalInvocationID.x][14] * 10394;
    temp2 = shared_step1[gl_LocalInvocationID.x][9] * 10394 + shared_step1[gl_LocalInvocationID.x][14] * 12665;
    shared_step2[gl_LocalInvocationID.x][9] = ROUND_SHIFT(temp1);
    shared_step2[gl_LocalInvocationID.x][14] = ROUND_SHIFT(temp2);
    
    temp1 = shared_step1[gl_LocalInvocationID.x][10] * 7723 - shared_step1[gl_LocalInvocationID.x][13] * 14449;
    temp2 = shared_step1[gl_LocalInvocationID.x][10] * 14449 + shared_step1[gl_LocalInvocationID.x][13] * 7723;
    shared_step2[gl_LocalInvocationID.x][10] = ROUND_SHIFT(temp1);
    shared_step2[gl_LocalInvocationID.x][13] = ROUND_SHIFT(temp2);
    
    temp1 = shared_step1[gl_LocalInvocationID.x][11] * 15679 - shared_step1[gl_LocalInvocationID.x][12] * 4756;
    temp2 = shared_step1[gl_LocalInvocationID.x][11] * 4756 + shared_step1[gl_LocalInvocationID.x][12] * 15679;
    shared_step2[gl_LocalInvocationID.x][11] = ROUND_SHIFT(temp1);
    shared_step2[gl_LocalInvocationID.x][12] = ROUND_SHIFT(temp2);
    
    // stage 3
    shared_step1[gl_LocalInvocationID.x][0] = shared_step2[gl_LocalInvocationID.x][0];
    shared_step1[gl_LocalInvocationID.x][1] = shared_step2[gl_LocalInvocationID.x][1];
    shared_step1[gl_LocalInvocationID.x][2] = shared_step2[gl_LocalInvocationID.x][2];
    shared_step1[gl_LocalInvocationID.x][3] = shared_step2[gl_LocalInvocationID.x][3];
    
    temp1 = shared_step2[gl_LocalInvocationID.x][4] * 3196 - shared_step2[gl_LocalInvocationID.x][7] * 16069;
    temp2 = shared_step2[gl_LocalInvocationID.x][4] * 16069 + shared_step2[gl_LocalInvocationID.x][7] * 3196;
    shared_step1[gl_LocalInvocationID.x][4] = ROUND_SHIFT(temp1);
    shared_step1[gl_LocalInvocationID.x][7] = ROUND_SHIFT(temp2);
    temp1 = shared_step2[gl_LocalInvocationID.x][5] * 13623 - shared_step2[gl_LocalInvocationID.x][6] * 9102;
    temp2 = shared_step2[gl_LocalInvocationID.x][5] * 9102 + shared_step2[gl_LocalInvocationID.x][6] * 13623;
    shared_step1[gl_LocalInvocationID.x][5] = ROUND_SHIFT(temp1);
    shared_step1[gl_LocalInvocationID.x][6] = ROUND_SHIFT(temp2);
    
    shared_step1[gl_LocalInvocationID.x][8] = (shared_step2[gl_LocalInvocationID.x][8] + shared_step2[gl_LocalInvocationID.x][9]);
    shared_step1[gl_LocalInvocationID.x][9] = (shared_step2[gl_LocalInvocationID.x][8] - shared_step2[gl_LocalInvocationID.x][9]);
    shared_step1[gl_LocalInvocationID.x][10] = (-shared_step2[gl_LocalInvocationID.x][10] + shared_step2[gl_LocalInvocationID.x][11]);
    shared_step1[gl_LocalInvocationID.x][11] = (shared_step2[gl_LocalInvocationID.x][10] + shared_step2[gl_LocalInvocationID.x][11]);
    shared_step1[gl_LocalInvocationID.x][12] = (shared_step2[gl_LocalInvocationID.x][12] + shared_step2[gl_LocalInvocationID.x][13]);
    shared_step1[gl_LocalInvocationID.x][13] = (shared_step2[gl_LocalInvocationID.x][12] - shared_step2[gl_LocalInvocationID.x][13]);
    shared_step1[gl_LocalInvocationID.x][14] = (-shared_step2[gl_LocalInvocationID.x][14] + shared_step2[gl_LocalInvocationID.x][15]);
    shared_step1[gl_LocalInvocationID.x][15] = (shared_step2[gl_LocalInvocationID.x][14] + shared_step2[gl_LocalInvocationID.x][15]);
    
    // stage 4
    temp1 = (shared_step1[gl_LocalInvocationID.x][0] + shared_step1[gl_LocalInvocationID.x][1]) * 11585;
    temp2 = (shared_step1[gl_LocalInvocationID.x][0] - shared_step1[gl_LocalInvocationID.x][1]) * 11585;
    shared_step2[gl_LocalInvocationID.x][0] = ROUND_SHIFT(temp1);
    shared_step2[gl_LocalInvocationID.x][1] = ROUND_SHIFT(temp2);
    temp1 = shared_step1[gl_LocalInvocationID.x][2] * 6270 - shared_step1[gl_LocalInvocationID.x][3] * 15137;
    temp2 = shared_step1[gl_LocalInvocationID.x][2] * 15137 + shared_step1[gl_LocalInvocationID.x][3] * 6270;
    shared_step2[gl_LocalInvocationID.x][2] = ROUND_SHIFT(temp1);
    shared_step2[gl_LocalInvocationID.x][3] = ROUND_SHIFT(temp2);
    shared_step2[gl_LocalInvocationID.x][4] = (shared_step1[gl_LocalInvocationID.x][4] + shared_step1[gl_LocalInvocationID.x][5]);
    shared_step2[gl_LocalInvocationID.x][5] = (shared_step1[gl_LocalInvocationID.x][4] - shared_step1[gl_LocalInvocationID.x][5]);
    shared_step2[gl_LocalInvocationID.x][6] = (-shared_step1[gl_LocalInvocationID.x][6] + shared_step1[gl_LocalInvocationID.x][7]);
    shared_step2[gl_LocalInvocationID.x][7] = (shared_step1[gl_LocalInvocationID.x][6] + shared_step1[gl_LocalInvocationID.x][7]);
    
    shared_step2[gl_LocalInvocationID.x][8] = shared_step1[gl_LocalInvocationID.x][8];
    shared_step2[gl_LocalInvocationID.x][15] = shared_step1[gl_LocalInvocationID.x][15];
    temp1 = -shared_step1[gl_LocalInvocationID.x][9] * 15137 + shared_step1[gl_LocalInvocationID.x][14] * 6270;
    temp2 = shared_step1[gl_LocalInvocationID.x][9] * 6270 + shared_step1[gl_LocalInvocationID.x][14] * 15137;
    shared_step2[gl_LocalInvocationID.x][9] = ROUND_SHIFT(temp1);
    shared_step2[gl_LocalInvocationID.x][14] = ROUND_SHIFT(temp2);
    temp1 = -shared_step1[gl_LocalInvocationID.x][10] * 6270 - shared_step1[gl_LocalInvocationID.x][13] * 15137;
    temp2 = -shared_step1[gl_LocalInvocationID.x][10] * 15137 + shared_step1[gl_LocalInvocationID.x][13] * 6270;
    shared_step2[gl_LocalInvocationID.x][10] = ROUND_SHIFT(temp1);
    shared_step2[gl_LocalInvocationID.x][13] = ROUND_SHIFT(temp2);
    shared_step2[gl_LocalInvocationID.x][11] = shared_step1[gl_LocalInvocationID.x][11];
    shared_step2[gl_LocalInvocationID.x][12] = shared_step1[gl_LocalInvocationID.x][12];
    
    // stage 5
    shared_step1[gl_LocalInvocationID.x][0] = (shared_step2[gl_LocalInvocationID.x][0] + shared_step2[gl_LocalInvocationID.x][3]);
    shared_step1[gl_LocalInvocationID.x][1] = (shared_step2[gl_LocalInvocationID.x][1] + shared_step2[gl_LocalInvocationID.x][2]);
    shared_step1[gl_LocalInvocationID.x][2] = (shared_step2[gl_LocalInvocationID.x][1] - shared_step2[gl_LocalInvocationID.x][2]);
    shared_step1[gl_LocalInvocationID.x][3] = (shared_step2[gl_LocalInvocationID.x][0] - shared_step2[gl_LocalInvocationID.x][3]);
    shared_step1[gl_LocalInvocationID.x][4] = shared_step2[gl_LocalInvocationID.x][4];
    temp1 = (shared_step2[gl_LocalInvocationID.x][6] - shared_step2[gl_LocalInvocationID.x][5]) * 11585;
    temp2 = (shared_step2[gl_LocalInvocationID.x][5] + shared_step2[gl_LocalInvocationID.x][6]) * 11585;
    shared_step1[gl_LocalInvocationID.x][5] = ROUND_SHIFT(temp1);
    shared_step1[gl_LocalInvocationID.x][6] = ROUND_SHIFT(temp2);
    shared_step1[gl_LocalInvocationID.x][7] = shared_step2[gl_LocalInvocationID.x][7];
    
    shared_step1[gl_LocalInvocationID.x][8] = (shared_step2[gl_LocalInvocationID.x][8] + shared_step2[gl_LocalInvocationID.x][11]);
    shared_step1[gl_LocalInvocationID.x][9] = (shared_step2[gl_LocalInvocationID.x][9] + shared_step2[gl_LocalInvocationID.x][10]);
    shared_step1[gl_LocalInvocationID.x][10] = (shared_step2[gl_LocalInvocationID.x][9] - shared_step2[gl_LocalInvocationID.x][10]);
    shared_step1[gl_LocalInvocationID.x][11] = (shared_step2[gl_LocalInvocationID.x][8] - shared_step2[gl_LocalInvocationID.x][11]);
    shared_step1[gl_LocalInvocationID.x][12] = (-shared_step2[gl_LocalInvocationID.x][12] + shared_step2[gl_LocalInvocationID.x][15]);
    shared_step1[gl_LocalInvocationID.x][13] = (-shared_step2[gl_LocalInvocationID.x][13] + shared_step2[gl_LocalInvocationID.x][14]);
    shared_step1[gl_LocalInvocationID.x][14] = (shared_step2[gl_LocalInvocationID.x][13] + shared_step2[gl_LocalInvocationID.x][14]);
    shared_step1[gl_LocalInvocationID.x][15] = (shared_step2[gl_LocalInvocationID.x][12] + shared_step2[gl_LocalInvocationID.x][15]);
    
    // stage 6
    shared_step2[gl_LocalInvocationID.x][0] = (shared_step1[gl_LocalInvocationID.x][0] + shared_step1[gl_LocalInvocationID.x][7]);
    shared_step2[gl_LocalInvocationID.x][1] = (shared_step1[gl_LocalInvocationID.x][1] + shared_step1[gl_LocalInvocationID.x][6]);
    shared_step2[gl_LocalInvocationID.x][2] = (shared_step1[gl_LocalInvocationID.x][2] + shared_step1[gl_LocalInvocationID.x][5]);
    shared_step2[gl_LocalInvocationID.x][3] = (shared_step1[gl_LocalInvocationID.x][3] + shared_step1[gl_LocalInvocationID.x][4]);
    shared_step2[gl_LocalInvocationID.x][4] = (shared_step1[gl_LocalInvocationID.x][3] - shared_step1[gl_LocalInvocationID.x][4]);
    shared_step2[gl_LocalInvocationID.x][5] = (shared_step1[gl_LocalInvocationID.x][2] - shared_step1[gl_LocalInvocationID.x][5]);
    shared_step2[gl_LocalInvocationID.x][6] = (shared_step1[gl_LocalInvocationID.x][1] - shared_step1[gl_LocalInvocationID.x][6]);
    shared_step2[gl_LocalInvocationID.x][7] = (shared_step1[gl_LocalInvocationID.x][0] - shared_step1[gl_LocalInvocationID.x][7]);
    shared_step2[gl_LocalInvocationID.x][8] = shared_step1[gl_LocalInvocationID.x][8];
    shared_step2[gl_LocalInvocationID.x][9] = shared_step1[gl_LocalInvocationID.x][9];
    temp1 = (-shared_step1[gl_LocalInvocationID.x][10] + shared_step1[gl_LocalInvocationID.x][13]) * 11585;
    temp2 = (shared_step1[gl_LocalInvocationID.x][10] + shared_step1[gl_LocalInvocationID.x][13]) * 11585;
    shared_step2[gl_LocalInvocationID.x][10] = ROUND_SHIFT(temp1);
    shared_step2[gl_LocalInvocationID.x][13] = ROUND_SHIFT(temp2);
    temp1 = (-shared_step1[gl_LocalInvocationID.x][11] + shared_step1[gl_LocalInvocationID.x][12]) * 11585;
    temp2 = (shared_step1[gl_LocalInvocationID.x][11] + shared_step1[gl_LocalInvocationID.x][12]) * 11585;
    shared_step2[gl_LocalInvocationID.x][11] = ROUND_SHIFT(temp1);
    shared_step2[gl_LocalInvocationID.x][12] = ROUND_SHIFT(temp2);
    shared_step2[gl_LocalInvocationID.x][14] = shared_step1[gl_LocalInvocationID.x][14];
    shared_step2[gl_LocalInvocationID.x][15] = shared_step1[gl_LocalInvocationID.x][15];
    
    // stage 7
    output_arr[0] = (shared_step2[gl_LocalInvocationID.x][0] + shared_step2[gl_LocalInvocationID.x][15]);
    output_arr[1] = (shared_step2[gl_LocalInvocationID.x][1] + shared_step2[gl_LocalInvocationID.x][14]);
    output_arr[2] = (shared_step2[gl_LocalInvocationID.x][2] + shared_step2[gl_LocalInvocationID.x][13]);
    output_arr[3] = (shared_step2[gl_LocalInvocationID.x][3] + shared_step2[gl_LocalInvocationID.x][12]);
    output_arr[4] = (shared_step2[gl_LocalInvocationID.x][4] + shared_step2[gl_LocalInvocationID.x][11]);
    output_arr[5] = (shared_step2[gl_LocalInvocationID.x][5] + shared_step2[gl_LocalInvocationID.x][10]);
    output_arr[6] = (shared_step2[gl_LocalInvocationID.x][6] + shared_step2[gl_LocalInvocationID.x][9]);
    output_arr[7] = (shared_step2[gl_LocalInvocationID.x][7] + shared_step2[gl_LocalInvocationID.x][8]);
    output_arr[8] = (shared_step2[gl_LocalInvocationID.x][7] - shared_step2[gl_LocalInvocationID.x][8]);
    output_arr[9] = (shared_step2[gl_LocalInvocationID.x][6] - shared_step2[gl_LocalInvocationID.x][9]);
    output_arr[10] = (shared_step2[gl_LocalInvocationID.x][5] - shared_step2[gl_LocalInvocationID.x][10]);
    output_arr[11] = (shared_step2[gl_LocalInvocationID.x][4] - shared_step2[gl_LocalInvocationID.x][11]);
    output_arr[12] = (shared_step2[gl_LocalInvocationID.x][3] - shared_step2[gl_LocalInvocationID.x][12]);
    output_arr[13] = (shared_step2[gl_LocalInvocationID.x][2] - shared_step2[gl_LocalInvocationID.x][13]);
    output_arr[14] = (shared_step2[gl_LocalInvocationID.x][1] - shared_step2[gl_LocalInvocationID.x][14]);
    output_arr[15] = (shared_step2[gl_LocalInvocationID.x][0] - shared_step2[gl_LocalInvocationID.x][15]);
}

void idct32_c(int input_arr[32], inout int output_arr[32]) {
    int temp1, temp2;
    
    // stage 1
    shared_step1[gl_LocalInvocationID.x][0] = input_arr[0];
    shared_step1[gl_LocalInvocationID.x][1] = input_arr[16];
    shared_step1[gl_LocalInvocationID.x][2] = input_arr[8];
    shared_step1[gl_LocalInvocationID.x][3] = input_arr[24];
    shared_step1[gl_LocalInvocationID.x][4] = input_arr[4];
    shared_step1[gl_LocalInvocationID.x][5] = input_arr[20];
    shared_step1[gl_LocalInvocationID.x][6] = input_arr[12];
    shared_step1[gl_LocalInvocationID.x][7] = input_arr[28];
    shared_step1[gl_LocalInvocationID.x][8] = input_arr[2];
    shared_step1[gl_LocalInvocationID.x][9] = input_arr[18];
    shared_step1[gl_LocalInvocationID.x][10] = input_arr[10];
    shared_step1[gl_LocalInvocationID.x][11] = input_arr[26];
    shared_step1[gl_LocalInvocationID.x][12] = input_arr[6];
    shared_step1[gl_LocalInvocationID.x][13] = input_arr[22];
    shared_step1[gl_LocalInvocationID.x][14] = input_arr[14];
    shared_step1[gl_LocalInvocationID.x][15] = input_arr[30];
    
    temp1 = input_arr[1] * 804 - input_arr[31] * 16364;
    temp2 = input_arr[1] * 16364 + input_arr[31] * 804;
    shared_step1[gl_LocalInvocationID.x][16] = ROUND_SHIFT(temp1);
    shared_step1[gl_LocalInvocationID.x][31] = ROUND_SHIFT(temp2);
    
    temp1 = input_arr[17] * 12140 - input_arr[15] * 11003;
    temp2 = input_arr[17] * 11003 + input_arr[15] * 12140;
    shared_step1[gl_LocalInvocationID.x][17] = ROUND_SHIFT(temp1);
    shared_step1[gl_LocalInvocationID.x][30] = ROUND_SHIFT(temp2);
    
    temp1 = input_arr[9] * 7005 - input_arr[23] * 14811;
    temp2 = input_arr[9] * 14811 + input_arr[23] * 7005;
    shared_step1[gl_LocalInvocationID.x][18] = ROUND_SHIFT(temp1);
    shared_step1[gl_LocalInvocationID.x][29] = ROUND_SHIFT(temp2);
    
    temp1 = input_arr[25] * 15426 - input_arr[7] * 5520;
    temp2 = input_arr[25] * 5520 + input_arr[7] * 15426;
    shared_step1[gl_LocalInvocationID.x][19] = ROUND_SHIFT(temp1);
    shared_step1[gl_LocalInvocationID.x][28] = ROUND_SHIFT(temp2);
    
    temp1 = input_arr[5] * 3981 - input_arr[27] * 15893;
    temp2 = input_arr[5] * 15893 + input_arr[27] * 3981;
    shared_step1[gl_LocalInvocationID.x][20] = ROUND_SHIFT(temp1);
    shared_step1[gl_LocalInvocationID.x][27] = ROUND_SHIFT(temp2);
    
    temp1 = input_arr[21] * 14053 - input_arr[11] * 8423;
    temp2 = input_arr[21] * 8423 + input_arr[11] * 14053;
    shared_step1[gl_LocalInvocationID.x][21] = ROUND_SHIFT(temp1);
    shared_step1[gl_LocalInvocationID.x][26] = ROUND_SHIFT(temp2);
    
    temp1 = input_arr[13] * 9760 - input_arr[19] * 13160;
    temp2 = input_arr[13] * 13160 + input_arr[19] * 9760;
    shared_step1[gl_LocalInvocationID.x][22] = ROUND_SHIFT(temp1);
    shared_step1[gl_LocalInvocationID.x][25] = ROUND_SHIFT(temp2);
    
    temp1 = input_arr[29] * 16207 - input_arr[3] * 2404;
    temp2 = input_arr[29] * 2404 + input_arr[3] * 16207;
    shared_step1[gl_LocalInvocationID.x][23] = ROUND_SHIFT(temp1);
    shared_step1[gl_LocalInvocationID.x][24] = ROUND_SHIFT(temp2);
    
    // stage 2
    shared_step2[gl_LocalInvocationID.x][0] = shared_step1[gl_LocalInvocationID.x][0];
    shared_step2[gl_LocalInvocationID.x][1] = shared_step1[gl_LocalInvocationID.x][1];
    shared_step2[gl_LocalInvocationID.x][2] = shared_step1[gl_LocalInvocationID.x][2];
    shared_step2[gl_LocalInvocationID.x][3] = shared_step1[gl_LocalInvocationID.x][3];
    shared_step2[gl_LocalInvocationID.x][4] = shared_step1[gl_LocalInvocationID.x][4];
    shared_step2[gl_LocalInvocationID.x][5] = shared_step1[gl_LocalInvocationID.x][5];
    shared_step2[gl_LocalInvocationID.x][6] = shared_step1[gl_LocalInvocationID.x][6];
    shared_step2[gl_LocalInvocationID.x][7] = shared_step1[gl_LocalInvocationID.x][7];
    
    temp1 = shared_step1[gl_LocalInvocationID.x][8] * 1606 - shared_step1[gl_LocalInvocationID.x][15] * 16305;
    temp2 = shared_step1[gl_LocalInvocationID.x][8] * 16305 + shared_step1[gl_LocalInvocationID.x][15] * 1606;
    shared_step2[gl_LocalInvocationID.x][8] = ROUND_SHIFT(temp1);
    shared_step2[gl_LocalInvocationID.x][15] = ROUND_SHIFT(temp2);
    
    temp1 = shared_step1[gl_LocalInvocationID.x][9] * 12665 - shared_step1[gl_LocalInvocationID.x][14] * 10394;
    temp2 = shared_step1[gl_LocalInvocationID.x][9] * 10394 + shared_step1[gl_LocalInvocationID.x][14] * 12665;
    shared_step2[gl_LocalInvocationID.x][9] = ROUND_SHIFT(temp1);
    shared_step2[gl_LocalInvocationID.x][14] = ROUND_SHIFT(temp2);
    
    temp1 = shared_step1[gl_LocalInvocationID.x][10] * 7723 - shared_step1[gl_LocalInvocationID.x][13] * 14449;
    temp2 = shared_step1[gl_LocalInvocationID.x][10] * 14449 + shared_step1[gl_LocalInvocationID.x][13] * 7723;
    shared_step2[gl_LocalInvocationID.x][10] = ROUND_SHIFT(temp1);
    shared_step2[gl_LocalInvocationID.x][13] = ROUND_SHIFT(temp2);
    
    temp1 = shared_step1[gl_LocalInvocationID.x][11] * 15679 - shared_step1[gl_LocalInvocationID.x][12] * 4756;
    temp2 = shared_step1[gl_LocalInvocationID.x][11] * 4756 + shared_step1[gl_LocalInvocationID.x][12] * 15679;
    shared_step2[gl_LocalInvocationID.x][11] = ROUND_SHIFT(temp1);
    shared_step2[gl_LocalInvocationID.x][12] = ROUND_SHIFT(temp2);
    
    shared_step2[gl_LocalInvocationID.x][16] = (shared_step1[gl_LocalInvocationID.x][16] + shared_step1[gl_LocalInvocationID.x][17]);
    shared_step2[gl_LocalInvocationID.x][17] = (shared_step1[gl_LocalInvocationID.x][16] - shared_step1[gl_LocalInvocationID.x][17]);
    shared_step2[gl_LocalInvocationID.x][18] = (-shared_step1[gl_LocalInvocationID.x][18] + shared_step1[gl_LocalInvocationID.x][19]);
    shared_step2[gl_LocalInvocationID.x][19] = (shared_step1[gl_LocalInvocationID.x][18] + shared_step1[gl_LocalInvocationID.x][19]);
    shared_step2[gl_LocalInvocationID.x][20] = (shared_step1[gl_LocalInvocationID.x][20] + shared_step1[gl_LocalInvocationID.x][21]);
    shared_step2[gl_LocalInvocationID.x][21] = (shared_step1[gl_LocalInvocationID.x][20] - shared_step1[gl_LocalInvocationID.x][21]);
    shared_step2[gl_LocalInvocationID.x][22] = (-shared_step1[gl_LocalInvocationID.x][22] + shared_step1[gl_LocalInvocationID.x][23]);
    shared_step2[gl_LocalInvocationID.x][23] = (shared_step1[gl_LocalInvocationID.x][22] + shared_step1[gl_LocalInvocationID.x][23]);
    shared_step2[gl_LocalInvocationID.x][24] = (shared_step1[gl_LocalInvocationID.x][24] + shared_step1[gl_LocalInvocationID.x][25]);
    shared_step2[gl_LocalInvocationID.x][25] = (shared_step1[gl_LocalInvocationID.x][24] - shared_step1[gl_LocalInvocationID.x][25]);
    shared_step2[gl_LocalInvocationID.x][26] = (-shared_step1[gl_LocalInvocationID.x][26] + shared_step1[gl_LocalInvocationID.x][27]);
    shared_step2[gl_LocalInvocationID.x][27] = (shared_step1[gl_LocalInvocationID.x][26] + shared_step1[gl_LocalInvocationID.x][27]);
    shared_step2[gl_LocalInvocationID.x][28] = (shared_step1[gl_LocalInvocationID.x][28] + shared_step1[gl_LocalInvocationID.x][29]);
    shared_step2[gl_LocalInvocationID.x][29] = (shared_step1[gl_LocalInvocationID.x][28] - shared_step1[gl_LocalInvocationID.x][29]);
    shared_step2[gl_LocalInvocationID.x][30] = (-shared_step1[gl_LocalInvocationID.x][30] + shared_step1[gl_LocalInvocationID.x][31]);
    shared_step2[gl_LocalInvocationID.x][31] = (shared_step1[gl_LocalInvocationID.x][30] + shared_step1[gl_LocalInvocationID.x][31]);
    
    // stage 3
    shared_step1[gl_LocalInvocationID.x][0] = shared_step2[gl_LocalInvocationID.x][0];
    shared_step1[gl_LocalInvocationID.x][1] = shared_step2[gl_LocalInvocationID.x][1];
    shared_step1[gl_LocalInvocationID.x][2] = shared_step2[gl_LocalInvocationID.x][2];
    shared_step1[gl_LocalInvocationID.x][3] = shared_step2[gl_LocalInvocationID.x][3];
    
    temp1 = shared_step2[gl_LocalInvocationID.x][4] * 3196 - shared_step2[gl_LocalInvocationID.x][7] * 16069;
    temp2 = shared_step2[gl_LocalInvocationID.x][4] * 16069 + shared_step2[gl_LocalInvocationID.x][7] * 3196;
    shared_step1[gl_LocalInvocationID.x][4] = ROUND_SHIFT(temp1);
    shared_step1[gl_LocalInvocationID.x][7] = ROUND_SHIFT(temp2);
    temp1 = shared_step2[gl_LocalInvocationID.x][5] * 13623 - shared_step2[gl_LocalInvocationID.x][6] * 9102;
    temp2 = shared_step2[gl_LocalInvocationID.x][5] * 9102 + shared_step2[gl_LocalInvocationID.x][6] * 13623;
    shared_step1[gl_LocalInvocationID.x][5] = ROUND_SHIFT(temp1);
    shared_step1[gl_LocalInvocationID.x][6] = ROUND_SHIFT(temp2);
    
    shared_step1[gl_LocalInvocationID.x][8] = (shared_step2[gl_LocalInvocationID.x][8] + shared_step2[gl_LocalInvocationID.x][9]);
    shared_step1[gl_LocalInvocationID.x][9] = (shared_step2[gl_LocalInvocationID.x][8] - shared_step2[gl_LocalInvocationID.x][9]);
    shared_step1[gl_LocalInvocationID.x][10] = (-shared_step2[gl_LocalInvocationID.x][10] + shared_step2[gl_LocalInvocationID.x][11]);
    shared_step1[gl_LocalInvocationID.x][11] = (shared_step2[gl_LocalInvocationID.x][10] + shared_step2[gl_LocalInvocationID.x][11]);
    shared_step1[gl_LocalInvocationID.x][12] = (shared_step2[gl_LocalInvocationID.x][12] + shared_step2[gl_LocalInvocationID.x][13]);
    shared_step1[gl_LocalInvocationID.x][13] = (shared_step2[gl_LocalInvocationID.x][12] - shared_step2[gl_LocalInvocationID.x][13]);
    shared_step1[gl_LocalInvocationID.x][14] = (-shared_step2[gl_LocalInvocationID.x][14] + shared_step2[gl_LocalInvocationID.x][15]);
    shared_step1[gl_LocalInvocationID.x][15] = (shared_step2[gl_LocalInvocationID.x][14] + shared_step2[gl_LocalInvocationID.x][15]);
    
    shared_step1[gl_LocalInvocationID.x][16] = shared_step2[gl_LocalInvocationID.x][16];
    shared_step1[gl_LocalInvocationID.x][31] = shared_step2[gl_LocalInvocationID.x][31];
    temp1 = -shared_step2[gl_LocalInvocationID.x][17] * 16069 + shared_step2[gl_LocalInvocationID.x][30] * 3196;
    temp2 = shared_step2[gl_LocalInvocationID.x][17] * 3196 + shared_step2[gl_LocalInvocationID.x][30] * 16069;
    shared_step1[gl_LocalInvocationID.x][17] = ROUND_SHIFT(temp1);
    shared_step1[gl_LocalInvocationID.x][30] = ROUND_SHIFT(temp2);
    temp1 = -shared_step2[gl_LocalInvocationID.x][18] * 3196 - shared_step2[gl_LocalInvocationID.x][29] * 16069;
    temp2 = -shared_step2[gl_LocalInvocationID.x][18] * 16069 + shared_step2[gl_LocalInvocationID.x][29] * 3196;
    shared_step1[gl_LocalInvocationID.x][18] = ROUND_SHIFT(temp1);
    shared_step1[gl_LocalInvocationID.x][29] = ROUND_SHIFT(temp2);
    shared_step1[gl_LocalInvocationID.x][19] = shared_step2[gl_LocalInvocationID.x][19];
    shared_step1[gl_LocalInvocationID.x][20] = shared_step2[gl_LocalInvocationID.x][20];
    temp1 = -shared_step2[gl_LocalInvocationID.x][21] * 9102 + shared_step2[gl_LocalInvocationID.x][26] * 13623;
    temp2 = shared_step2[gl_LocalInvocationID.x][21] * 13623 + shared_step2[gl_LocalInvocationID.x][26] * 9102;
    shared_step1[gl_LocalInvocationID.x][21] = ROUND_SHIFT(temp1);
    shared_step1[gl_LocalInvocationID.x][26] = ROUND_SHIFT(temp2);
    temp1 = -shared_step2[gl_LocalInvocationID.x][22] * 13623 - shared_step2[gl_LocalInvocationID.x][25] * 9102;
    temp2 = -shared_step2[gl_LocalInvocationID.x][22] * 9102 + shared_step2[gl_LocalInvocationID.x][25] * 13623;
    shared_step1[gl_LocalInvocationID.x][22] = ROUND_SHIFT(temp1);
    shared_step1[gl_LocalInvocationID.x][25] = ROUND_SHIFT(temp2);
    shared_step1[gl_LocalInvocationID.x][23] = shared_step2[gl_LocalInvocationID.x][23];
    shared_step1[gl_LocalInvocationID.x][24] = shared_step2[gl_LocalInvocationID.x][24];
    shared_step1[gl_LocalInvocationID.x][27] = shared_step2[gl_LocalInvocationID.x][27];
    shared_step1[gl_LocalInvocationID.x][28] = shared_step2[gl_LocalInvocationID.x][28];
    
    // stage 4
    temp1 = (shared_step1[gl_LocalInvocationID.x][0] + shared_step1[gl_LocalInvocationID.x][1]) * 11585;
    temp2 = (shared_step1[gl_LocalInvocationID.x][0] - shared_step1[gl_LocalInvocationID.x][1]) * 11585;
    shared_step2[gl_LocalInvocationID.x][0] = ROUND_SHIFT(temp1);
    shared_step2[gl_LocalInvocationID.x][1] = ROUND_SHIFT(temp2);
    temp1 = shared_step1[gl_LocalInvocationID.x][2] * 6270 - shared_step1[gl_LocalInvocationID.x][3] * 15137;
    temp2 = shared_step1[gl_LocalInvocationID.x][2] * 15137 + shared_step1[gl_LocalInvocationID.x][3] * 6270;
    shared_step2[gl_LocalInvocationID.x][2] = ROUND_SHIFT(temp1);
    shared_step2[gl_LocalInvocationID.x][3] = ROUND_SHIFT(temp2);
    shared_step2[gl_LocalInvocationID.x][4] = (shared_step1[gl_LocalInvocationID.x][4] + shared_step1[gl_LocalInvocationID.x][5]);
    shared_step2[gl_LocalInvocationID.x][5] = (shared_step1[gl_LocalInvocationID.x][4] - shared_step1[gl_LocalInvocationID.x][5]);
    shared_step2[gl_LocalInvocationID.x][6] = (-shared_step1[gl_LocalInvocationID.x][6] + shared_step1[gl_LocalInvocationID.x][7]);
    shared_step2[gl_LocalInvocationID.x][7] = (shared_step1[gl_LocalInvocationID.x][6] + shared_step1[gl_LocalInvocationID.x][7]);
    
    shared_step2[gl_LocalInvocationID.x][8] = shared_step1[gl_LocalInvocationID.x][8];
    shared_step2[gl_LocalInvocationID.x][15] = shared_step1[gl_LocalInvocationID.x][15];
    temp1 = -shared_step1[gl_LocalInvocationID.x][9] * 15137 + shared_step1[gl_LocalInvocationID.x][14] * 6270;
    temp2 = shared_step1[gl_LocalInvocationID.x][9] * 6270 + shared_step1[gl_LocalInvocationID.x][14] * 15137;
    shared_step2[gl_LocalInvocationID.x][9] = ROUND_SHIFT(temp1);
    shared_step2[gl_LocalInvocationID.x][14] = ROUND_SHIFT(temp2);
    temp1 = -shared_step1[gl_LocalInvocationID.x][10] * 6270 - shared_step1[gl_LocalInvocationID.x][13] * 15137;
    temp2 = -shared_step1[gl_LocalInvocationID.x][10] * 15137 + shared_step1[gl_LocalInvocationID.x][13] * 6270;
    shared_step2[gl_LocalInvocationID.x][10] = ROUND_SHIFT(temp1);
    shared_step2[gl_LocalInvocationID.x][13] = ROUND_SHIFT(temp2);
    shared_step2[gl_LocalInvocationID.x][11] = shared_step1[gl_LocalInvocationID.x][11];
    shared_step2[gl_LocalInvocationID.x][12] = shared_step1[gl_LocalInvocationID.x][12];
    
    shared_step2[gl_LocalInvocationID.x][16] = (shared_step1[gl_LocalInvocationID.x][16] + shared_step1[gl_LocalInvocationID.x][19]);
    shared_step2[gl_LocalInvocationID.x][17] = (shared_step1[gl_LocalInvocationID.x][17] + shared_step1[gl_LocalInvocationID.x][18]);
    shared_step2[gl_LocalInvocationID.x][18] = (shared_step1[gl_LocalInvocationID.x][17] - shared_step1[gl_LocalInvocationID.x][18]);
    shared_step2[gl_LocalInvocationID.x][19] = (shared_step1[gl_LocalInvocationID.x][16] - shared_step1[gl_LocalInvocationID.x][19]);
    shared_step2[gl_LocalInvocationID.x][20] = (-shared_step1[gl_LocalInvocationID.x][20] + shared_step1[gl_LocalInvocationID.x][23]);
    shared_step2[gl_LocalInvocationID.x][21] = (-shared_step1[gl_LocalInvocationID.x][21] + shared_step1[gl_LocalInvocationID.x][22]);
    shared_step2[gl_LocalInvocationID.x][22] = (shared_step1[gl_LocalInvocationID.x][21] + shared_step1[gl_LocalInvocationID.x][22]);
    shared_step2[gl_LocalInvocationID.x][23] = (shared_step1[gl_LocalInvocationID.x][20] + shared_step1[gl_LocalInvocationID.x][23]);
    
    shared_step2[gl_LocalInvocationID.x][24] = (shared_step1[gl_LocalInvocationID.x][24] + shared_step1[gl_LocalInvocationID.x][27]);
    shared_step2[gl_LocalInvocationID.x][25] = (shared_step1[gl_LocalInvocationID.x][25] + shared_step1[gl_LocalInvocationID.x][26]);
    shared_step2[gl_LocalInvocationID.x][26] = (shared_step1[gl_LocalInvocationID.x][25] - shared_step1[gl_LocalInvocationID.x][26]);
    shared_step2[gl_LocalInvocationID.x][27] = (shared_step1[gl_LocalInvocationID.x][24] - shared_step1[gl_LocalInvocationID.x][27]);
    shared_step2[gl_LocalInvocationID.x][28] = (-shared_step1[gl_LocalInvocationID.x][28] + shared_step1[gl_LocalInvocationID.x][31]);
    shared_step2[gl_LocalInvocationID.x][29] = (-shared_step1[gl_LocalInvocationID.x][29] + shared_step1[gl_LocalInvocationID.x][30]);
    shared_step2[gl_LocalInvocationID.x][30] = (shared_step1[gl_LocalInvocationID.x][29] + shared_step1[gl_LocalInvocationID.x][30]);
    shared_step2[gl_LocalInvocationID.x][31] = (shared_step1[gl_LocalInvocationID.x][28] + shared_step1[gl_LocalInvocationID.x][31]);
    
    // stage 5
    shared_step1[gl_LocalInvocationID.x][0] = (shared_step2[gl_LocalInvocationID.x][0] + shared_step2[gl_LocalInvocationID.x][3]);
    shared_step1[gl_LocalInvocationID.x][1] = (shared_step2[gl_LocalInvocationID.x][1] + shared_step2[gl_LocalInvocationID.x][2]);
    shared_step1[gl_LocalInvocationID.x][2] = (shared_step2[gl_LocalInvocationID.x][1] - shared_step2[gl_LocalInvocationID.x][2]);
    shared_step1[gl_LocalInvocationID.x][3] = (shared_step2[gl_LocalInvocationID.x][0] - shared_step2[gl_LocalInvocationID.x][3]);
    shared_step1[gl_LocalInvocationID.x][4] = shared_step2[gl_LocalInvocationID.x][4];
    temp1 = (shared_step2[gl_LocalInvocationID.x][6] - shared_step2[gl_LocalInvocationID.x][5]) * 11585;
    temp2 = (shared_step2[gl_LocalInvocationID.x][5] + shared_step2[gl_LocalInvocationID.x][6]) * 11585;
    shared_step1[gl_LocalInvocationID.x][5] = ROUND_SHIFT(temp1);
    shared_step1[gl_LocalInvocationID.x][6] = ROUND_SHIFT(temp2);
    shared_step1[gl_LocalInvocationID.x][7] = shared_step2[gl_LocalInvocationID.x][7];
    
    shared_step1[gl_LocalInvocationID.x][8] = (shared_step2[gl_LocalInvocationID.x][8] + shared_step2[gl_LocalInvocationID.x][11]);
    shared_step1[gl_LocalInvocationID.x][9] = (shared_step2[gl_LocalInvocationID.x][9] + shared_step2[gl_LocalInvocationID.x][10]);
    shared_step1[gl_LocalInvocationID.x][10] = (shared_step2[gl_LocalInvocationID.x][9] - shared_step2[gl_LocalInvocationID.x][10]);
    shared_step1[gl_LocalInvocationID.x][11] = (shared_step2[gl_LocalInvocationID.x][8] - shared_step2[gl_LocalInvocationID.x][11]);
    shared_step1[gl_LocalInvocationID.x][12] = (-shared_step2[gl_LocalInvocationID.x][12] + shared_step2[gl_LocalInvocationID.x][15]);
    shared_step1[gl_LocalInvocationID.x][13] = (-shared_step2[gl_LocalInvocationID.x][13] + shared_step2[gl_LocalInvocationID.x][14]);
    shared_step1[gl_LocalInvocationID.x][14] = (shared_step2[gl_LocalInvocationID.x][13] + shared_step2[gl_LocalInvocationID.x][14]);
    shared_step1[gl_LocalInvocationID.x][15] = (shared_step2[gl_LocalInvocationID.x][12] + shared_step2[gl_LocalInvocationID.x][15]);
    
    shared_step1[gl_LocalInvocationID.x][16] = shared_step2[gl_LocalInvocationID.x][16];
    shared_step1[gl_LocalInvocationID.x][17] = shared_step2[gl_LocalInvocationID.x][17];
    temp1 = -shared_step2[gl_LocalInvocationID.x][18] * 15137 + shared_step2[gl_LocalInvocationID.x][29] * 6270;
    temp2 = shared_step2[gl_LocalInvocationID.x][18] * 6270 + shared_step2[gl_LocalInvocationID.x][29] * 15137;
    shared_step1[gl_LocalInvocationID.x][18] = ROUND_SHIFT(temp1);
    shared_step1[gl_LocalInvocationID.x][29] = ROUND_SHIFT(temp2);
    temp1 = -shared_step2[gl_LocalInvocationID.x][19] * 15137 + shared_step2[gl_LocalInvocationID.x][28] * 6270;
    temp2 = shared_step2[gl_LocalInvocationID.x][19] * 6270 + shared_step2[gl_LocalInvocationID.x][28] * 15137;
    shared_step1[gl_LocalInvocationID.x][19] = ROUND_SHIFT(temp1);
    shared_step1[gl_LocalInvocationID.x][28] = ROUND_SHIFT(temp2);
    temp1 = -shared_step2[gl_LocalInvocationID.x][20] * 6270 - shared_step2[gl_LocalInvocationID.x][27] * 15137;
    temp2 = -shared_step2[gl_LocalInvocationID.x][20] * 15137 + shared_step2[gl_LocalInvocationID.x][27] * 6270;
    shared_step1[gl_LocalInvocationID.x][20] = ROUND_SHIFT(temp1);
    shared_step1[gl_LocalInvocationID.x][27] = ROUND_SHIFT(temp2);
    temp1 = -shared_step2[gl_LocalInvocationID.x][21] * 6270 - shared_step2[gl_LocalInvocationID.x][26] * 15137;
    temp2 = -shared_step2[gl_LocalInvocationID.x][21] * 15137 + shared_step2[gl_LocalInvocationID.x][26] * 6270;
    shared_step1[gl_LocalInvocationID.x][21] = ROUND_SHIFT(temp1);
    shared_step1[gl_LocalInvocationID.x][26] = ROUND_SHIFT(temp2);
    shared_step1[gl_LocalInvocationID.x][22] = shared_step2[gl_LocalInvocationID.x][22];
    shared_step1[gl_LocalInvocationID.x][23] = shared_step2[gl_LocalInvocationID.x][23];
    shared_step1[gl_LocalInvocationID.x][24] = shared_step2[gl_LocalInvocationID.x][24];
    shared_step1[gl_LocalInvocationID.x][25] = shared_step2[gl_LocalInvocationID.x][25];
    shared_step1[gl_LocalInvocationID.x][30] = shared_step2[gl_LocalInvocationID.x][30];
    shared_step1[gl_LocalInvocationID.x][31] = shared_step2[gl_LocalInvocationID.x][31];
    
    // stage 6
    shared_step2[gl_LocalInvocationID.x][0] = (shared_step1[gl_LocalInvocationID.x][0] + shared_step1[gl_LocalInvocationID.x][7]);
    shared_step2[gl_LocalInvocationID.x][1] = (shared_step1[gl_LocalInvocationID.x][1] + shared_step1[gl_LocalInvocationID.x][6]);
    shared_step2[gl_LocalInvocationID.x][2] = (shared_step1[gl_LocalInvocationID.x][2] + shared_step1[gl_LocalInvocationID.x][5]);
    shared_step2[gl_LocalInvocationID.x][3] = (shared_step1[gl_LocalInvocationID.x][3] + shared_step1[gl_LocalInvocationID.x][4]);
    shared_step2[gl_LocalInvocationID.x][4] = (shared_step1[gl_LocalInvocationID.x][3] - shared_step1[gl_LocalInvocationID.x][4]);
    shared_step2[gl_LocalInvocationID.x][5] = (shared_step1[gl_LocalInvocationID.x][2] - shared_step1[gl_LocalInvocationID.x][5]);
    shared_step2[gl_LocalInvocationID.x][6] = (shared_step1[gl_LocalInvocationID.x][1] - shared_step1[gl_LocalInvocationID.x][6]);
    shared_step2[gl_LocalInvocationID.x][7] = (shared_step1[gl_LocalInvocationID.x][0] - shared_step1[gl_LocalInvocationID.x][7]);
    shared_step2[gl_LocalInvocationID.x][8] = shared_step1[gl_LocalInvocationID.x][8];
    shared_step2[gl_LocalInvocationID.x][9] = shared_step1[gl_LocalInvocationID.x][9];
    temp1 = (-shared_step1[gl_LocalInvocationID.x][10] + shared_step1[gl_LocalInvocationID.x][13]) * 11585;
    temp2 = (shared_step1[gl_LocalInvocationID.x][10] + shared_step1[gl_LocalInvocationID.x][13]) * 11585;
    shared_step2[gl_LocalInvocationID.x][10] = ROUND_SHIFT(temp1);
    shared_step2[gl_LocalInvocationID.x][13] = ROUND_SHIFT(temp2);
    temp1 = (-shared_step1[gl_LocalInvocationID.x][11] + shared_step1[gl_LocalInvocationID.x][12]) * 11585;
    temp2 = (shared_step1[gl_LocalInvocationID.x][11] + shared_step1[gl_LocalInvocationID.x][12]) * 11585;
    shared_step2[gl_LocalInvocationID.x][11] = ROUND_SHIFT(temp1);
    shared_step2[gl_LocalInvocationID.x][12] = ROUND_SHIFT(temp2);
    shared_step2[gl_LocalInvocationID.x][14] = shared_step1[gl_LocalInvocationID.x][14];
    shared_step2[gl_LocalInvocationID.x][15] = shared_step1[gl_LocalInvocationID.x][15];
    
    shared_step2[gl_LocalInvocationID.x][16] = (shared_step1[gl_LocalInvocationID.x][16] + shared_step1[gl_LocalInvocationID.x][23]);
    shared_step2[gl_LocalInvocationID.x][17] = (shared_step1[gl_LocalInvocationID.x][17] + shared_step1[gl_LocalInvocationID.x][22]);
    shared_step2[gl_LocalInvocationID.x][18] = (shared_step1[gl_LocalInvocationID.x][18] + shared_step1[gl_LocalInvocationID.x][21]);
    shared_step2[gl_LocalInvocationID.x][19] = (shared_step1[gl_LocalInvocationID.x][19] + shared_step1[gl_LocalInvocationID.x][20]);
    shared_step2[gl_LocalInvocationID.x][20] = (shared_step1[gl_LocalInvocationID.x][19] - shared_step1[gl_LocalInvocationID.x][20]);
    shared_step2[gl_LocalInvocationID.x][21] = (shared_step1[gl_LocalInvocationID.x][18] - shared_step1[gl_LocalInvocationID.x][21]);
    shared_step2[gl_LocalInvocationID.x][22] = (shared_step1[gl_LocalInvocationID.x][17] - shared_step1[gl_LocalInvocationID.x][22]);
    shared_step2[gl_LocalInvocationID.x][23] = (shared_step1[gl_LocalInvocationID.x][16] - shared_step1[gl_LocalInvocationID.x][23]);
    
    shared_step2[gl_LocalInvocationID.x][24] = (-shared_step1[gl_LocalInvocationID.x][24] + shared_step1[gl_LocalInvocationID.x][31]);
    shared_step2[gl_LocalInvocationID.x][25] = (-shared_step1[gl_LocalInvocationID.x][25] + shared_step1[gl_LocalInvocationID.x][30]);
    shared_step2[gl_LocalInvocationID.x][26] = (-shared_step1[gl_LocalInvocationID.x][26] + shared_step1[gl_LocalInvocationID.x][29]);
    shared_step2[gl_LocalInvocationID.x][27] = (-shared_step1[gl_LocalInvocationID.x][27] + shared_step1[gl_LocalInvocationID.x][28]);
    shared_step2[gl_LocalInvocationID.x][28] = (shared_step1[gl_LocalInvocationID.x][27] + shared_step1[gl_LocalInvocationID.x][28]);
    shared_step2[gl_LocalInvocationID.x][29] = (shared_step1[gl_LocalInvocationID.x][26] + shared_step1[gl_LocalInvocationID.x][29]);
    shared_step2[gl_LocalInvocationID.x][30] = (shared_step1[gl_LocalInvocationID.x][25] + shared_step1[gl_LocalInvocationID.x][30]);
    shared_step2[gl_LocalInvocationID.x][31] = (shared_step1[gl_LocalInvocationID.x][24] + shared_step1[gl_LocalInvocationID.x][31]);
    
    // stage 7
    shared_step1[gl_LocalInvocationID.x][0] = (shared_step2[gl_LocalInvocationID.x][0] + shared_step2[gl_LocalInvocationID.x][15]);
    shared_step1[gl_LocalInvocationID.x][1] = (shared_step2[gl_LocalInvocationID.x][1] + shared_step2[gl_LocalInvocationID.x][14]);
    shared_step1[gl_LocalInvocationID.x][2] = (shared_step2[gl_LocalInvocationID.x][2] + shared_step2[gl_LocalInvocationID.x][13]);
    shared_step1[gl_LocalInvocationID.x][3] = (shared_step2[gl_LocalInvocationID.x][3] + shared_step2[gl_LocalInvocationID.x][12]);
    shared_step1[gl_LocalInvocationID.x][4] = (shared_step2[gl_LocalInvocationID.x][4] + shared_step2[gl_LocalInvocationID.x][11]);
    shared_step1[gl_LocalInvocationID.x][5] = (shared_step2[gl_LocalInvocationID.x][5] + shared_step2[gl_LocalInvocationID.x][10]);
    shared_step1[gl_LocalInvocationID.x][6] = (shared_step2[gl_LocalInvocationID.x][6] + shared_step2[gl_LocalInvocationID.x][9]);
    shared_step1[gl_LocalInvocationID.x][7] = (shared_step2[gl_LocalInvocationID.x][7] + shared_step2[gl_LocalInvocationID.x][8]);
    shared_step1[gl_LocalInvocationID.x][8] = (shared_step2[gl_LocalInvocationID.x][7] - shared_step2[gl_LocalInvocationID.x][8]);
    shared_step1[gl_LocalInvocationID.x][9] = (shared_step2[gl_LocalInvocationID.x][6] - shared_step2[gl_LocalInvocationID.x][9]);
    shared_step1[gl_LocalInvocationID.x][10] = (shared_step2[gl_LocalInvocationID.x][5] - shared_step2[gl_LocalInvocationID.x][10]);
    shared_step1[gl_LocalInvocationID.x][11] = (shared_step2[gl_LocalInvocationID.x][4] - shared_step2[gl_LocalInvocationID.x][11]);
    shared_step1[gl_LocalInvocationID.x][12] = (shared_step2[gl_LocalInvocationID.x][3] - shared_step2[gl_LocalInvocationID.x][12]);
    shared_step1[gl_LocalInvocationID.x][13] = (shared_step2[gl_LocalInvocationID.x][2] - shared_step2[gl_LocalInvocationID.x][13]);
    shared_step1[gl_LocalInvocationID.x][14] = (shared_step2[gl_LocalInvocationID.x][1] - shared_step2[gl_LocalInvocationID.x][14]);
    shared_step1[gl_LocalInvocationID.x][15] = (shared_step2[gl_LocalInvocationID.x][0] - shared_step2[gl_LocalInvocationID.x][15]);
    
    shared_step1[gl_LocalInvocationID.x][16] = shared_step2[gl_LocalInvocationID.x][16];
    shared_step1[gl_LocalInvocationID.x][17] = shared_step2[gl_LocalInvocationID.x][17];
    shared_step1[gl_LocalInvocationID.x][18] = shared_step2[gl_LocalInvocationID.x][18];
    shared_step1[gl_LocalInvocationID.x][19] = shared_step2[gl_LocalInvocationID.x][19];
    temp1 = (-shared_step2[gl_LocalInvocationID.x][20] + shared_step2[gl_LocalInvocationID.x][27]) * 11585;
    temp2 = (shared_step2[gl_LocalInvocationID.x][20] + shared_step2[gl_LocalInvocationID.x][27]) * 11585;
    shared_step1[gl_LocalInvocationID.x][20] = ROUND_SHIFT(temp1);
    shared_step1[gl_LocalInvocationID.x][27] = ROUND_SHIFT(temp2);
    temp1 = (-shared_step2[gl_LocalInvocationID.x][21] + shared_step2[gl_LocalInvocationID.x][26]) * 11585;
    temp2 = (shared_step2[gl_LocalInvocationID.x][21] + shared_step2[gl_LocalInvocationID.x][26]) * 11585;
    shared_step1[gl_LocalInvocationID.x][21] = ROUND_SHIFT(temp1);
    shared_step1[gl_LocalInvocationID.x][26] = ROUND_SHIFT(temp2);
    temp1 = (-shared_step2[gl_LocalInvocationID.x][22] + shared_step2[gl_LocalInvocationID.x][25]) * 11585;
    temp2 = (shared_step2[gl_LocalInvocationID.x][22] + shared_step2[gl_LocalInvocationID.x][25]) * 11585;
    shared_step1[gl_LocalInvocationID.x][22] = ROUND_SHIFT(temp1);
    shared_step1[gl_LocalInvocationID.x][25] = ROUND_SHIFT(temp2);
    temp1 = (-shared_step2[gl_LocalInvocationID.x][23] + shared_step2[gl_LocalInvocationID.x][24]) * 11585;
    temp2 = (shared_step2[gl_LocalInvocationID.x][23] + shared_step2[gl_LocalInvocationID.x][24]) * 11585;
    shared_step1[gl_LocalInvocationID.x][23] = ROUND_SHIFT(temp1);
    shared_step1[gl_LocalInvocationID.x][24] = ROUND_SHIFT(temp2);
    shared_step1[gl_LocalInvocationID.x][28] = shared_step2[gl_LocalInvocationID.x][28];
    shared_step1[gl_LocalInvocationID.x][29] = shared_step2[gl_LocalInvocationID.x][29];
    shared_step1[gl_LocalInvocationID.x][30] = shared_step2[gl_LocalInvocationID.x][30];
    shared_step1[gl_LocalInvocationID.x][31] = shared_step2[gl_LocalInvocationID.x][31];
    
    // final stage
    output_arr[0] = (shared_step1[gl_LocalInvocationID.x][0] + shared_step1[gl_LocalInvocationID.x][31]);
    output_arr[1] = (shared_step1[gl_LocalInvocationID.x][1] + shared_step1[gl_LocalInvocationID.x][30]);
    output_arr[2] = (shared_step1[gl_LocalInvocationID.x][2] + shared_step1[gl_LocalInvocationID.x][29]);
    output_arr[3] = (shared_step1[gl_LocalInvocationID.x][3] + shared_step1[gl_LocalInvocationID.x][28]);
    output_arr[4] = (shared_step1[gl_LocalInvocationID.x][4] + shared_step1[gl_LocalInvocationID.x][27]);
    output_arr[5] = (shared_step1[gl_LocalInvocationID.x][5] + shared_step1[gl_LocalInvocationID.x][26]);
    output_arr[6] = (shared_step1[gl_LocalInvocationID.x][6] + shared_step1[gl_LocalInvocationID.x][25]);
    output_arr[7] = (shared_step1[gl_LocalInvocationID.x][7] + shared_step1[gl_LocalInvocationID.x][24]);
    output_arr[8] = (shared_step1[gl_LocalInvocationID.x][8] + shared_step1[gl_LocalInvocationID.x][23]);
    output_arr[9] = (shared_step1[gl_LocalInvocationID.x][9] + shared_step1[gl_LocalInvocationID.x][22]);
    output_arr[10] = (shared_step1[gl_LocalInvocationID.x][10] + shared_step1[gl_LocalInvocationID.x][21]);
    output_arr[11] = (shared_step1[gl_LocalInvocationID.x][11] + shared_step1[gl_LocalInvocationID.x][20]);
    output_arr[12] = (shared_step1[gl_LocalInvocationID.x][12] + shared_step1[gl_LocalInvocationID.x][19]);
    output_arr[13] = (shared_step1[gl_LocalInvocationID.x][13] + shared_step1[gl_LocalInvocationID.x][18]);
    output_arr[14] = (shared_step1[gl_LocalInvocationID.x][14] + shared_step1[gl_LocalInvocationID.x][17]);
    output_arr[15] = (shared_step1[gl_LocalInvocationID.x][15] + shared_step1[gl_LocalInvocationID.x][16]);
    output_arr[16] = (shared_step1[gl_LocalInvocationID.x][15] - shared_step1[gl_LocalInvocationID.x][16]);
    output_arr[17] = (shared_step1[gl_LocalInvocationID.x][14] - shared_step1[gl_LocalInvocationID.x][17]);
    output_arr[18] = (shared_step1[gl_LocalInvocationID.x][13] - shared_step1[gl_LocalInvocationID.x][18]);
    output_arr[19] = (shared_step1[gl_LocalInvocationID.x][12] - shared_step1[gl_LocalInvocationID.x][19]);
    output_arr[20] = (shared_step1[gl_LocalInvocationID.x][11] - shared_step1[gl_LocalInvocationID.x][20]);
    output_arr[21] = (shared_step1[gl_LocalInvocationID.x][10] - shared_step1[gl_LocalInvocationID.x][21]);
    output_arr[22] = (shared_step1[gl_LocalInvocationID.x][9] - shared_step1[gl_LocalInvocationID.x][22]);
    output_arr[23] = (shared_step1[gl_LocalInvocationID.x][8] - shared_step1[gl_LocalInvocationID.x][23]);
    output_arr[24] = (shared_step1[gl_LocalInvocationID.x][7] - shared_step1[gl_LocalInvocationID.x][24]);
    output_arr[25] = (shared_step1[gl_LocalInvocationID.x][6] - shared_step1[gl_LocalInvocationID.x][25]);
    output_arr[26] = (shared_step1[gl_LocalInvocationID.x][5] - shared_step1[gl_LocalInvocationID.x][26]);
    output_arr[27] = (shared_step1[gl_LocalInvocationID.x][4] - shared_step1[gl_LocalInvocationID.x][27]);
    output_arr[28] = (shared_step1[gl_LocalInvocationID.x][3] - shared_step1[gl_LocalInvocationID.x][28]);
    output_arr[29] = (shared_step1[gl_LocalInvocationID.x][2] - shared_step1[gl_LocalInvocationID.x][29]);
    output_arr[30] = (shared_step1[gl_LocalInvocationID.x][1] - shared_step1[gl_LocalInvocationID.x][30]);
    output_arr[31] = (shared_step1[gl_LocalInvocationID.x][0] - shared_step1[gl_LocalInvocationID.x][31]);
}


shared int shared_data[32][33];

void main()
{
    uint block_idx = gl_WorkGroupID.x;
    BlockData b = blocks[block_idx];
    
    if (b.skip != 0) return;
    
    uint tid = gl_LocalInvocationID.x;
    uint N = b.tx_size;

    if (N == 4) {
        int arr[4];
        if (tid < 4) {
            for (int row = 0; row < 4; row++) {
                arr[row] = int(coeff[b.coeff_offset + row * 4 + tid]) * b.qstep;
            }
            idct4_c(arr, arr);
            for (int row = 0; row < 4; row++) {
                shared_data[row][tid] = arr[row];
            }
        }
        barrier();
        if (tid < 4) {
            for (int col = 0; col < 4; col++) {
                arr[col] = shared_data[tid][col];
            }
            idct4_c(arr, arr);
            for (int col = 0; col < 4; col++) {
                shared_data[tid][col] = arr[col];
            }
        }
        barrier();
        if (tid < 4) {
            for (int row = 0; row < 4; row++) {
                int val = shared_data[row][tid];
                int rounded = (val + 8) >> 4;
                uint pixel_offset = b.dst_offset + row * b.dst_stride + tid;
                int pred_val = int(dst_pixels[pixel_offset]);
                dst_pixels[pixel_offset] = uint8_t(clamp(pred_val + rounded, 0, 255));
            }
        }
    } else if (N == 8) {
        int arr[8];
        if (tid < 8) {
            for (int row = 0; row < 8; row++) {
                arr[row] = int(coeff[b.coeff_offset + row * 8 + tid]) * b.qstep;
            }
            idct8_c(arr, arr);
            for (int row = 0; row < 8; row++) {
                shared_data[row][tid] = arr[row];
            }
        }
        barrier();
        if (tid < 8) {
            for (int col = 0; col < 8; col++) {
                arr[col] = shared_data[tid][col];
            }
            idct8_c(arr, arr);
            for (int col = 0; col < 8; col++) {
                shared_data[tid][col] = arr[col];
            }
        }
        barrier();
        if (tid < 8) {
            for (int row = 0; row < 8; row++) {
                int val = shared_data[row][tid];
                int rounded = (val + 16) >> 5;
                uint pixel_offset = b.dst_offset + row * b.dst_stride + tid;
                int pred_val = int(dst_pixels[pixel_offset]);
                dst_pixels[pixel_offset] = uint8_t(clamp(pred_val + rounded, 0, 255));
            }
        }
    } else if (N == 16) {
        int arr[16];
        if (tid < 16) {
            for (int row = 0; row < 16; row++) {
                arr[row] = int(coeff[b.coeff_offset + row * 16 + tid]) * b.qstep;
            }
            idct16_c(arr, arr);
            for (int row = 0; row < 16; row++) {
                shared_data[row][tid] = arr[row];
            }
        }
        barrier();
        if (tid < 16) {
            for (int col = 0; col < 16; col++) {
                arr[col] = shared_data[tid][col];
            }
            idct16_c(arr, arr);
            for (int col = 0; col < 16; col++) {
                shared_data[tid][col] = arr[col];
            }
        }
        barrier();
        if (tid < 16) {
            for (int row = 0; row < 16; row++) {
                int val = shared_data[row][tid];
                int rounded = (val + 32) >> 6;
                uint pixel_offset = b.dst_offset + row * b.dst_stride + tid;
                int pred_val = int(dst_pixels[pixel_offset]);
                dst_pixels[pixel_offset] = uint8_t(clamp(pred_val + rounded, 0, 255));
            }
        }
    } else if (N == 32) {
        int arr[32];
        if (tid < 32) {
            for (int row = 0; row < 32; row++) {
                arr[row] = int(coeff[b.coeff_offset + row * 32 + tid]) * b.qstep;
            }
            idct32_c(arr, arr);
            for (int row = 0; row < 32; row++) {
                shared_data[row][tid] = arr[row];
            }
        }
        barrier();
        if (tid < 32) {
            for (int col = 0; col < 32; col++) {
                arr[col] = shared_data[tid][col];
            }
            idct32_c(arr, arr);
            for (int col = 0; col < 32; col++) {
                shared_data[tid][col] = arr[col];
            }
        }
        barrier();
        if (tid < 32) {
            for (int row = 0; row < 32; row++) {
                int val = shared_data[row][tid];
                int rounded = (val + 32) >> 6;
                uint pixel_offset = b.dst_offset + row * b.dst_stride + tid;
                int pred_val = int(dst_pixels[pixel_offset]);
                dst_pixels[pixel_offset] = uint8_t(clamp(pred_val + rounded, 0, 255));
            }
        }
    }
}
