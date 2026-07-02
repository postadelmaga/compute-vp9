/**
 * compute-vp9 — VP9 entropy decoding implementation
 */
#include "vp9_entropy.h"
#include <stdlib.h>
#include <string.h>
#include "vp9_parsed_frame.h"
#include "vp9_default_probs.h"

/* Scan tables allocated globally and generated dynamically on first use */
uint16_t vp9_default_scan_4x4[16];
uint16_t vp9_default_scan_8x8[64];
uint16_t vp9_default_scan_16x16[256];
uint16_t vp9_default_scan_32x32[1024];

static bool scans_initialized = false;

static void generate_zigzag_scan(uint16_t *scan, int size)
{
    int i = 0;
    for (int sum = 0; sum < 2 * size - 1; sum++) {
        if (sum % 2 == 1) { /* down-left */
            for (int r = 0; r < size; r++) {
                int c = sum - r;
                if (c >= 0 && c < size) {
                    scan[i++] = r * size + c;
                }
            }
        } else { /* up-right */
            for (int c = 0; c < size; c++) {
                int r = sum - c;
                if (r >= 0 && r < size) {
                    scan[i++] = r * size + c;
                }
            }
        }
    }
}

static void init_scans(void)
{
    if (scans_initialized) return;
    generate_zigzag_scan(vp9_default_scan_4x4, 4);
    generate_zigzag_scan(vp9_default_scan_8x8, 8);
    generate_zigzag_scan(vp9_default_scan_16x16, 16);
    generate_zigzag_scan(vp9_default_scan_32x32, 32);
    scans_initialized = true;
}

void vp9_entropy_probs_init(vp9_entropy_probs_t *probs)
{
    init_scans();

    /* Spec default frame context (libvpx tables) */
    memcpy(probs->coef_probs[0], vp9_def_coef_probs_4x4, sizeof(vp9_def_coef_probs_4x4));
    memcpy(probs->coef_probs[1], vp9_def_coef_probs_8x8, sizeof(vp9_def_coef_probs_8x8));
    memcpy(probs->coef_probs[2], vp9_def_coef_probs_16x16, sizeof(vp9_def_coef_probs_16x16));
    memcpy(probs->coef_probs[3], vp9_def_coef_probs_32x32, sizeof(vp9_def_coef_probs_32x32));

    memcpy(probs->skip_probs, vp9_def_skip_probs, sizeof(probs->skip_probs));
    memcpy(probs->intra_inter_probs, vp9_def_intra_inter_probs, sizeof(probs->intra_inter_probs));
    memcpy(probs->tx8_probs, vp9_def_tx8_probs, sizeof(probs->tx8_probs));
    memcpy(probs->tx16_probs, vp9_def_tx16_probs, sizeof(probs->tx16_probs));
    memcpy(probs->tx32_probs, vp9_def_tx32_probs, sizeof(probs->tx32_probs));
    memcpy(probs->partition_probs, vp9_def_partition_probs, sizeof(probs->partition_probs));
    memcpy(probs->y_mode_probs, vp9_def_if_y_probs, sizeof(probs->y_mode_probs));
    memcpy(probs->uv_mode_probs, vp9_def_if_uv_probs, sizeof(probs->uv_mode_probs));
    memcpy(probs->inter_mode_probs, vp9_def_inter_mode_probs, sizeof(probs->inter_mode_probs));
    memcpy(probs->switchable_interp_probs, vp9_def_switchable_interp_probs,
           sizeof(probs->switchable_interp_probs));
    memcpy(probs->comp_inter_probs, vp9_def_comp_inter_probs, sizeof(probs->comp_inter_probs));
    memcpy(probs->single_ref_probs, vp9_def_single_ref_probs, sizeof(probs->single_ref_probs));
    memcpy(probs->comp_ref_probs, vp9_def_comp_ref_probs, sizeof(probs->comp_ref_probs));

    memcpy(probs->mv_joint_probs, vp9_def_mv_joint_probs, sizeof(probs->mv_joint_probs));
    memcpy(probs->mv_sign_probs, vp9_def_mv_sign_probs, sizeof(probs->mv_sign_probs));
    memcpy(probs->mv_class_probs, vp9_def_mv_class_probs, sizeof(probs->mv_class_probs));
    memcpy(probs->mv_class0_probs, vp9_def_mv_class0_probs, sizeof(probs->mv_class0_probs));
    memcpy(probs->mv_bits_probs, vp9_def_mv_bits_probs, sizeof(probs->mv_bits_probs));
    memcpy(probs->mv_class0_fr_probs, vp9_def_mv_class0_fr_probs, sizeof(probs->mv_class0_fr_probs));
    memcpy(probs->mv_fr_probs, vp9_def_mv_fr_probs, sizeof(probs->mv_fr_probs));
    memcpy(probs->mv_class0_hp_probs, vp9_def_mv_class0_hp_probs, sizeof(probs->mv_class0_hp_probs));
    memcpy(probs->mv_hp_probs, vp9_def_mv_hp_probs, sizeof(probs->mv_hp_probs));

    probs->tx_mode = VP9_TX_MODE_SELECT;
    probs->reference_mode = VP9_SINGLE_REFERENCE;
}

/* VP9 coefficient Huffman tree decisions */
int vp9_read_coef_token(vpx_reader *r, const uint8_t probs[UNCONSTRAINED_NODES])
{
    /* Node 0: EOB vs others */
    if (!vpx_read(r, probs[0]))
        return EOB_TOKEN;

    /* Node 1: ZERO vs others */
    if (!vpx_read(r, probs[1]))
        return ZERO_TOKEN;

    /* Node 2: ONE vs others */
    if (!vpx_read(r, probs[2]))
        return ONE_TOKEN;

    /* Tree structure for category selection */
    /* Node 3: TWO vs others */
    if (!vpx_read(r, 128)) /* unconstrained in standard contexts or custom prob */
        return TWO_TOKEN;

    /* Node 4: THREE vs others */
    if (!vpx_read(r, 128))
        return THREE_TOKEN;

    /* Node 5: FOUR vs others */
    if (!vpx_read(r, 128))
        return FOUR_TOKEN;

    /* Nodes for CAT1 to CAT6 */
    if (!vpx_read(r, 128)) {
        /* Node 6: CAT1 vs CAT2 */
        if (!vpx_read(r, 128))
            return DCT_VAL_CAT1;
        else
            return DCT_VAL_CAT2;
    } else {
        /* Node 7: CAT3/4 vs CAT5/6 */
        if (!vpx_read(r, 128)) {
            if (!vpx_read(r, 128))
                return DCT_VAL_CAT3;
            else
                return DCT_VAL_CAT4;
        } else {
            if (!vpx_read(r, 128))
                return DCT_VAL_CAT5;
            else
                return DCT_VAL_CAT6;
        }
    }
}

int32_t vp9_decode_coef_value(vpx_reader *r, int token, int bit_depth)
{
    if (token <= ZERO_TOKEN) return 0;
    if (token == ONE_TOKEN)  return 1;
    if (token == TWO_TOKEN)  return 2;
    if (token == THREE_TOKEN) return 3;
    if (token == FOUR_TOKEN)  return 4;

    int32_t val = 0;
    int extra_bits = 0;
    int32_t base = 0;

    switch (token) {
    case DCT_VAL_CAT1:
        base = 5;
        extra_bits = 1;
        break;
    case DCT_VAL_CAT2:
        base = 7;
        extra_bits = 2;
        break;
    case DCT_VAL_CAT3:
        base = 11;
        extra_bits = 3;
        break;
    case DCT_VAL_CAT4:
        base = 19;
        extra_bits = 4;
        break;
    case DCT_VAL_CAT5:
        base = 35;
        extra_bits = 5;
        break;
    case DCT_VAL_CAT6:
        base = 67;
        extra_bits = 11 + (bit_depth - 8);
        break;
    default:
        return 0;
    }

    /* Read extra bits from range coder */
    int32_t extra = 0;
    for (int i = 0; i < extra_bits; i++) {
        extra = (extra << 1) | vpx_read_bit(r);
    }
    
    val = base + extra;
    return val;
}

int vp9_decode_tx_block(vpx_reader *r, int tx_size, int plane_type, int ref_type,
                        const vp9_entropy_probs_t *probs, int neighbor_context,
                        int16_t *out_coeffs)
{
    init_scans();

    int max_coeffs = 1 << (2 * tx_size + 4); /* 16, 64, 256, 1024 */
    const uint16_t *scan = NULL;

    switch (tx_size) {
    case 0: scan = vp9_default_scan_4x4; break;
    case 1: scan = vp9_default_scan_8x8; break;
    case 2: scan = vp9_default_scan_16x16; break;
    case 3: scan = vp9_default_scan_32x32; break;
    default: return 0;
    }

    memset(out_coeffs, 0, max_coeffs * sizeof(int16_t));

    int eob = 0;
    int prev_val = 0;

    for (int i = 0; i < max_coeffs; i++) {
        /* Determine context band */
        int band;
        if (i == 0)      band = 0;
        else if (i == 1) band = 1;
        else if (i < 6)  band = 2;
        else if (i < 14) band = 3;
        else if (i < 30) band = 4;
        else             band = 5;

        /* Derive coefficient context from previous parsed value (except for DC) */
        int coef_ctx;
        if (i == 0) {
            coef_ctx = neighbor_context;
        } else {
            coef_ctx = (prev_val == 0) ? 0 : (prev_val == 1) ? 1 : 2;
        }

        const uint8_t *node_probs = probs->coef_probs[tx_size][plane_type][ref_type][band][coef_ctx];

        int token = vp9_read_coef_token(r, node_probs);
        if (token == EOB_TOKEN) {
            eob = i;
            break;
        }

        int32_t val = vp9_decode_coef_value(r, token, 8); /* Default to 8-bit depth */
        prev_val = val;

        if (val > 0) {
            int sign = vpx_read_bit(r);
            int16_t signed_val = (sign) ? -val : val;
            out_coeffs[scan[i]] = signed_val;
        }

        eob = i + 1;
    }

    return eob;
}

int vp9_read_intra_mode(vpx_reader *r, const uint8_t probs[9])
{
    /* Spec vp9_intra_mode_tree (libvpx vp9_entropymode.c) */
    static const int8_t tree[] = {
        -DC_PRED,   2,
        -TM_PRED,   4,
        -V_PRED,    6,
        8,          12,
        -H_PRED,    10,
        -D135_PRED, -D117_PRED,
        -D45_PRED,  14,
        -D63_PRED,  16,
        -D127_PRED, -D207_PRED   /* D127 == spec D153 slot */
    };

    int i = 0;
    while ((i = tree[i + vpx_read(r, probs[i >> 1])]) > 0)
        ;
    return -i;
}

int vp9_read_inter_mode(vpx_reader *r, const uint8_t probs[3])
{
    static const int8_t tree[] = {
        -NEARESTMV, 2,
        -NEARMV, 4,
        -ZEROMV, -NEWMV
    };
    
    int i = 0;
    while ((i = tree[i + vpx_read(r, probs[i >> 1])]) > 0)
        ;
    return -i;
}

static uint8_t adapt_prob(uint8_t prob, int count0, int count1)
{
    int total = count0 + count1;
    if (total == 0) return prob;
    
    int percentage = (count0 * 256 + (total >> 1)) / total;
    if (percentage < 1) percentage = 1;
    if (percentage > 255) percentage = 255;
    
    int factor = 15;
    return (uint8_t)((prob * factor + percentage) / (factor + 1));
}

void vp9_adapt_probabilities(vp9_entropy_probs_t *probs, const vp9_parsed_frame_t *pf)
{
    if (pf->num_blocks == 0) return;

    /* 1. Adapt skip probabilities */
    int skip_count[2] = {0};
    /* 2. Adapt intra/inter probabilities */
    int intra_inter_count[2] = {0};

    for (uint32_t i = 0; i < pf->num_blocks; i++) {
        const vp9_macroblock_info_t *block = &pf->blocks[i];
        if (block->skip) skip_count[1]++; else skip_count[0]++;
        if (block->is_intra) intra_inter_count[0]++; else intra_inter_count[1]++;
    }

    probs->skip_probs[0] = adapt_prob(probs->skip_probs[0], skip_count[0], skip_count[1]);
    probs->intra_inter_probs[0] = adapt_prob(probs->intra_inter_probs[0], intra_inter_count[0], intra_inter_count[1]);
}


