/**
 * compute-vp9 — VP9 entropy decoding headers
 */
#pragma once

#include <stdint.h>
#include <stdbool.h>
#include "vpx_reader.h"
#include "vp9_parsed_frame.h"

#define TX_SIZES 4
#define PLANE_TYPES 2
#define REF_TYPES 2
#define COEF_BANDS 6
#define COEFF_CONTEXTS 3
#define UNCONSTRAINED_NODES 3

/* VP9 coefficient token definitions */
typedef enum {
    EOB_TOKEN = 0,
    ZERO_TOKEN = 1,
    ONE_TOKEN = 2,
    TWO_TOKEN = 3,
    THREE_TOKEN = 4,
    FOUR_TOKEN = 5,
    DCT_VAL_CAT1 = 6,
    DCT_VAL_CAT2 = 7,
    DCT_VAL_CAT3 = 8,
    DCT_VAL_CAT4 = 9,
    DCT_VAL_CAT5 = 10,
    DCT_VAL_CAT6 = 11,
    NUM_TOKENS = 12
} vp9_token_t;

/* VP9 prediction modes */
typedef enum {
    DC_PRED = 0,
    V_PRED = 1,
    H_PRED = 2,
    D45_PRED = 3,
    D135_PRED = 4,
    D117_PRED = 5,
    D127_PRED = 6,
    D207_PRED = 7,
    D63_PRED = 8,
    TM_PRED = 9,
    NEARESTMV = 10,
    NEARMV = 11,
    ZEROMV = 12,
    NEWMV = 13
} vp9_prediction_mode_t;

/* Probability context tables */
typedef struct {
    uint8_t coef_probs[TX_SIZES][PLANE_TYPES][REF_TYPES][COEF_BANDS][COEFF_CONTEXTS][UNCONSTRAINED_NODES];
    uint8_t skip_probs[3];
    uint8_t intra_inter_probs[4];
    uint8_t tx_probs[3][2];
    uint8_t partition_probs[16][3];
    uint8_t y_mode_probs[4][9];
    uint8_t uv_mode_probs[10][9];
    uint8_t inter_mode_probs[7][3];
} vp9_entropy_probs_t;

/* Scan tables mapping scan-index to raster-index */
extern uint16_t vp9_default_scan_4x4[16];
extern uint16_t vp9_default_scan_8x8[64];
extern uint16_t vp9_default_scan_16x16[256];
extern uint16_t vp9_default_scan_32x32[1024];

/* ── Entropy API ─────────────────────────────────────────────────────────── */

/**
 * Initialize default VP9 entropy probabilities.
 */
void vp9_entropy_probs_init(vp9_entropy_probs_t *probs);

/**
 * Traverses the token tree to read a coefficient token.
 */
int vp9_read_coef_token(vpx_reader *r, const uint8_t probs[UNCONSTRAINED_NODES]);

/**
 * Decodes the absolute value and sign of a token.
 */
int32_t vp9_decode_coef_value(vpx_reader *r, int token, int bit_depth);

/**
 * Decodes a single transform block of coefficients.
 * Writes coefficients in raster order into 'out_coeffs'.
 * Returns the end of block (EOB) index.
 */
int vp9_decode_tx_block(vpx_reader *r, int tx_size, int plane_type, int ref_type,
                        const vp9_entropy_probs_t *probs, int neighbor_context,
                        int16_t *out_coeffs);

/**
 * Decodes the intra prediction mode using standard trees.
 */
int vp9_read_intra_mode(vpx_reader *r, const uint8_t probs[9]);

/**
 * Decodes the inter prediction mode using standard trees.
 */
int vp9_read_inter_mode(vpx_reader *r, const uint8_t probs[3]);

/**
 * Adapts frame context probabilities dynamically based on decoded symbol counts.
 */
void vp9_adapt_probabilities(vp9_entropy_probs_t *probs, const vp9_parsed_frame_t *pf);


