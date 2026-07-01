/**
 * compute-vp9 — structures for parsed frame data
 */
#pragma once

#include <stdint.h>
#include <stdbool.h>
#include "vp9_bitstream.h"

/* Motion vector matching GLSL ivec2 layout */
typedef struct {
    int32_t x;  /* 1/8-pixel precision horizontal MV component */
    int32_t y;  /* 1/8-pixel precision vertical MV component */
} cvp9_mv_t;

/* Prediction and block configuration metadata */
typedef struct {
    uint16_t x;             /* Pixel X position of block */
    uint16_t y;             /* Pixel Y position of block */
    uint8_t  width;         /* Block width in pixels (4 to 64) */
    uint8_t  height;        /* Block height in pixels (4 to 64) */
    uint8_t  tx_size;       /* Transform size (4, 8, 16, 32) */
    uint8_t  is_intra;      /* 1 for Intra-predicted, 0 for Inter-predicted */
    uint8_t  y_mode;        /* Y prediction mode (0-9) */
    uint8_t  uv_mode;       /* UV prediction mode (0-9) */
    uint8_t  ref_frame[2];  /* Reference frame indices */
    int16_t  mv[2][2];      /* Motion vectors for primary/secondary references */
    uint8_t  skip;          /* Skip coefficients flag (1 = skip, 0 = decode) */
    uint32_t coeff_offset;  /* Offset in the frame's coefficients buffer */
    uint16_t eob;           /* End of block (number of coefficients to transform) */
} vp9_macroblock_info_t;

/* Holds parsed information for a single frame */
typedef struct {
    vp9_frame_header_t      hdr;
    
    /* Decoded macroblocks metadata */
    vp9_macroblock_info_t  *blocks;
    uint32_t                num_blocks;
    uint32_t                blocks_capacity;

    /* Coefficients buffer (host-visible and mapped directly to GPU input buffer) */
    int16_t                *coeffs;
    uint32_t                num_coeffs;
    uint32_t                coeffs_capacity;

    /* Motion vector grid for GPU motion compensation (1/8-pel resolution per 4x4 grid block) */
    cvp9_mv_t              *mv_grid;
    uint32_t                mv_grid_width;
    uint32_t                mv_grid_height;
} vp9_parsed_frame_t;

/* ── Constructor/Destructor ──────────────────────────────────────────────── */
vp9_parsed_frame_t *vp9_parsed_frame_alloc(uint32_t width, uint32_t height);
void                vp9_parsed_frame_free(vp9_parsed_frame_t *pf);
void                vp9_parsed_frame_reset(vp9_parsed_frame_t *pf);
bool                vp9_parsed_frame_ensure_blocks(vp9_parsed_frame_t *pf, uint32_t count);
bool                vp9_parsed_frame_ensure_coeffs(vp9_parsed_frame_t *pf, uint32_t count);
