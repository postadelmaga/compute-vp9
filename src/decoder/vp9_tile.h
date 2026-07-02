/**
 * compute-vp9 — VP9 tile parsing headers
 */
#pragma once

#include "vp9_bitstream.h"
#include "vp9_parsed_frame.h"
#include "vp9_entropy.h"

/* ── Partition types ─────────────────────────────────────────────────────── */
typedef enum {
    PARTITION_NONE = 0,
    PARTITION_HORZ = 1,
    PARTITION_VERT = 2,
    PARTITION_SPLIT = 3
} vp9_partition_t;

/* ── Tile parser API ─────────────────────────────────────────────────────── */

/**
 * Decodes all tiles in a VP9 frame bitstream package.
 * Populates the 'pf' structure with macroblock metadata and coefficients.
 */
int vp9_decode_tiles(const vp9_frame_header_t *hdr,
                     const uint8_t *data, size_t size,
                     const vp9_entropy_probs_t *probs,
                     vp9_parsed_frame_t *pf);

/**
 * Spec-conformant tile decode, key and inter frames (vp9_kf_tile.c).
 * counts (nullable): symbol counts for backward adaptation.
 * prev_*: previous frame per-mi refs/MVs for the MV candidate scan.
 * out_*: per-mi refs/MVs of this frame (for the next frame's scan).
 */
int vp9_decode_tile_conformant(vpx_reader *r,
                               int mi_row_start, int mi_row_end,
                               int mi_col_start, int mi_col_end,
                               const vp9_entropy_probs_t *probs,
                               vp9_parsed_frame_t *pf,
                               vp9_counts_t *counts,
                               const int8_t *prev_ref0, const int8_t *prev_ref1,
                               const int16_t *prev_mv, int use_prev_mvs,
                               int8_t *out_ref0, int8_t *out_ref1, int16_t *out_mv);

/**
 * Convenience wrapper: conformant decode without counts/prev-MV state.
 */
int vp9_decode_tile_kf(vpx_reader *r,
                       int mi_row_start, int mi_row_end,
                       int mi_col_start, int mi_col_end,
                       const vp9_entropy_probs_t *probs,
                       vp9_parsed_frame_t *pf);

/**
 * Decodes a single tile (legacy approximate parser, inter frames).
 */
int vp9_decode_tile(vpx_reader *r,
                    int mi_row_start, int mi_row_end,
                    int mi_col_start, int mi_col_end,
                    const vp9_entropy_probs_t *probs,
                    vp9_parsed_frame_t *pf);

/**
 * Decodes partition recursively for a superblock.
 * size_mi is 8 (64x64), 4 (32x32), 2 (16x16), 1 (8x8).
 */
void vp9_decode_partition(vpx_reader *r, int mi_row, int mi_col, int size_mi,
                          const vp9_entropy_probs_t *probs,
                          vp9_parsed_frame_t *pf);

/**
 * Decodes a single mode-info block.
 */
void vp9_decode_block(vpx_reader *r, int mi_row, int mi_col,
                      int width_mi, int height_mi,
                      const vp9_entropy_probs_t *probs,
                      vp9_parsed_frame_t *pf);
