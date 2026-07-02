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
 * Decodes a single keyframe tile on the spec-conformant path (vp9_kf_tile.c).
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
