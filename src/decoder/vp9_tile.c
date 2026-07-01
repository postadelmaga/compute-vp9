/**
 * compute-vp9 — VP9 tile parsing implementation
 */
#include "vp9_tile.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

int vp9_decode_tiles(const vp9_frame_header_t *hdr,
                     const uint8_t *data, size_t size,
                     const vp9_entropy_probs_t *probs,
                     vp9_parsed_frame_t *pf)
{
    vp9_parsed_frame_reset(pf);
    memcpy(&pf->hdr, hdr, sizeof(*hdr));

    int mi_cols = (hdr->width + 7) >> 3;
    int mi_rows = (hdr->height + 7) >> 3;

    int sb_cols = (mi_cols + 7) >> 3;
    int sb_rows = (mi_rows + 7) >> 3;

    int tile_cols = 1 << hdr->log2_tile_cols;
    int tile_rows = 1 << hdr->log2_tile_rows;

    const uint8_t *curr_ptr = data;
    const uint8_t *data_end = data + size;

    /* Loop through tiles in raster order */
    for (int row = 0; row < tile_rows; row++) {
        int mi_row_start = ((row * sb_rows) >> hdr->log2_tile_rows) << 3;
        int mi_row_end = (((row + 1) * sb_rows) >> hdr->log2_tile_rows) << 3;
        if (mi_row_end > mi_rows) mi_row_end = mi_rows;

        for (int col = 0; col < tile_cols; col++) {
            int mi_col_start = ((col * sb_cols) >> hdr->log2_tile_cols) << 3;
            int mi_col_end = (((col + 1) * sb_cols) >> hdr->log2_tile_cols) << 3;
            if (mi_col_end > mi_cols) mi_col_end = mi_cols;

            size_t tile_size = 0;
            if (row == tile_rows - 1 && col == tile_cols - 1) {
                /* Last tile in frame contains all remaining data */
                tile_size = data_end - curr_ptr;
            } else {
                /* Non-last tiles have a 4-byte size header */
                if (curr_ptr + 4 > data_end) {
                    return -1;
                }
                tile_size = ((size_t)curr_ptr[0] << 24) |
                            ((size_t)curr_ptr[1] << 16) |
                            ((size_t)curr_ptr[2] << 8)  |
                            ((size_t)curr_ptr[3]);
                curr_ptr += 4;
            }

            if (curr_ptr + tile_size > data_end) {
                return -1;
            }

            /* Initialize range coder for this tile */
            vpx_reader r;
            if (vpx_reader_init(&r, curr_ptr, tile_size)) {
                return -1;
            }

            /* Decode tile slice */
            if (vp9_decode_tile(&r, mi_row_start, mi_row_end, mi_col_start, mi_col_end, probs, pf) != 0) {
                return -1;
            }

            curr_ptr += tile_size;
        }
    }

    return 0;
}

int vp9_decode_tile(vpx_reader *r,
                    int mi_row_start, int mi_row_end,
                    int mi_col_start, int mi_col_end,
                    const vp9_entropy_probs_t *probs,
                    vp9_parsed_frame_t *pf)
{
    /* Iterate over superblocks (64x64 pixels = 8x8 ModeInfo units) */
    for (int mi_row = mi_row_start; mi_row < mi_row_end; mi_row += 8) {
        for (int mi_col = mi_col_start; mi_col < mi_col_end; mi_col += 8) {
            vp9_decode_partition(r, mi_row, mi_col, 8, probs, pf);
        }
    }
    return 0;
}

void vp9_decode_partition(vpx_reader *r, int mi_row, int mi_col, int size_mi,
                          const vp9_entropy_probs_t *probs,
                          vp9_parsed_frame_t *pf)
{
    int mi_cols = pf->mv_grid_width * 2;
    int mi_rows = pf->mv_grid_height * 2;

    if (mi_row >= mi_rows || mi_col >= mi_cols) return;

    bool has_rows = (mi_row + size_mi) < mi_rows;
    bool has_cols = (mi_col + size_mi) < mi_cols;

    vp9_partition_t partition;

    if (size_mi == 1) {
        partition = PARTITION_NONE;
    } else if (has_rows && has_cols) {
        /* Standard 4-way split decision (simplified) */
        int bit0 = vpx_read(r, 128);
        int bit1 = vpx_read(r, 128);
        partition = (vp9_partition_t)((bit0 << 1) | bit1);
    } else if (!has_rows && has_cols) {
        partition = PARTITION_HORZ;
    } else if (has_rows && !has_cols) {
        partition = PARTITION_VERT;
    } else {
        partition = PARTITION_SPLIT;
    }

    int half = size_mi / 2;

    switch (partition) {
    case PARTITION_NONE:
        vp9_decode_block(r, mi_row, mi_col, size_mi, size_mi, probs, pf);
        break;
    case PARTITION_HORZ:
        vp9_decode_block(r, mi_row, mi_col, size_mi, half, probs, pf);
        if (mi_row + half < mi_rows) {
            vp9_decode_block(r, mi_row + half, mi_col, size_mi, half, probs, pf);
        }
        break;
    case PARTITION_VERT:
        vp9_decode_block(r, mi_row, mi_col, half, size_mi, probs, pf);
        if (mi_col + half < mi_cols) {
            vp9_decode_block(r, mi_row, mi_col + half, half, size_mi, probs, pf);
        }
        break;
    case PARTITION_SPLIT:
        vp9_decode_partition(r, mi_row, mi_col, half, probs, pf);
        vp9_decode_partition(r, mi_row, mi_col + half, half, probs, pf);
        vp9_decode_partition(r, mi_row + half, mi_col, half, probs, pf);
        vp9_decode_partition(r, mi_row + half, mi_col + half, half, probs, pf);
        break;
    }
}

void vp9_decode_block(vpx_reader *r, int mi_row, int mi_col,
                      int width_mi, int height_mi,
                      const vp9_entropy_probs_t *probs,
                      vp9_parsed_frame_t *pf)
{
    if (!vp9_parsed_frame_ensure_blocks(pf, 1)) return;

    vp9_macroblock_info_t *block = &pf->blocks[pf->num_blocks];
    block->x = mi_col * 8;
    block->y = mi_row * 8;
    block->width = width_mi * 8;
    block->height = height_mi * 8;

    /* Read skip flag */
    block->skip = vpx_read(r, probs->skip_probs[0]);

    /* Read intra/inter flag */
    block->is_intra = !vpx_read(r, probs->intra_inter_probs[0]);

    if (block->is_intra) {
        block->y_mode = vpx_read_literal(r, 4);
        if (block->y_mode > 9) block->y_mode = 0; /* Fallback to DC */
        block->uv_mode = block->y_mode;

        block->ref_frame[0] = 0;
        block->ref_frame[1] = 0;
        block->mv[0][0] = 0; block->mv[0][1] = 0;
        block->mv[1][0] = 0; block->mv[1][1] = 0;
    } else {
        block->ref_frame[0] = vpx_read_bit(r) + 1;
        block->ref_frame[1] = 0;

        /* Reconstruct Motion Vector (1/8-pel deltas) */
        int mv_sign_x = vpx_read_bit(r);
        int mv_val_x = vpx_read_literal(r, 6);
        block->mv[0][0] = mv_sign_x ? -mv_val_x : mv_val_x;

        int mv_sign_y = vpx_read_bit(r);
        int mv_val_y = vpx_read_literal(r, 6);
        block->mv[0][1] = mv_sign_y ? -mv_val_y : mv_val_y;

        /* Write motion vector to grid for GPU motion compensation */
        int start_gx = block->x / 4;
        int start_gy = block->y / 4;
        int end_gx = (block->x + block->width) / 4;
        int end_gy = (block->y + block->height) / 4;

        for (int gy = start_gy; gy < end_gy; gy++) {
            for (int gx = start_gx; gx < end_gx; gx++) {
                if (gx < (int)pf->mv_grid_width && gy < (int)pf->mv_grid_height) {
                    uint32_t idx = gy * pf->mv_grid_width + gx;
                    pf->mv_grid[idx].x = block->mv[0][0];
                    pf->mv_grid[idx].y = block->mv[0][1];
                }
            }
        }
    }

    /* Select transform size */
    int max_tx = 0;
    if (block->width >= 32 && block->height >= 32) max_tx = 3;
    else if (block->width >= 16 && block->height >= 16) max_tx = 2;
    else if (block->width >= 8 && block->height >= 8) max_tx = 1;

    if (max_tx > 0) {
        block->tx_size = vpx_read_literal(r, 2);
        if (block->tx_size > max_tx) block->tx_size = max_tx;
    } else {
        block->tx_size = 0;
    }

    block->coeff_offset = pf->num_coeffs;
    block->eob = 0;

    /* Decode transform block coefficients */
    if (!block->skip) {
        int tx_pixels = 1 << (block->tx_size + 2);
        int num_tx_x = block->width / tx_pixels;
        int num_tx_y = block->height / tx_pixels;
        int tx_area = tx_pixels * tx_pixels;

        uint32_t total_coeffs = num_tx_x * num_tx_y * tx_area;
        if (vp9_parsed_frame_ensure_coeffs(pf, total_coeffs)) {
            for (int ty = 0; ty < num_tx_y; ty++) {
                for (int tx = 0; tx < num_tx_x; tx++) {
                    int16_t *tx_coeff_dest = &pf->coeffs[pf->num_coeffs];
                    int eob = vp9_decode_tx_block(r, block->tx_size, 0, block->is_intra ? 0 : 1,
                                                 probs, 0, tx_coeff_dest);
                    if (eob > block->eob) block->eob = eob;
                    pf->num_coeffs += tx_area;
                }
            }
        }
    }

    pf->num_blocks++;
}
