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
        /* Standard-conformant context-dependent partition decision */
        int size_group = 3;
        if (size_mi == 8)      size_group = 0;
        else if (size_mi == 4) size_group = 1;
        else if (size_mi == 2) size_group = 2;

        bool above_has_smaller_block = false;
        if (mi_row > 0) {
            uint32_t above_idx = (mi_row - 1) * pf->mi_grid_width + mi_col;
            uint32_t above_w = pf->mi_width_grid[above_idx];
            above_has_smaller_block = (above_w > 0 && above_w < size_mi * 8);
        }
        bool left_has_smaller_block = false;
        if (mi_col > 0) {
            uint32_t left_idx = mi_row * pf->mi_grid_width + (mi_col - 1);
            uint32_t left_h = pf->mi_height_grid[left_idx];
            left_has_smaller_block = (left_h > 0 && left_h < size_mi * 8);
        }
        int ctx = (left_has_smaller_block ? 2 : 0) + (above_has_smaller_block ? 1 : 0);

        int p_offset = size_group * 4 + ctx;
        const uint8_t *p = probs->partition_probs[p_offset];

        if (vpx_read(r, p[0])) {
            partition = PARTITION_SPLIT;
        } else if (vpx_read(r, p[1])) {
            partition = vpx_read(r, p[2]) ? PARTITION_VERT : PARTITION_HORZ;
        } else {
            partition = PARTITION_NONE;
        }
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

static const vp9_macroblock_info_t *find_block_at(const vp9_parsed_frame_t *pf, int x, int y)
{
    if (x < 0 || y < 0) return NULL;
    for (uint32_t i = 0; i < pf->num_blocks; i++) {
        const vp9_macroblock_info_t *b = &pf->blocks[i];
        if (x >= b->x && x < b->x + b->width && y >= b->y && y < b->y + b->height) {
            return b;
        }
    }
    return NULL;
}

static int get_y_mode_context(const vp9_macroblock_info_t *above, const vp9_macroblock_info_t *left)
{
    uint8_t above_mode = above ? (above->is_intra ? above->y_mode : DC_PRED) : DC_PRED;
    uint8_t left_mode = left ? (left->is_intra ? left->y_mode : DC_PRED) : DC_PRED;
    
    static const uint8_t mode_to_ctx[10] = {
        0, 1, 2, 3, 4, 4, 4, 4, 4, 4
    };
    int above_ctx = mode_to_ctx[above_mode];
    int left_ctx = mode_to_ctx[left_mode];
    
    return (above_ctx + left_ctx) >> 1;
}

static int get_inter_mode_context(const vp9_macroblock_info_t *above, const vp9_macroblock_info_t *left)
{
    int above_intra = above ? above->is_intra : 1;
    int left_intra = left ? left->is_intra : 1;
    
    if (above_intra && left_intra) return 0;
    if (above_intra || left_intra) return 2;
    return 4;
}

static void vp9_find_mv_predictors(const vp9_parsed_frame_t *pf, int x, int y,
                                   cvp9_mv_t *nearest, cvp9_mv_t *near)
{
    nearest->x = 0; nearest->y = 0;
    near->x = 0; near->y = 0;

    const vp9_macroblock_info_t *above = find_block_at(pf, x, y - 1);
    const vp9_macroblock_info_t *left = find_block_at(pf, x - 1, y);

    int count = 0;
    if (above && !above->is_intra) {
        nearest->x = above->mv[0][0];
        nearest->y = above->mv[0][1];
        count++;
    }
    if (left && !left->is_intra) {
        if (count == 0) {
            nearest->x = left->mv[0][0];
            nearest->y = left->mv[0][1];
        } else {
            near->x = left->mv[0][0];
            near->y = left->mv[0][1];
        }
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

    const vp9_macroblock_info_t *above_blk = find_block_at(pf, block->x, block->y - 1);
    const vp9_macroblock_info_t *left_blk = find_block_at(pf, block->x - 1, block->y);

    if (block->is_intra) {
        int ctx_idx = get_y_mode_context(above_blk, left_blk);
        block->y_mode = vp9_read_intra_mode(r, probs->y_mode_probs[ctx_idx]);
        block->uv_mode = vp9_read_intra_mode(r, probs->uv_mode_probs[block->y_mode]);

        block->ref_frame[0] = 0;
        block->ref_frame[1] = 0;
        block->mv[0][0] = 0; block->mv[0][1] = 0;
        block->mv[1][0] = 0; block->mv[1][1] = 0;
    } else {
        block->ref_frame[0] = vpx_read_bit(r) + 1;
        block->ref_frame[1] = 0;

        /* Find motion vector predictors */
        cvp9_mv_t nearest_mv, near_mv;
        vp9_find_mv_predictors(pf, block->x, block->y, &nearest_mv, &near_mv);

        /* Read inter prediction mode */
        int ctx_idx = get_inter_mode_context(above_blk, left_blk);
        int mode = vp9_read_inter_mode(r, probs->inter_mode_probs[ctx_idx]);
        block->y_mode = mode; // Store inter mode in y_mode for consistency

        if (mode == ZEROMV) {
            block->mv[0][0] = 0;
            block->mv[0][1] = 0;
        } else if (mode == NEARESTMV) {
            block->mv[0][0] = nearest_mv.x;
            block->mv[0][1] = nearest_mv.y;
        } else if (mode == NEARMV) {
            block->mv[0][0] = near_mv.x;
            block->mv[0][1] = near_mv.y;
        } else { /* NEWMV: parse delta from stream and add to nearest */
            int mv_sign_x = vpx_read_bit(r);
            int mv_val_x = vpx_read_literal(r, 6);
            int delta_x = mv_sign_x ? -mv_val_x : mv_val_x;

            int mv_sign_y = vpx_read_bit(r);
            int mv_val_y = vpx_read_literal(r, 6);
            int delta_y = mv_sign_y ? -mv_val_y : mv_val_y;

            block->mv[0][0] = nearest_mv.x + delta_x;
            block->mv[0][1] = nearest_mv.y + delta_y;
        }

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

    /* Save block dimensions to ModeInfo grid for neighbor context calculation */
    int end_mi_row = mi_row + height_mi;
    int end_mi_col = mi_col + width_mi;
    for (int r = mi_row; r < end_mi_row; r++) {
        for (int c = mi_col; c < end_mi_col; c++) {
            if (r < (int)pf->mi_grid_height && c < (int)pf->mi_grid_width) {
                uint32_t idx = r * pf->mi_grid_width + c;
                pf->mi_width_grid[idx] = block->width;
                pf->mi_height_grid[idx] = block->height;
            }
        }
    }

    pf->num_blocks++;
}
