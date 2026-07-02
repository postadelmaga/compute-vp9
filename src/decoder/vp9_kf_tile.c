/**
 * compute-vp9 — spec-conformant keyframe tile decode
 *
 * Bit-exact entropy decode of intra-only tiles: partition tree with spec
 * contexts, keyframe intra modes (kf trees keyed on neighbor modes),
 * tx_size/skip with neighbor contexts, and full coefficient token decode
 * (band maps, scan-neighbor contexts, pareto tail model).
 *
 * Mirrors libvpx vp9_decodeframe.c / vp9_decodemv.c / vp9_detokenize.c
 * for the intra-frame paths.
 */
#include "vp9_tile.h"
#include "vp9_entropy.h"
#include "vp9_default_probs.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

enum { PARTITION_NONE_ = 0, PARTITION_HORZ_ = 1, PARTITION_VERT_ = 2, PARTITION_SPLIT_ = 3 };
enum { BLOCK_8X8_ = 3, BLOCK_64X64_ = 12, BLOCK_INVALID_ = 255 };

typedef struct {
    const vp9_frame_header_t  *hdr;
    const vp9_entropy_probs_t *fc;
    vp9_parsed_frame_t        *pf;
    vpx_reader                *r;

    int mi_rows, mi_cols;
    int tile_col_start;
    int lossless;

    /* Partition contexts (1 byte per mi unit) */
    uint8_t *above_seg;      /* [mi_cols] */
    uint8_t  left_seg[8];

    /* Token entropy contexts, 4x4 granularity: y full res, uv subsampled */
    uint8_t *above_tok[3];   /* [mi_cols*2] (uv use half) */
    uint8_t  left_tok[3][16];

    /* Per-mi mode info (frame-sized, tile touches its own columns) */
    uint8_t *mi_mode;        /* block y mode */
    uint8_t *mi_bmi;         /* [4] sub-modes per mi */
    uint8_t *mi_skip;
    uint8_t *mi_txsz;
    uint8_t *mi_sub8;        /* bsize < 8x8 flag */

    uint8_t token_cache[1024];
} kf_tile_ctx_t;

/* ── Neighbor mode helpers (libvpx vp9_blockd.c) ────────────────────────── */

static int mi_get_y_mode(const kf_tile_ctx_t *t, int mi, int b)
{
    return t->mi_sub8[mi] ? t->mi_bmi[mi * 4 + b] : t->mi_mode[mi];
}

static int above_block_mode(const kf_tile_ctx_t *t, int mi_row, int mi_col,
                            const uint8_t *cur_bmi, int b)
{
    if (b == 0 || b == 1) {
        if (mi_row == 0) return DC_PRED;
        return mi_get_y_mode(t, (mi_row - 1) * t->mi_cols + mi_col, b + 2);
    }
    return cur_bmi[b - 2];
}

static int left_block_mode(const kf_tile_ctx_t *t, int mi_row, int mi_col,
                           const uint8_t *cur_bmi, int b)
{
    if (b == 0 || b == 2) {
        if (mi_col <= t->tile_col_start) return DC_PRED;
        return mi_get_y_mode(t, mi_row * t->mi_cols + (mi_col - 1), b + 1);
    }
    return cur_bmi[b - 1];
}

static int read_intra_mode_kf(kf_tile_ctx_t *t, int mi_row, int mi_col,
                              const uint8_t *cur_bmi, int b)
{
    int a = above_block_mode(t, mi_row, mi_col, cur_bmi, b);
    int l = left_block_mode(t, mi_row, mi_col, cur_bmi, b);
    return vp9_read_intra_mode(t->r, vp9_kf_y_mode_probs[a][l]);
}

/* ── skip / tx_size (libvpx vp9_decodemv.c) ─────────────────────────────── */

static int read_skip(kf_tile_ctx_t *t, int mi_row, int mi_col)
{
    int ctx = 0;
    if (mi_row > 0)
        ctx += t->mi_skip[(mi_row - 1) * t->mi_cols + mi_col];
    if (mi_col > t->tile_col_start)
        ctx += t->mi_skip[mi_row * t->mi_cols + (mi_col - 1)];
    return vpx_read(t->r, t->fc->skip_probs[ctx]);
}

static int read_tx_size(kf_tile_ctx_t *t, int mi_row, int mi_col, int bsize, int allow_select)
{
    static const int biggest_tx[5] = { 0, 1, 2, 3, 3 };
    int max_tx = vp9_max_txsize[bsize];
    int tx_mode = t->fc->tx_mode;

    if (!(allow_select && tx_mode == VP9_TX_MODE_SELECT && bsize >= BLOCK_8X8_)) {
        int m = biggest_tx[tx_mode];
        return max_tx < m ? max_tx : m;
    }

    /* get_tx_size_context */
    int has_above = mi_row > 0;
    int has_left = mi_col > t->tile_col_start;
    int above = max_tx, left = max_tx;
    if (has_above) {
        int mi = (mi_row - 1) * t->mi_cols + mi_col;
        if (!t->mi_skip[mi]) above = t->mi_txsz[mi];
    }
    if (has_left) {
        int mi = mi_row * t->mi_cols + (mi_col - 1);
        if (!t->mi_skip[mi]) left = t->mi_txsz[mi];
    }
    if (!has_left) left = above;
    if (!has_above) above = left;
    int ctx = (above + left) > max_tx;

    const uint8_t *p =
        max_tx == 1 ? t->fc->tx8_probs[ctx] :
        max_tx == 2 ? t->fc->tx16_probs[ctx] : t->fc->tx32_probs[ctx];

    int tx_size = vpx_read(t->r, p[0]);
    if (tx_size != 0 && max_tx >= 2) {
        tx_size += vpx_read(t->r, p[1]);
        if (tx_size != 1 && max_tx >= 3) tx_size += vpx_read(t->r, p[2]);
    }
    return tx_size;
}

/* ── Coefficient token decode (libvpx vp9_detokenize.c) ─────────────────── */

static const int16_t *scan_for(int tx_size, int tx_type, const int16_t **nb)
{
    /* tx_type: 0=DCT(default) 1=ADST_DCT(row) 2=DCT_ADST(col) 3=ADST(default) */
    switch (tx_size) {
    case 0:
        if (tx_type == 1) { *nb = vp9_row_scan_4x4_nb; return vp9_row_scan_4x4_tbl; }
        if (tx_type == 2) { *nb = vp9_col_scan_4x4_nb; return vp9_col_scan_4x4_tbl; }
        *nb = vp9_default_scan_4x4_nb; return vp9_default_scan_4x4_tbl;
    case 1:
        if (tx_type == 1) { *nb = vp9_row_scan_8x8_nb; return vp9_row_scan_8x8_tbl; }
        if (tx_type == 2) { *nb = vp9_col_scan_8x8_nb; return vp9_col_scan_8x8_tbl; }
        *nb = vp9_default_scan_8x8_nb; return vp9_default_scan_8x8_tbl;
    case 2:
        if (tx_type == 1) { *nb = vp9_row_scan_16x16_nb; return vp9_row_scan_16x16_tbl; }
        if (tx_type == 2) { *nb = vp9_col_scan_16x16_nb; return vp9_col_scan_16x16_tbl; }
        *nb = vp9_default_scan_16x16_nb; return vp9_default_scan_16x16_tbl;
    default:
        *nb = vp9_default_scan_32x32_nb; return vp9_default_scan_32x32_tbl;
    }
}

static int read_extra_bits(vpx_reader *r, const uint8_t *probs, int n)
{
    int val = 0;
    for (int i = 0; i < n; i++) val = (val << 1) | vpx_read(r, probs[i]);
    return val;
}

/* Decode one transform block; returns eob. Coefficients (raw magnitudes with
 * sign, pre-dequant) are written in raster order into out (tx_area entries). */
static int decode_coefs(kf_tile_ctx_t *t, int plane_type, int tx_size, int tx_type,
                        int ctx_in, int16_t *out)
{
    vpx_reader *r = t->r;
    const int16_t *nb;
    const int16_t *scan = scan_for(tx_size, tx_type, &nb);
    const int max_eob = 16 << (tx_size << 1);
    const uint8_t *band_translate = tx_size == 0
        ? vp9_coefband_trans_4x4_tbl : vp9_coefband_trans_8x8plus_tbl;
    const uint8_t (*coef_probs)[COEFF_CONTEXTS][UNCONSTRAINED_NODES] =
        t->fc->coef_probs[tx_size][plane_type][0];  /* ref_type 0 = intra */
    uint8_t *token_cache = t->token_cache;

    memset(out, 0, max_eob * sizeof(int16_t));

    int c = 0;
    int ctx = ctx_in;

    while (c < max_eob) {
        int band = band_translate[c];
        const uint8_t *prob = coef_probs[band][ctx];
        int val;

        if (!vpx_read(r, prob[0])) break;  /* EOB */

        while (!vpx_read(r, prob[1])) {    /* ZERO run */
            token_cache[scan[c]] = 0;
            c++;
            if (c >= max_eob) return c;
            ctx = (1 + token_cache[nb[2 * c]] + token_cache[nb[2 * c + 1]]) >> 1;
            band = band_translate[c];
            prob = coef_probs[band][ctx];
        }

        if (!vpx_read(r, prob[2])) {
            token_cache[scan[c]] = 1;      /* energy class of ONE */
            val = 1;
        } else {
            const uint8_t *p = vp9_pareto8_full[prob[2] - 1];
            if (!vpx_read(r, p[0])) {
                if (!vpx_read(r, p[1])) {
                    token_cache[scan[c]] = 2;
                    val = 2;
                } else {
                    token_cache[scan[c]] = 3;
                    val = vpx_read(r, p[2]) ? 4 : 3;
                }
            } else {
                if (!vpx_read(r, p[3])) {
                    if (!vpx_read(r, p[4])) {
                        token_cache[scan[c]] = 4;
                        val = 5 + read_extra_bits(r, vp9_cat1_prob_tbl, 1);
                    } else {
                        token_cache[scan[c]] = 4;
                        val = 7 + read_extra_bits(r, vp9_cat2_prob_tbl, 2);
                    }
                } else if (!vpx_read(r, p[5])) {
                    if (!vpx_read(r, p[6])) {
                        token_cache[scan[c]] = 5;
                        val = 11 + read_extra_bits(r, vp9_cat3_prob_tbl, 3);
                    } else {
                        token_cache[scan[c]] = 5;
                        val = 19 + read_extra_bits(r, vp9_cat4_prob_tbl, 4);
                    }
                } else if (!vpx_read(r, p[7])) {
                    token_cache[scan[c]] = 5;
                    val = 35 + read_extra_bits(r, vp9_cat5_prob_tbl, 5);
                } else {
                    token_cache[scan[c]] = 5;
                    val = 67 + read_extra_bits(r, vp9_cat6_prob_tbl, 14);
                }
            }
        }

        out[scan[c]] = vpx_read_bit(r) ? (int16_t)-val : (int16_t)val;
        c++;
        if (c < max_eob) {
            ctx = (1 + token_cache[nb[2 * c]] + token_cache[nb[2 * c + 1]]) >> 1;
        }
    }

    return c;
}

/* Entropy context of a tx block from the above/left arrays */
static int entropy_ctx(const uint8_t *a, const uint8_t *l, int tx_size)
{
    int n = 1 << tx_size;
    int av = 0, lv = 0;
    for (int i = 0; i < n; i++) { av |= a[i]; lv |= l[i]; }
    return (av != 0) + (lv != 0);
}

/* ── Per-block residual (tokens) ────────────────────────────────────────── */

static int decode_block_tokens(kf_tile_ctx_t *t, int mi_row, int mi_col,
                               int bsize, int y_tx, int skip,
                               vp9_macroblock_info_t *block)
{
    vp9_parsed_frame_t *pf = t->pf;
    int max_eob_total = 0;

    /* Residuals cover the full MI area: sub-8x8 partitions still code a
     * complete 8x8 of luma (4 tx blocks) and one 4x4 per chroma plane */
    int eff_bsize = bsize < BLOCK_8X8_ ? BLOCK_8X8_ : bsize;

    for (int plane = 0; plane < 3; plane++) {
        int ss = plane > 0;  /* 4:2:0 */
        int plane_type = plane > 0;
        int pb = ss ? vp9_ss_size_lookup[eff_bsize][1][1] : eff_bsize;
        int tx = ss ? (bsize < BLOCK_8X8_ ? 0 : (y_tx > vp9_max_txsize[pb] ? vp9_max_txsize[pb] : y_tx))
                    : y_tx;
        int n4w = vp9_num_4x4_w[pb], n4h = vp9_num_4x4_h[pb];
        int txs4 = 1 << tx;

        /* Frame-edge clipping in plane 4x4 units */
        int origin4x = (mi_col * 2) >> ss, origin4y = (mi_row * 2) >> ss;
        int plane_w4 = ((t->mi_cols * 2) >> ss);
        int plane_h4 = ((t->mi_rows * 2) >> ss);
        int max_w = n4w, max_h = n4h;
        if (origin4x + max_w > plane_w4) max_w = plane_w4 - origin4x;
        if (origin4y + max_h > plane_h4) max_h = plane_h4 - origin4y;

        uint8_t *A = t->above_tok[plane] + origin4x;
        uint8_t *L = t->left_tok[plane] + (((mi_row & 7) * 2) >> ss);

        if (skip) {
            /* Context reset spans the unclipped block dims (spec) */
            memset(A, 0, n4w);
            memset(L, 0, n4h);
            continue;
        }

        int tx_area = 16 << (tx << 1);
        for (int y4 = 0; y4 < max_h; y4 += txs4) {
            for (int x4 = 0; x4 < max_w; x4 += txs4) {
                int tx_type = 0;
                if (plane == 0 && !t->lossless && tx < 3) {
                    /* intra: tx type from the (sub-)block prediction mode */
                    int mode;
                    if (bsize < BLOCK_8X8_) {
                        int b = (y4 > 0 ? 2 : 0) + (x4 > 0 ? 1 : 0);
                        mode = t->mi_bmi[(mi_row * t->mi_cols + mi_col) * 4 + b];
                    } else {
                        mode = block->y_mode;
                    }
                    tx_type = vp9_intra_mode_to_tx_type[mode];
                }

                int ctx0 = entropy_ctx(A + x4, L + y4, tx);

                if (!vp9_parsed_frame_ensure_coeffs(pf, tx_area)) return -1;
                int16_t *dst = &pf->coeffs[pf->num_coeffs];
                int eob = decode_coefs(t, plane_type, tx, tx_type, ctx0, dst);
                pf->num_coeffs += tx_area;
                if (plane == 0 && eob > max_eob_total) max_eob_total = eob;

                uint8_t nz = eob > 0;
                for (int i = 0; i < txs4 && x4 + i < max_w; i++) A[x4 + i] = nz;
                for (int i = 0; i < txs4 && y4 + i < max_h; i++) L[y4 + i] = nz;
            }
        }
    }

    block->eob = (uint16_t)max_eob_total;
    return 0;
}

/* ── Block decode (mode info + tokens) ──────────────────────────────────── */

static int decode_block_kf(kf_tile_ctx_t *t, int mi_row, int mi_col, int bsize)
{
    vp9_parsed_frame_t *pf = t->pf;
    vpx_reader *r = t->r;

    if (!vp9_parsed_frame_ensure_blocks(pf, 1)) return -1;
    vp9_macroblock_info_t *block = &pf->blocks[pf->num_blocks];
    memset(block, 0, sizeof(*block));

    block->x = (uint16_t)(mi_col * 8);
    block->y = (uint16_t)(mi_row * 8);
    block->width = (uint8_t)(vp9_num_4x4_w[bsize] * 4);
    block->height = (uint8_t)(vp9_num_4x4_h[bsize] * 4);
    block->is_intra = 1;

    /* read_intra_frame_mode_info order: skip, tx_size, y mode(s), uv mode.
     * Note: intra frames always read the selected tx_size (allow_select=1),
     * unlike inter frames where skip suppresses it. */
    int skip = read_skip(t, mi_row, mi_col);
    int y_tx = read_tx_size(t, mi_row, mi_col, bsize, 1);

    uint8_t bmi[4] = { DC_PRED, DC_PRED, DC_PRED, DC_PRED };
    int mode;
    switch (bsize) {
    case 0:  /* 4x4 */
        for (int i = 0; i < 4; i++)
            bmi[i] = (uint8_t)read_intra_mode_kf(t, mi_row, mi_col, bmi, i);
        mode = bmi[3];
        break;
    case 1:  /* 4x8 */
        bmi[0] = bmi[2] = (uint8_t)read_intra_mode_kf(t, mi_row, mi_col, bmi, 0);
        bmi[1] = bmi[3] = (uint8_t)read_intra_mode_kf(t, mi_row, mi_col, bmi, 1);
        mode = bmi[3];
        break;
    case 2:  /* 8x4 */
        bmi[0] = bmi[1] = (uint8_t)read_intra_mode_kf(t, mi_row, mi_col, bmi, 0);
        bmi[2] = bmi[3] = (uint8_t)read_intra_mode_kf(t, mi_row, mi_col, bmi, 2);
        mode = bmi[3];
        break;
    default:
        mode = read_intra_mode_kf(t, mi_row, mi_col, bmi, 0);
        bmi[0] = bmi[1] = bmi[2] = bmi[3] = (uint8_t)mode;
        break;
    }
    int uv_mode = vp9_read_intra_mode(r, vp9_kf_uv_mode_probs[mode]);

    block->skip = (uint8_t)skip;
    block->tx_size = (uint8_t)y_tx;
    block->y_mode = (uint8_t)mode;
    block->uv_mode = (uint8_t)uv_mode;
    block->coeff_offset = pf->num_coeffs;

    /* Store mode info over the block's mi rectangle for neighbor contexts */
    int bw_mi = vp9_num_8x8_w[bsize], bh_mi = vp9_num_8x8_h[bsize];
    if (bw_mi == 0) bw_mi = 1;
    if (bh_mi == 0) bh_mi = 1;
    int sub8 = bsize < BLOCK_8X8_;
    for (int ry = 0; ry < bh_mi && mi_row + ry < t->mi_rows; ry++) {
        for (int cx = 0; cx < bw_mi && mi_col + cx < t->mi_cols; cx++) {
            int mi = (mi_row + ry) * t->mi_cols + (mi_col + cx);
            t->mi_mode[mi] = (uint8_t)mode;
            t->mi_skip[mi] = (uint8_t)skip;
            t->mi_txsz[mi] = (uint8_t)y_tx;
            t->mi_sub8[mi] = (uint8_t)sub8;
            memcpy(&t->mi_bmi[mi * 4], bmi, 4);

            /* Legacy grids for merge/GPU compatibility */
            uint32_t gidx = (uint32_t)mi;
            pf->mi_width_grid[gidx] = block->width;
            pf->mi_height_grid[gidx] = block->height;
            pf->mi_block_grid[gidx] = pf->num_blocks + 1;
        }
    }

    if (decode_block_tokens(t, mi_row, mi_col, bsize, y_tx, skip, block) != 0)
        return -1;

    pf->num_blocks++;
    return 0;
}

/* ── Partition recursion (libvpx decode_partition) ──────────────────────── */

static void update_partition_ctx(kf_tile_ctx_t *t, int mi_row, int mi_col,
                                 int subsize, int bw_mi)
{
    /* Unclipped spans (arrays are SB-padded; left is SB-aligned by design) */
    memset(t->above_seg + mi_col, vp9_partition_ctx_above[subsize], bw_mi);
    memset(t->left_seg + (mi_row & 7), vp9_partition_ctx_left[subsize], bw_mi);
}

static int read_partition_kf(kf_tile_ctx_t *t, int mi_row, int mi_col,
                             int has_rows, int has_cols, int bsl)
{
    int above = (t->above_seg[mi_col] >> bsl) & 1;
    int left = (t->left_seg[mi_row & 7] >> bsl) & 1;
    int ctx = (left * 2 + above) + bsl * 4;
    const uint8_t *probs = vp9_kf_partition_probs_tbl[ctx];

    if (has_rows && has_cols) {
        if (!vpx_read(t->r, probs[0])) return PARTITION_NONE_;
        if (!vpx_read(t->r, probs[1])) return PARTITION_HORZ_;
        return vpx_read(t->r, probs[2]) ? PARTITION_SPLIT_ : PARTITION_VERT_;
    }
    if (!has_rows && has_cols)
        return vpx_read(t->r, probs[1]) ? PARTITION_SPLIT_ : PARTITION_HORZ_;
    if (has_rows && !has_cols)
        return vpx_read(t->r, probs[2]) ? PARTITION_SPLIT_ : PARTITION_VERT_;
    return PARTITION_SPLIT_;
}

static int decode_partition_kf(kf_tile_ctx_t *t, int mi_row, int mi_col,
                               int bsize, int n4x4_l2)
{
    if (mi_row >= t->mi_rows || mi_col >= t->mi_cols) return 0;

    int n8x8_l2 = n4x4_l2 - 1;
    int num_8x8 = 1 << n8x8_l2;
    int hbs = num_8x8 >> 1;
    int has_rows = (mi_row + hbs) < t->mi_rows;
    int has_cols = (mi_col + hbs) < t->mi_cols;

    int partition = read_partition_kf(t, mi_row, mi_col, has_rows, has_cols, n8x8_l2);
    int subsize = vp9_subsize_lookup[partition][bsize];
    int rc = 0;

    if (!hbs) {
        /* 8x8 block: partition selects the sub-8x8 size directly */
        rc = decode_block_kf(t, mi_row, mi_col, subsize);
    } else {
        switch (partition) {
        case PARTITION_NONE_:
            rc = decode_block_kf(t, mi_row, mi_col, subsize);
            break;
        case PARTITION_HORZ_:
            rc = decode_block_kf(t, mi_row, mi_col, subsize);
            if (!rc && has_rows)
                rc = decode_block_kf(t, mi_row + hbs, mi_col, subsize);
            break;
        case PARTITION_VERT_:
            rc = decode_block_kf(t, mi_row, mi_col, subsize);
            if (!rc && has_cols)
                rc = decode_block_kf(t, mi_row, mi_col + hbs, subsize);
            break;
        default:
            rc |= decode_partition_kf(t, mi_row, mi_col, subsize, n8x8_l2);
            rc |= decode_partition_kf(t, mi_row, mi_col + hbs, subsize, n8x8_l2);
            rc |= decode_partition_kf(t, mi_row + hbs, mi_col, subsize, n8x8_l2);
            rc |= decode_partition_kf(t, mi_row + hbs, mi_col + hbs, subsize, n8x8_l2);
            break;
        }
    }

    if (bsize >= BLOCK_8X8_ &&
        (bsize == BLOCK_8X8_ || partition != PARTITION_SPLIT_)) {
        update_partition_ctx(t, mi_row, mi_col, subsize, num_8x8);
    }
    return rc;
}

/* ── Tile entry point ───────────────────────────────────────────────────── */

int vp9_decode_tile_kf(vpx_reader *r,
                       int mi_row_start, int mi_row_end,
                       int mi_col_start, int mi_col_end,
                       const vp9_entropy_probs_t *probs,
                       vp9_parsed_frame_t *pf)
{
    kf_tile_ctx_t t = {0};
    t.hdr = &pf->hdr;
    t.fc = probs;
    t.pf = pf;
    t.r = r;
    t.mi_rows = (int)pf->mi_grid_height;
    t.mi_cols = (int)pf->mi_grid_width;
    t.tile_col_start = mi_col_start;
    t.lossless = pf->hdr.base_qindex == 0 && pf->hdr.y_dc_delta_q == 0 &&
                 pf->hdr.uv_dc_delta_q == 0 && pf->hdr.uv_ac_delta_q == 0;

    size_t mi_count = (size_t)t.mi_rows * t.mi_cols;
    /* Above arrays padded to a superblock multiple: context updates use
     * unclipped block dims at the frame edge (spec behavior) */
    size_t sb_cols = ((size_t)t.mi_cols + 7) & ~(size_t)7;
    t.above_seg = calloc(sb_cols, 1);
    t.above_tok[0] = calloc(sb_cols * 2, 1);
    t.above_tok[1] = calloc(sb_cols, 1);
    t.above_tok[2] = calloc(sb_cols, 1);
    t.mi_mode = calloc(mi_count, 1);
    t.mi_bmi = calloc(mi_count * 4, 1);
    t.mi_skip = calloc(mi_count, 1);
    t.mi_txsz = calloc(mi_count, 1);
    t.mi_sub8 = calloc(mi_count, 1);

    int rc = -1;
    if (t.above_seg && t.above_tok[0] && t.above_tok[1] && t.above_tok[2] &&
        t.mi_mode && t.mi_bmi && t.mi_skip && t.mi_txsz && t.mi_sub8) {
        rc = 0;
        for (int mi_row = mi_row_start; mi_row < mi_row_end && rc == 0; mi_row += 8) {
            memset(t.left_seg, 0, sizeof(t.left_seg));
            memset(t.left_tok, 0, sizeof(t.left_tok));
            for (int mi_col = mi_col_start; mi_col < mi_col_end && rc == 0; mi_col += 8) {
                rc = decode_partition_kf(&t, mi_row, mi_col, BLOCK_64X64_, 4);
            }
        }
        if (rc == 0 && vpx_reader_has_error(r)) rc = -1;
    }

    free(t.above_seg);
    free(t.above_tok[0]);
    free(t.above_tok[1]);
    free(t.above_tok[2]);
    free(t.mi_mode);
    free(t.mi_bmi);
    free(t.mi_skip);
    free(t.mi_txsz);
    free(t.mi_sub8);
    return rc;
}
