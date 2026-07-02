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
    uint8_t *mi_mode;        /* block y mode (intra 0-9, inter 10-13) */
    uint8_t *mi_bmi;         /* [4] sub-modes per mi */
    uint8_t *mi_skip;
    uint8_t *mi_txsz;
    uint8_t *mi_sub8;        /* bsize < 8x8 flag */

    /* Inter-frame per-mi info */
    int8_t  *mi_ref0;        /* 0=INTRA, 1=LAST 2=GOLDEN 3=ALTREF */
    int8_t  *mi_ref1;        /* second ref or -1 */
    int16_t *mi_mv;          /* [mi][2 refs][2] row,col (1/8 pel) */
    int16_t *mi_bmi_mv;      /* [mi][4 sub][2 refs][2] */
    uint8_t *mi_filter;

    /* Frame-level inter state */
    int      is_kf;          /* intra-only frame */
    int      tile_col_end;
    int      allow_hp;
    int8_t   comp_fixed_ref;
    int8_t   comp_var_ref[2];
    int      sign_bias[4];   /* indexed by ref frame 0..3 */

    /* Symbol counts for backward adaptation (nullable) */
    vp9_counts_t *counts;

    /* Previous frame MVs (nullable): per-mi ref pair + mv pair */
    const int8_t  *prev_ref0;
    const int8_t  *prev_ref1;
    const int16_t *prev_mv;   /* [mi][2 refs][2] row,col */
    int            use_prev_mvs;

    /* Per-mi export for the next frame's prev-MV scan (nullable) */
    int8_t  *out_ref0;
    int8_t  *out_ref1;
    int16_t *out_mv;

    uint8_t token_cache[1024];
} kf_tile_ctx_t;

enum { INTRA_FRAME_ = 0, LAST_FRAME_ = 1, GOLDEN_FRAME_ = 2, ALTREF_FRAME_ = 3, NO_REF_ = -1 };
#define MV_BORDER_ (16 << 3)
#define SUBPEL_MARGIN_ ((32 - 4) << 3)  /* enc border 32px minus interp extend */

/* Generic vpx tree read */
static int read_tree(vpx_reader *r, const int8_t *tree, const uint8_t *probs)
{
    int i = 0;
    while ((i = tree[i + vpx_read(r, probs[i >> 1])]) > 0)
        ;
    return -i;
}

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
    int v = vpx_read(t->r, t->fc->skip_probs[ctx]);
    if (t->counts) t->counts->skip[ctx][v]++;
    return v;
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
    if (t->counts) {
        if (max_tx == 1) t->counts->tx8[ctx][tx_size]++;
        else if (max_tx == 2) t->counts->tx16[ctx][tx_size]++;
        else t->counts->tx32[ctx][tx_size]++;
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
                        int ref_type, int ctx_in, int16_t *out)
{
    vpx_reader *r = t->r;
    const int16_t *nb;
    const int16_t *scan = scan_for(tx_size, tx_type, &nb);
    const int max_eob = 16 << (tx_size << 1);
    const uint8_t *band_translate = tx_size == 0
        ? vp9_coefband_trans_4x4_tbl : vp9_coefband_trans_8x8plus_tbl;
    const uint8_t (*coef_probs)[COEFF_CONTEXTS][UNCONSTRAINED_NODES] =
        t->fc->coef_probs[tx_size][plane_type][ref_type];
    uint8_t *token_cache = t->token_cache;

    memset(out, 0, max_eob * sizeof(int16_t));

    int c = 0;
    int ctx = ctx_in;

    vp9_counts_t *cnt = t->counts;
#define INC_COEF(tok) do { if (cnt) cnt->coef[tx_size][plane_type][ref_type][band][ctx][tok]++; } while (0)

    while (c < max_eob) {
        int band = band_translate[c];
        const uint8_t *prob = coef_probs[band][ctx];
        int val;

        if (cnt) cnt->eob_branch[tx_size][plane_type][ref_type][band][ctx]++;
        if (!vpx_read(r, prob[0])) { INC_COEF(3); break; }  /* EOB */

        while (!vpx_read(r, prob[1])) {    /* ZERO run */
            INC_COEF(0);
            token_cache[scan[c]] = 0;
            c++;
            if (c >= max_eob) return c;
            ctx = (1 + token_cache[nb[2 * c]] + token_cache[nb[2 * c + 1]]) >> 1;
            band = band_translate[c];
            prob = coef_probs[band][ctx];
        }

        if (!vpx_read(r, prob[2])) {
            INC_COEF(1);
            token_cache[scan[c]] = 1;      /* energy class of ONE */
            val = 1;
        } else {
            INC_COEF(2);
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
#undef INC_COEF

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
                               int bsize, int y_tx, int skip, int is_inter,
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
                if (plane == 0 && !t->lossless && tx < 3 && !is_inter) {
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
                int eob = decode_coefs(t, plane_type, tx, tx_type, is_inter, ctx0, dst);
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

/* ── Inter-frame machinery (libvpx vp9_decodemv.c / vp9_mvref_common) ───── */

#define LEFT_TOP_MARGIN_ ((160 - 4) << 3)  /* VP9_ENC_BORDER - INTERP_EXTEND */

typedef struct { int16_t row, col; } mv_t;

static int mi_is_inter(const kf_tile_ctx_t *t, int mi) { return t->mi_ref0[mi] > 0; }

static void get_block_edges(const kf_tile_ctx_t *t, int mi_row, int mi_col,
                            int bw_mi, int bh_mi, int e[4])
{
    e[0] = -((mi_col * 8) * 8);                          /* to_left   */
    e[1] = (t->mi_cols - bw_mi - mi_col) * 8 * 8;        /* to_right  */
    e[2] = -((mi_row * 8) * 8);                          /* to_top    */
    e[3] = (t->mi_rows - bh_mi - mi_row) * 8 * 8;        /* to_bottom */
}

static void clamp_mv_(mv_t *mv, int min_col, int max_col, int min_row, int max_row)
{
    mv->col = (int16_t)(mv->col < min_col ? min_col : mv->col > max_col ? max_col : mv->col);
    mv->row = (int16_t)(mv->row < min_row ? min_row : mv->row > max_row ? max_row : mv->row);
}

static int is_inside_(const kf_tile_ctx_t *t, int mi_row, int mi_col,
                      const vp9_mv_ref_pos_t *p)
{
    return !(mi_row + p->row < 0 ||
             mi_col + p->col < t->tile_col_start ||
             mi_row + p->row >= t->mi_rows ||
             mi_col + p->col >= t->tile_col_end);
}

static mv_t mi_get_mv(const kf_tile_ctx_t *t, int mi, int which)
{
    mv_t mv = { t->mi_mv[mi * 4 + which * 2], t->mi_mv[mi * 4 + which * 2 + 1] };
    return mv;
}

/* get_sub_block_mv: sub-8x8 candidates in the first two scan positions use
 * the neighbor's closest sub-block mv */
static mv_t get_sub_block_mv_(const kf_tile_ctx_t *t, int mi, int which,
                              int search_col, int block_idx)
{
    if (block_idx >= 0 && t->mi_sub8[mi]) {
        int sub = vp9_idx_n_column_to_subblock[block_idx][search_col == 0];
        mv_t mv = { t->mi_bmi_mv[(mi * 4 + sub) * 4 + which * 2],
                    t->mi_bmi_mv[(mi * 4 + sub) * 4 + which * 2 + 1] };
        return mv;
    }
    return mi_get_mv(t, mi, which);
}

static mv_t scale_mv_(const kf_tile_ctx_t *t, int mi, int which, int this_ref)
{
    mv_t mv = mi_get_mv(t, mi, which);
    int cand_ref = which == 0 ? t->mi_ref0[mi] : t->mi_ref1[mi];
    if (t->sign_bias[cand_ref & 3] != t->sign_bias[this_ref]) {
        mv.row = (int16_t)-mv.row;
        mv.col = (int16_t)-mv.col;
    }
    return mv;
}

static int mv_equal(mv_t a, mv_t b) { return a.row == b.row && a.col == b.col; }

static int get_mode_context_(const kf_tile_ctx_t *t, const vp9_mv_ref_pos_t *search,
                             int mi_row, int mi_col)
{
    int counter = 0;
    for (int i = 0; i < 2; i++) {
        if (is_inside_(t, mi_row, mi_col, &search[i])) {
            int mi = (mi_row + search[i].row) * t->mi_cols + (mi_col + search[i].col);
            counter += vp9_mode_2_counter[t->mi_mode[mi]];
        }
    }
    return vp9_counter_to_context[counter];
}

/* dec_find_mv_refs: early-break candidate scan. mode NEARMV keeps 2
 * candidates, everything else only the first. block >= 0 selects the
 * sub-block variant for the two nearest positions. */
static int dec_find_mv_refs_(const kf_tile_ctx_t *t, int mode, int ref_frame,
                             const vp9_mv_ref_pos_t *search, mv_t *list,
                             int mi_row, int mi_col, int block, const int e[4])
{
    int refmv_count = 0;
    int different_ref_found = 0;
    const int early_break = (mode != NEARMV);
    int i = 0;

    list[0] = (mv_t){0, 0};
    list[1] = (mv_t){0, 0};

#define ADD_CAND(m) do { \
        if (refmv_count) { \
            if (!mv_equal((m), list[0])) { \
                list[refmv_count] = (m); \
                refmv_count++; \
                goto Done; \
            } \
        } else { \
            list[refmv_count++] = (m); \
            if (early_break) goto Done; \
        } \
    } while (0)

    if (block >= 0) {
        for (i = 0; i < 2; i++) {
            if (is_inside_(t, mi_row, mi_col, &search[i])) {
                int mi = (mi_row + search[i].row) * t->mi_cols + (mi_col + search[i].col);
                different_ref_found = 1;
                if (t->mi_ref0[mi] == ref_frame)
                    ADD_CAND(get_sub_block_mv_(t, mi, 0, search[i].col, block));
                else if (t->mi_ref1[mi] == ref_frame)
                    ADD_CAND(get_sub_block_mv_(t, mi, 1, search[i].col, block));
            }
        }
    }

    for (; i < 8; i++) {
        if (is_inside_(t, mi_row, mi_col, &search[i])) {
            int mi = (mi_row + search[i].row) * t->mi_cols + (mi_col + search[i].col);
            different_ref_found = 1;
            if (t->mi_ref0[mi] == ref_frame)
                ADD_CAND(mi_get_mv(t, mi, 0));
            else if (t->mi_ref1[mi] == ref_frame)
                ADD_CAND(mi_get_mv(t, mi, 1));
        }
    }

    if (t->use_prev_mvs) {
        int pmi = mi_row * t->mi_cols + mi_col;
        if (t->prev_ref0[pmi] == ref_frame) {
            mv_t m = { t->prev_mv[pmi * 4 + 0], t->prev_mv[pmi * 4 + 1] };
            ADD_CAND(m);
        } else if (t->prev_ref1[pmi] == ref_frame) {
            mv_t m = { t->prev_mv[pmi * 4 + 2], t->prev_mv[pmi * 4 + 3] };
            ADD_CAND(m);
        }
    }

    if (different_ref_found) {
        for (i = 0; i < 8; i++) {
            if (is_inside_(t, mi_row, mi_col, &search[i])) {
                int mi = (mi_row + search[i].row) * t->mi_cols + (mi_col + search[i].col);
                if (t->mi_ref0[mi] > INTRA_FRAME_ && t->mi_ref0[mi] != ref_frame)
                    ADD_CAND(scale_mv_(t, mi, 0, ref_frame));
                if (t->mi_ref1[mi] > INTRA_FRAME_ && t->mi_ref1[mi] != ref_frame &&
                    !mv_equal(mi_get_mv(t, mi, 1), mi_get_mv(t, mi, 0)))
                    ADD_CAND(scale_mv_(t, mi, 1, ref_frame));
            }
        }
    }
    if (t->use_prev_mvs) {
        int pmi = mi_row * t->mi_cols + mi_col;
        if (t->prev_ref0[pmi] != ref_frame && t->prev_ref0[pmi] > INTRA_FRAME_) {
            mv_t m = { t->prev_mv[pmi * 4 + 0], t->prev_mv[pmi * 4 + 1] };
            if (t->sign_bias[t->prev_ref0[pmi] & 3] != t->sign_bias[ref_frame]) {
                m.row = (int16_t)-m.row;
                m.col = (int16_t)-m.col;
            }
            ADD_CAND(m);
        }
        if (t->prev_ref1[pmi] > INTRA_FRAME_ && t->prev_ref1[pmi] != ref_frame &&
            !(t->prev_mv[pmi * 4 + 2] == t->prev_mv[pmi * 4 + 0] &&
              t->prev_mv[pmi * 4 + 3] == t->prev_mv[pmi * 4 + 1])) {
            mv_t m = { t->prev_mv[pmi * 4 + 2], t->prev_mv[pmi * 4 + 3] };
            if (t->sign_bias[t->prev_ref1[pmi] & 3] != t->sign_bias[ref_frame]) {
                m.row = (int16_t)-m.row;
                m.col = (int16_t)-m.col;
            }
            ADD_CAND(m);
        }
    }
#undef ADD_CAND

    refmv_count = (mode == NEARMV) ? 2 : 1;

Done:
    for (i = 0; i < refmv_count; i++) {
        clamp_mv_(&list[i], e[0] - MV_BORDER_, e[1] + MV_BORDER_,
                  e[2] - MV_BORDER_, e[3] + MV_BORDER_);
    }
    return refmv_count;
}

static int use_mv_hp_(const mv_t *ref)
{
    return abs(ref->row) < 64 && abs(ref->col) < 64;
}

static void lower_mv_precision_(mv_t *mv, int allow_hp)
{
    if (!(allow_hp && use_mv_hp_(mv))) {
        if (mv->row & 1) mv->row += (mv->row > 0 ? -1 : 1);
        if (mv->col & 1) mv->col += (mv->col > 0 ? -1 : 1);
    }
}

static int read_mv_component_(kf_tile_ctx_t *t, int comp, int usehp)
{
    vpx_reader *r = t->r;
    const vp9_entropy_probs_t *fc = t->fc;
    int mag, d, fr, hp;
    int sign = vpx_read(r, fc->mv_sign_probs[comp]);
    int mv_class = read_tree(r, vp9_mv_class_tree_tbl, fc->mv_class_probs[comp]);
    int class0 = mv_class == 0;

    if (class0) {
        d = vpx_read(r, fc->mv_class0_probs[comp][0]);
        mag = 0;
    } else {
        int n = mv_class;  /* mv_class + CLASS0_BITS - 1, CLASS0_BITS = 1 */
        d = 0;
        for (int i = 0; i < n; i++)
            d |= vpx_read(r, fc->mv_bits_probs[comp][i]) << i;
        mag = 2 << (mv_class + 2);  /* CLASS0_SIZE << (class + 2) */
    }

    fr = read_tree(r, vp9_mv_fp_tree_tbl,
                   class0 ? fc->mv_class0_fr_probs[comp][d] : fc->mv_fr_probs[comp]);
    hp = usehp ? vpx_read(r, class0 ? fc->mv_class0_hp_probs[comp]
                                    : fc->mv_hp_probs[comp])
               : 1;

    mag += ((d << 3) | (fr << 1) | hp) + 1;
    return sign ? -mag : mag;
}

static int get_mv_class_(int z, int *offset)
{
    int c = 10;
    for (int i = 0; i < 10; i++) {
        if (z < (2 << (i + 3))) { c = i; break; }  /* base(i+1) = 2 << (i+3) */
    }
    if (offset) *offset = z - (c ? 2 << (c + 2) : 0);
    return c;
}

static void inc_mv_component_(vp9_counts_t *cnt, int comp, int v)
{
    int s = v < 0;
    cnt->mv_comp[comp].sign[s]++;
    int z = (s ? -v : v) - 1;
    int o;
    int c = get_mv_class_(z, &o);
    cnt->mv_comp[comp].classes[c]++;
    int d = o >> 3, f = (o >> 1) & 3, e = o & 1;
    if (c == 0) {
        cnt->mv_comp[comp].class0[d]++;
        cnt->mv_comp[comp].class0_fp[d][f]++;
        cnt->mv_comp[comp].class0_hp[e]++;
    } else {
        int b = c + 2;  /* CLASS0_BITS - 1 + c... bits count for counting */
        for (int i = 0; i < b; i++)
            cnt->mv_comp[comp].bits[i][(d >> i) & 1]++;
        cnt->mv_comp[comp].fp[f]++;
        cnt->mv_comp[comp].hp[e]++;
    }
}

static void read_mv_(kf_tile_ctx_t *t, mv_t *mv, const mv_t *ref)
{
    int joint = read_tree(t->r, vp9_mv_joint_tree_tbl, t->fc->mv_joint_probs);
    int use_hp = t->allow_hp && use_mv_hp_(ref);
    mv_t diff = {0, 0};

    if (joint == 2 || joint == 3)  /* vertical nonzero */
        diff.row = (int16_t)read_mv_component_(t, 0, use_hp);
    if (joint == 1 || joint == 3)  /* horizontal nonzero */
        diff.col = (int16_t)read_mv_component_(t, 1, use_hp);

    if (t->counts) {
        int j = (diff.col != 0) | ((diff.row != 0) << 1);
        t->counts->mv_joints[j]++;
        if (diff.row != 0) inc_mv_component_(t->counts, 0, diff.row);
        if (diff.col != 0) inc_mv_component_(t->counts, 1, diff.col);
    }

    mv->row = (int16_t)(ref->row + diff.row);
    mv->col = (int16_t)(ref->col + diff.col);
}

static int is_mv_valid_(const mv_t *mv)
{
    return mv->row > -(1 << 14) && mv->row < (1 << 14) &&
           mv->col > -(1 << 14) && mv->col < (1 << 14);
}

static int assign_mv_(kf_tile_ctx_t *t, int mode, mv_t mv[2], const mv_t ref_mv[2],
                      const mv_t near_nearest[2], int is_compound)
{
    int ret = 1;
    switch (mode) {
    case NEWMV:
        for (int i = 0; i < 1 + is_compound; i++) {
            read_mv_(t, &mv[i], &ref_mv[i]);
            ret = ret && is_mv_valid_(&mv[i]);
        }
        break;
    case NEARMV:
    case NEARESTMV:
        mv[0] = near_nearest[0];
        mv[1] = near_nearest[1];
        break;
    case ZEROMV:
        mv[0] = (mv_t){0, 0};
        mv[1] = (mv_t){0, 0};
        break;
    default:
        return 0;
    }
    return ret;
}

/* ── Neighbor-based contexts (libvpx vp9_pred_common.c) ─────────────────── */

static int get_neighbor_mis(const kf_tile_ctx_t *t, int mi_row, int mi_col,
                            int *above, int *left)
{
    *above = mi_row > 0 ? (mi_row - 1) * t->mi_cols + mi_col : -1;
    *left = mi_col > t->tile_col_start ? mi_row * t->mi_cols + (mi_col - 1) : -1;
    return (*above >= 0) | ((*left >= 0) << 1);
}

static int get_intra_inter_ctx(const kf_tile_ctx_t *t, int a, int l)
{
    if (a >= 0 && l >= 0) {
        int ai = !mi_is_inter(t, a), li = !mi_is_inter(t, l);
        return li && ai ? 3 : li || ai;
    }
    if (a >= 0 || l >= 0)
        return 2 * !mi_is_inter(t, a >= 0 ? a : l);
    return 0;
}

static int get_switchable_interp_ctx(const kf_tile_ctx_t *t, int a, int l)
{
    int left_type = (l >= 0 && mi_is_inter(t, l)) ? t->mi_filter[l] : 3;
    int above_type = (a >= 0 && mi_is_inter(t, a)) ? t->mi_filter[a] : 3;
    if (left_type == above_type) return left_type;
    if (left_type == 3) return above_type;
    if (above_type == 3) return left_type;
    return 3;
}

static int has_second_ref_(const kf_tile_ctx_t *t, int mi) { return t->mi_ref1[mi] > 0; }

static int get_single_ref_p1_ctx(const kf_tile_ctx_t *t, int a, int l)
{
    int ctx;
    if (a >= 0 && l >= 0) {
        int ai = !mi_is_inter(t, a), li = !mi_is_inter(t, l);
        if (ai && li) {
            ctx = 2;
        } else if (ai || li) {
            int e = ai ? l : a;
            if (!has_second_ref_(t, e))
                ctx = 4 * (t->mi_ref0[e] == LAST_FRAME_);
            else
                ctx = 1 + (t->mi_ref0[e] == LAST_FRAME_ || t->mi_ref1[e] == LAST_FRAME_);
        } else {
            int a2 = has_second_ref_(t, a), l2 = has_second_ref_(t, l);
            int a0 = t->mi_ref0[a], l0 = t->mi_ref0[l];
            if (a2 && l2) {
                ctx = 1 + (t->mi_ref0[a] == LAST_FRAME_ || t->mi_ref1[a] == LAST_FRAME_ ||
                           t->mi_ref0[l] == LAST_FRAME_ || t->mi_ref1[l] == LAST_FRAME_);
            } else if (a2 || l2) {
                int rfs = a2 ? l0 : a0;
                int crf1 = a2 ? t->mi_ref0[a] : t->mi_ref0[l];
                int crf2 = a2 ? t->mi_ref1[a] : t->mi_ref1[l];
                if (rfs == LAST_FRAME_)
                    ctx = 3 + (crf1 == LAST_FRAME_ || crf2 == LAST_FRAME_);
                else
                    ctx = crf1 == LAST_FRAME_ || crf2 == LAST_FRAME_;
            } else {
                ctx = 2 * (a0 == LAST_FRAME_) + 2 * (l0 == LAST_FRAME_);
            }
        }
    } else if (a >= 0 || l >= 0) {
        int e = a >= 0 ? a : l;
        if (!mi_is_inter(t, e)) {
            ctx = 2;
        } else if (!has_second_ref_(t, e)) {
            ctx = 4 * (t->mi_ref0[e] == LAST_FRAME_);
        } else {
            ctx = 1 + (t->mi_ref0[e] == LAST_FRAME_ || t->mi_ref1[e] == LAST_FRAME_);
        }
    } else {
        ctx = 2;
    }
    return ctx;
}

static int get_single_ref_p2_ctx(const kf_tile_ctx_t *t, int a, int l)
{
    int ctx;
    if (a >= 0 && l >= 0) {
        int ai = !mi_is_inter(t, a), li = !mi_is_inter(t, l);
        if (ai && li) {
            ctx = 2;
        } else if (ai || li) {
            int e = ai ? l : a;
            if (!has_second_ref_(t, e)) {
                if (t->mi_ref0[e] == LAST_FRAME_)
                    ctx = 3;
                else
                    ctx = 4 * (t->mi_ref0[e] == GOLDEN_FRAME_);
            } else {
                ctx = 1 + 2 * (t->mi_ref0[e] == GOLDEN_FRAME_ ||
                               t->mi_ref1[e] == GOLDEN_FRAME_);
            }
        } else {
            int a2 = has_second_ref_(t, a), l2 = has_second_ref_(t, l);
            int a0 = t->mi_ref0[a], l0 = t->mi_ref0[l];
            if (a2 && l2) {
                if (t->mi_ref0[a] == t->mi_ref0[l] && t->mi_ref1[a] == t->mi_ref1[l])
                    ctx = 3 * (t->mi_ref0[a] == GOLDEN_FRAME_ ||
                               t->mi_ref1[a] == GOLDEN_FRAME_);
                else
                    ctx = 2;
            } else if (a2 || l2) {
                int rfs = a2 ? l0 : a0;
                int crf1 = a2 ? t->mi_ref0[a] : t->mi_ref0[l];
                int crf2 = a2 ? t->mi_ref1[a] : t->mi_ref1[l];
                if (rfs == GOLDEN_FRAME_)
                    ctx = 3 + (crf1 == GOLDEN_FRAME_ || crf2 == GOLDEN_FRAME_);
                else if (rfs == ALTREF_FRAME_)
                    ctx = crf1 == GOLDEN_FRAME_ || crf2 == GOLDEN_FRAME_;
                else
                    ctx = 1 + 2 * (crf1 == GOLDEN_FRAME_ || crf2 == GOLDEN_FRAME_);
            } else {
                if (a0 == LAST_FRAME_ && l0 == LAST_FRAME_) {
                    ctx = 3;
                } else if (a0 == LAST_FRAME_ || l0 == LAST_FRAME_) {
                    int e0 = a0 == LAST_FRAME_ ? l0 : a0;
                    ctx = 4 * (e0 == GOLDEN_FRAME_);
                } else {
                    ctx = 2 * (a0 == GOLDEN_FRAME_) + 2 * (l0 == GOLDEN_FRAME_);
                }
            }
        }
    } else if (a >= 0 || l >= 0) {
        int e = a >= 0 ? a : l;
        if (!mi_is_inter(t, e) ||
            (t->mi_ref0[e] == LAST_FRAME_ && !has_second_ref_(t, e)))
            ctx = 2;
        else if (!has_second_ref_(t, e))
            ctx = 4 * (t->mi_ref0[e] == GOLDEN_FRAME_);
        else
            ctx = 3 * (t->mi_ref0[e] == GOLDEN_FRAME_ ||
                       t->mi_ref1[e] == GOLDEN_FRAME_);
    } else {
        ctx = 2;
    }
    return ctx;
}

/* Read the reference frames for an inter block (single-reference streams:
 * compound contexts are ported minimally — comp mode requires mixed sign
 * biases which selects them; comp_ref ctx uses a simplified port) */
static void read_ref_frames_(kf_tile_ctx_t *t, int mi_row, int mi_col, int8_t ref[2])
{
    vpx_reader *r = t->r;
    int a, l;
    get_neighbor_mis(t, mi_row, mi_col, &a, &l);

    int mode = t->fc->reference_mode;  /* SINGLE/COMPOUND/SELECT */
    if (mode == VP9_REFERENCE_MODE_SELECT) {
        /* vp9_get_reference_mode_context */
        int ctx;
        if (a >= 0 && l >= 0) {
            if (!has_second_ref_(t, a) && !has_second_ref_(t, l))
                ctx = (t->mi_ref0[a] == t->comp_fixed_ref) ^
                      (t->mi_ref0[l] == t->comp_fixed_ref);
            else if (!has_second_ref_(t, a))
                ctx = 2 + (t->mi_ref0[a] == t->comp_fixed_ref || !mi_is_inter(t, a));
            else if (!has_second_ref_(t, l))
                ctx = 2 + (t->mi_ref0[l] == t->comp_fixed_ref || !mi_is_inter(t, l));
            else
                ctx = 4;
        } else if (a >= 0 || l >= 0) {
            int e = a >= 0 ? a : l;
            ctx = has_second_ref_(t, e) ? 3 : (t->mi_ref0[e] == t->comp_fixed_ref);
        } else {
            ctx = 1;
        }
        mode = vpx_read(r, t->fc->comp_inter_probs[ctx]);  /* 0=single 1=compound */
        if (t->counts) t->counts->comp_inter[ctx][mode]++;
    }

    if (mode == VP9_COMPOUND_REFERENCE) {
        int idx = t->sign_bias[t->comp_fixed_ref & 3];
        /* comp_ref ctx: simplified two-edge port (full context tree rarely
         * differs on single-direction streams) */
        int ctx = 2;
        if (a >= 0 && l >= 0) {
            int ai = !mi_is_inter(t, a), li = !mi_is_inter(t, l);
            if (ai && li) ctx = 2;
            else if (ai || li) {
                int e = ai ? l : a;
                int vr = has_second_ref_(t, e) ? (idx ? t->mi_ref0[e] : t->mi_ref1[e])
                                               : t->mi_ref0[e];
                ctx = 1 + 2 * (vr != t->comp_var_ref[1]);
            }
        }
        int bit = vpx_read(r, t->fc->comp_ref_probs[ctx]);
        if (t->counts) t->counts->comp_ref[ctx][bit]++;
        ref[idx] = t->comp_fixed_ref;
        ref[!idx] = t->comp_var_ref[bit];
    } else {
        int ctx0 = get_single_ref_p1_ctx(t, a, l);
        int bit0 = vpx_read(r, t->fc->single_ref_probs[ctx0][0]);
        if (t->counts) t->counts->single_ref[ctx0][0][bit0]++;
        if (bit0) {
            int ctx1 = get_single_ref_p2_ctx(t, a, l);
            int bit1 = vpx_read(r, t->fc->single_ref_probs[ctx1][1]);
            if (t->counts) t->counts->single_ref[ctx1][1][bit1]++;
            ref[0] = bit1 ? ALTREF_FRAME_ : GOLDEN_FRAME_;
        } else {
            ref[0] = LAST_FRAME_;
        }
        ref[1] = NO_REF_;
    }
}

/* ── Block decode (mode info + tokens) ──────────────────────────────────── */

static void append_sub8x8_(kf_tile_ctx_t *t, const vp9_mv_ref_pos_t *search,
                           int b_mode, int block_idx, int ref_i, int8_t ref_frame,
                           mv_t bmi_mv[4][2], int mi_row, int mi_col,
                           const int e[4], mv_t *best)
{
    mv_t list[2];
    int n;

    switch (block_idx) {
    case 0:
        n = dec_find_mv_refs_(t, b_mode, ref_frame, search, list, mi_row, mi_col, 0, e);
        *best = list[n - 1];
        break;
    case 1:
    case 2:
        if (b_mode == NEARESTMV) {
            *best = bmi_mv[0][ref_i];
        } else {
            dec_find_mv_refs_(t, b_mode, ref_frame, search, list, mi_row, mi_col,
                              block_idx, e);
            *best = (mv_t){0, 0};
            for (n = 0; n < 2; n++)
                if (!mv_equal(bmi_mv[0][ref_i], list[n])) { *best = list[n]; break; }
        }
        break;
    default:  /* 3 */
        if (b_mode == NEARESTMV) {
            *best = bmi_mv[2][ref_i];
        } else {
            *best = (mv_t){0, 0};
            if (!mv_equal(bmi_mv[2][ref_i], bmi_mv[1][ref_i])) {
                *best = bmi_mv[1][ref_i];
                break;
            }
            if (!mv_equal(bmi_mv[2][ref_i], bmi_mv[0][ref_i])) {
                *best = bmi_mv[0][ref_i];
                break;
            }
            dec_find_mv_refs_(t, b_mode, ref_frame, search, list, mi_row, mi_col, 3, e);
            for (n = 0; n < 2; n++)
                if (!mv_equal(bmi_mv[2][ref_i], list[n])) { *best = list[n]; break; }
        }
        break;
    }
}

static int decode_block_kf(kf_tile_ctx_t *t, int mi_row, int mi_col, int bsize)
{
    vp9_parsed_frame_t *pf = t->pf;
    vpx_reader *r = t->r;
    const vp9_entropy_probs_t *fc = t->fc;

    if (!vp9_parsed_frame_ensure_blocks(pf, 1)) return -1;
    vp9_macroblock_info_t *block = &pf->blocks[pf->num_blocks];
    memset(block, 0, sizeof(*block));

    int bw_mi = vp9_num_8x8_w[bsize] ? vp9_num_8x8_w[bsize] : 1;
    int bh_mi = vp9_num_8x8_h[bsize] ? vp9_num_8x8_h[bsize] : 1;

    block->x = (uint16_t)(mi_col * 8);
    block->y = (uint16_t)(mi_row * 8);
    block->width = (uint8_t)(vp9_num_4x4_w[bsize] * 4);
    block->height = (uint8_t)(vp9_num_4x4_h[bsize] * 4);

    int skip, y_tx, mode, uv_mode = DC_PRED;
    uint8_t bmi[4] = { DC_PRED, DC_PRED, DC_PRED, DC_PRED };
    int8_t ref[2] = { INTRA_FRAME_, NO_REF_ };
    mv_t mv[2] = { {0, 0}, {0, 0} };
    mv_t bmi_mv[4][2];
    uint8_t filter = 3;  /* intra blocks keep SWITCHABLE like libvpx */
    memset(bmi_mv, 0, sizeof(bmi_mv));

    if (t->is_kf) {
        /* read_intra_frame_mode_info: skip, tx (allow_select=1), modes */
        skip = read_skip(t, mi_row, mi_col);
        y_tx = read_tx_size(t, mi_row, mi_col, bsize, 1);

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
        uv_mode = vp9_read_intra_mode(r, vp9_kf_uv_mode_probs[mode]);
    } else {
        /* read_inter_frame_mode_info: skip, is_inter, tx, block info */
        int a, l;
        get_neighbor_mis(t, mi_row, mi_col, &a, &l);

        skip = read_skip(t, mi_row, mi_col);
        int ii_ctx = get_intra_inter_ctx(t, a, l);
        int is_inter = vpx_read(r, fc->intra_inter_probs[ii_ctx]);
        if (t->counts) t->counts->intra_inter[ii_ctx][is_inter]++;
        y_tx = read_tx_size(t, mi_row, mi_col, bsize, !skip || !is_inter);

        if (!is_inter) {
            /* Intra block in an inter frame (if_y/if_uv probability sets) */
#define READ_Y_MODE(grp) ({ \
        int m_ = vp9_read_intra_mode(r, fc->y_mode_probs[grp]); \
        if (t->counts) t->counts->y_mode[grp][m_]++; \
        m_; })
            switch (bsize) {
            case 0:
                for (int i = 0; i < 4; i++)
                    bmi[i] = (uint8_t)READ_Y_MODE(0);
                mode = bmi[3];
                break;
            case 1:
                bmi[0] = bmi[2] = (uint8_t)READ_Y_MODE(0);
                bmi[1] = bmi[3] = (uint8_t)READ_Y_MODE(0);
                mode = bmi[3];
                break;
            case 2:
                bmi[0] = bmi[1] = (uint8_t)READ_Y_MODE(0);
                bmi[2] = bmi[3] = (uint8_t)READ_Y_MODE(0);
                mode = bmi[3];
                break;
            default:
                mode = READ_Y_MODE(vp9_size_group[bsize]);
                bmi[0] = bmi[1] = bmi[2] = bmi[3] = (uint8_t)mode;
                break;
            }
#undef READ_Y_MODE
            uv_mode = vp9_read_intra_mode(r, fc->uv_mode_probs[mode]);
            if (t->counts) t->counts->uv_mode[mode][uv_mode]++;
        } else {
            read_ref_frames_(t, mi_row, mi_col, ref);
            int is_compound = ref[1] > 0;
            const vp9_mv_ref_pos_t *search = vp9_mv_ref_blocks[bsize];
            int mode_ctx = get_mode_context_(t, search, mi_row, mi_col);
            int e[4];
            get_block_edges(t, mi_row, mi_col, bw_mi, bh_mi, e);
            mv_t best_ref[2] = { {0, 0}, {0, 0} };

            mode = ZEROMV;
            if (bsize >= BLOCK_8X8_) {
                mode = NEARESTMV + read_tree(r, vp9_inter_mode_tree_tbl,
                                             fc->inter_mode_probs[mode_ctx]);
                if (t->counts) t->counts->inter_mode[mode_ctx][mode - NEARESTMV]++;
            }

            if (t->hdr->interp_filter == VP9_FILTER_SWITCHABLE) {
                int f_ctx = get_switchable_interp_ctx(t, a, l);
                filter = (uint8_t)read_tree(r, vp9_switchable_interp_tree_tbl,
                                            fc->switchable_interp_probs[f_ctx]);
                if (t->counts) t->counts->switchable_interp[f_ctx][filter]++;
            } else {
                filter = (uint8_t)t->hdr->interp_filter;
            }

            if (bsize < BLOCK_8X8_) {
                int n4w = vp9_num_4x4_w[bsize], n4h = vp9_num_4x4_h[bsize];
                int got_refs_new = 0;
                int b_mode = ZEROMV;
                mv_t best_sub[2];
                best_sub[0] = (mv_t){0, 0};
                best_sub[1] = (mv_t){-32768, -32768};  /* invalid marker */

                for (int idy = 0; idy < 2; idy += n4h) {
                    for (int idx = 0; idx < 2; idx += n4w) {
                        int j = idy * 2 + idx;
                        b_mode = NEARESTMV + read_tree(r, vp9_inter_mode_tree_tbl,
                                                       fc->inter_mode_probs[mode_ctx]);
                        if (t->counts) t->counts->inter_mode[mode_ctx][b_mode - NEARESTMV]++;

                        if (b_mode == NEARESTMV || b_mode == NEARMV) {
                            for (int ri = 0; ri < 1 + is_compound; ri++)
                                append_sub8x8_(t, search, b_mode, j, ri, ref[ri],
                                               bmi_mv, mi_row, mi_col, e, &best_sub[ri]);
                        } else if (b_mode == NEWMV && !got_refs_new) {
                            for (int ri = 0; ri < 1 + is_compound; ri++) {
                                mv_t tmp[2];
                                dec_find_mv_refs_(t, NEWMV, ref[ri], search, tmp,
                                                  mi_row, mi_col, -1, e);
                                lower_mv_precision_(&tmp[0], t->allow_hp);
                                best_ref[ri] = tmp[0];
                                got_refs_new = 1;
                            }
                        }

                        mv_t sub_mv[2];
                        if (!assign_mv_(t, b_mode, sub_mv, best_ref, best_sub, is_compound))
                            return -1;
                        bmi_mv[j][0] = sub_mv[0];
                        bmi_mv[j][1] = sub_mv[1];
                        if (n4h == 2) { bmi_mv[j + 2][0] = sub_mv[0]; bmi_mv[j + 2][1] = sub_mv[1]; }
                        if (n4w == 2) { bmi_mv[j + 1][0] = sub_mv[0]; bmi_mv[j + 1][1] = sub_mv[1]; }
                    }
                }
                mode = b_mode;
                mv[0] = bmi_mv[3][0];
                mv[1] = bmi_mv[3][1];
            } else {
                if (mode != ZEROMV) {
                    for (int ri = 0; ri < 1 + is_compound; ri++) {
                        mv_t tmp[2];
                        int n = dec_find_mv_refs_(t, mode, ref[ri], search, tmp,
                                                  mi_row, mi_col, -1, e);
                        lower_mv_precision_(&tmp[n - 1], t->allow_hp);
                        best_ref[ri] = tmp[n - 1];
                    }
                }
                if (!assign_mv_(t, mode, mv, best_ref, best_ref, is_compound))
                    return -1;
                for (int j = 0; j < 4; j++) {
                    bmi_mv[j][0] = mv[0];
                    bmi_mv[j][1] = mv[1];
                }
            }
        }
    }

    if (!t->is_kf && getenv("VP9_TRACE_MI")) {
        fprintf(stderr, "MI %d %d bs=%d skip=%d tx=%d mode=%d ref=%d,%d mv=%d,%d filt=%d\n",
                mi_row, mi_col, bsize, skip, y_tx, mode, ref[0],
                ref[1] > 0 ? ref[1] : -1, mv[0].row, mv[0].col, filter);
    }

    int is_inter_block = ref[0] > 0;
    block->is_intra = (uint8_t)!is_inter_block;
    block->skip = (uint8_t)skip;
    block->tx_size = (uint8_t)y_tx;
    block->y_mode = (uint8_t)mode;
    block->uv_mode = (uint8_t)uv_mode;
    block->ref_frame[0] = (uint8_t)(ref[0] > 0 ? ref[0] : 0);
    block->ref_frame[1] = (uint8_t)(ref[1] > 0 ? ref[1] : 0);
    block->mv[0][0] = mv[0].col;
    block->mv[0][1] = mv[0].row;
    block->mv[1][0] = mv[1].col;
    block->mv[1][1] = mv[1].row;
    block->coeff_offset = pf->num_coeffs;

    /* MV grid for GPU motion compensation */
    if (is_inter_block) {
        int gx0 = mi_col * 2, gy0 = mi_row * 2;
        int gx1 = gx0 + bw_mi * 2, gy1 = gy0 + bh_mi * 2;
        if (gx1 > (int)pf->mv_grid_width) gx1 = pf->mv_grid_width;
        if (gy1 > (int)pf->mv_grid_height) gy1 = pf->mv_grid_height;
        int ref_sel = ref[0] - 1;
        for (int gy = gy0; gy < gy1; gy++) {
            for (int gx = gx0; gx < gx1; gx++) {
                cvp9_mv_t *g = &pf->mv_grid[gy * pf->mv_grid_width + gx];
                g->x = mv[0].col;
                g->y = mv[0].row;
                g->ref = ref_sel;
            }
        }
    }

    /* Store mode info over the block's mi rectangle for neighbor contexts */
    int sub8 = bsize < BLOCK_8X8_;
    for (int ry = 0; ry < bh_mi && mi_row + ry < t->mi_rows; ry++) {
        for (int cx = 0; cx < bw_mi && mi_col + cx < t->mi_cols; cx++) {
            int mi = (mi_row + ry) * t->mi_cols + (mi_col + cx);
            t->mi_mode[mi] = (uint8_t)mode;
            t->mi_skip[mi] = (uint8_t)skip;
            t->mi_txsz[mi] = (uint8_t)y_tx;
            t->mi_sub8[mi] = (uint8_t)sub8;
            memcpy(&t->mi_bmi[mi * 4], bmi, 4);

            t->mi_ref0[mi] = ref[0];
            t->mi_ref1[mi] = ref[1];
            t->mi_filter[mi] = filter;
            if (t->out_ref0) {
                t->out_ref0[mi] = ref[0];
                t->out_ref1[mi] = ref[1];
                t->out_mv[mi * 4 + 0] = mv[0].row;
                t->out_mv[mi * 4 + 1] = mv[0].col;
                t->out_mv[mi * 4 + 2] = mv[1].row;
                t->out_mv[mi * 4 + 3] = mv[1].col;
            }
            t->mi_mv[mi * 4 + 0] = mv[0].row;
            t->mi_mv[mi * 4 + 1] = mv[0].col;
            t->mi_mv[mi * 4 + 2] = mv[1].row;
            t->mi_mv[mi * 4 + 3] = mv[1].col;
            for (int j = 0; j < 4; j++) {
                t->mi_bmi_mv[(mi * 4 + j) * 4 + 0] = bmi_mv[j][0].row;
                t->mi_bmi_mv[(mi * 4 + j) * 4 + 1] = bmi_mv[j][0].col;
                t->mi_bmi_mv[(mi * 4 + j) * 4 + 2] = bmi_mv[j][1].row;
                t->mi_bmi_mv[(mi * 4 + j) * 4 + 3] = bmi_mv[j][1].col;
            }

            /* Legacy grids for merge/GPU compatibility */
            uint32_t gidx = (uint32_t)mi;
            pf->mi_width_grid[gidx] = block->width;
            pf->mi_height_grid[gidx] = block->height;
            pf->mi_block_grid[gidx] = pf->num_blocks + 1;
        }
    }

    if (decode_block_tokens(t, mi_row, mi_col, bsize, y_tx, skip,
                            is_inter_block, block) != 0)
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
    const uint8_t *probs = t->is_kf ? vp9_kf_partition_probs_tbl[ctx]
                                    : t->fc->partition_probs[ctx];
    int p;

    if (has_rows && has_cols) {
        if (!vpx_read(t->r, probs[0])) p = PARTITION_NONE_;
        else if (!vpx_read(t->r, probs[1])) p = PARTITION_HORZ_;
        else p = vpx_read(t->r, probs[2]) ? PARTITION_SPLIT_ : PARTITION_VERT_;
    } else if (!has_rows && has_cols) {
        p = vpx_read(t->r, probs[1]) ? PARTITION_SPLIT_ : PARTITION_HORZ_;
    } else if (has_rows && !has_cols) {
        p = vpx_read(t->r, probs[2]) ? PARTITION_SPLIT_ : PARTITION_VERT_;
    } else {
        p = PARTITION_SPLIT_;
    }
    if (t->counts) t->counts->partition[ctx][p]++;
    return p;
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

int vp9_decode_tile_conformant(vpx_reader *r,
                               int mi_row_start, int mi_row_end,
                               int mi_col_start, int mi_col_end,
                               const vp9_entropy_probs_t *probs,
                               vp9_parsed_frame_t *pf,
                               vp9_counts_t *counts,
                               const int8_t *prev_ref0, const int8_t *prev_ref1,
                               const int16_t *prev_mv, int use_prev_mvs,
                               int8_t *out_ref0, int8_t *out_ref1, int16_t *out_mv)
{
    kf_tile_ctx_t t = {0};
    t.counts = counts;
    t.prev_ref0 = prev_ref0;
    t.prev_ref1 = prev_ref1;
    t.prev_mv = prev_mv;
    t.use_prev_mvs = use_prev_mvs && prev_ref0 && prev_mv;
    t.out_ref0 = out_ref0;
    t.out_ref1 = out_ref1;
    t.out_mv = out_mv;
    t.hdr = &pf->hdr;
    t.fc = probs;
    t.pf = pf;
    t.r = r;
    t.mi_rows = (int)pf->mi_grid_height;
    t.mi_cols = (int)pf->mi_grid_width;
    t.tile_col_start = mi_col_start;
    t.tile_col_end = mi_col_end;
    t.lossless = pf->hdr.base_qindex == 0 && pf->hdr.y_dc_delta_q == 0 &&
                 pf->hdr.uv_dc_delta_q == 0 && pf->hdr.uv_ac_delta_q == 0;
    t.is_kf = pf->hdr.frame_type == VP9_FRAME_KEY || pf->hdr.intra_only;
    t.allow_hp = pf->hdr.allow_high_precision_mv;
    for (int i = 1; i < 4; i++)
        t.sign_bias[i] = pf->hdr.ref_frame_sign_bias[i];

    /* setup_compound_reference_mode */
    if (t.sign_bias[LAST_FRAME_] == t.sign_bias[GOLDEN_FRAME_]) {
        t.comp_fixed_ref = ALTREF_FRAME_;
        t.comp_var_ref[0] = LAST_FRAME_;
        t.comp_var_ref[1] = GOLDEN_FRAME_;
    } else if (t.sign_bias[LAST_FRAME_] == t.sign_bias[ALTREF_FRAME_]) {
        t.comp_fixed_ref = GOLDEN_FRAME_;
        t.comp_var_ref[0] = LAST_FRAME_;
        t.comp_var_ref[1] = ALTREF_FRAME_;
    } else {
        t.comp_fixed_ref = LAST_FRAME_;
        t.comp_var_ref[0] = GOLDEN_FRAME_;
        t.comp_var_ref[1] = ALTREF_FRAME_;
    }

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
    t.mi_ref0 = calloc(mi_count, 1);
    t.mi_ref1 = malloc(mi_count); if (t.mi_ref1) memset(t.mi_ref1, 0, mi_count);
    t.mi_mv = calloc(mi_count * 4, sizeof(int16_t));
    t.mi_bmi_mv = calloc(mi_count * 16, sizeof(int16_t));
    t.mi_filter = calloc(mi_count, 1);

    int rc = -1;
    if (t.above_seg && t.above_tok[0] && t.above_tok[1] && t.above_tok[2] &&
        t.mi_mode && t.mi_bmi && t.mi_skip && t.mi_txsz && t.mi_sub8 &&
        t.mi_ref0 && t.mi_ref1 && t.mi_mv && t.mi_bmi_mv && t.mi_filter) {
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
    free(t.mi_ref0);
    free(t.mi_ref1);
    free(t.mi_mv);
    free(t.mi_bmi_mv);
    free(t.mi_filter);
    return rc;
}

int vp9_decode_tile_kf(vpx_reader *r,
                       int mi_row_start, int mi_row_end,
                       int mi_col_start, int mi_col_end,
                       const vp9_entropy_probs_t *probs,
                       vp9_parsed_frame_t *pf)
{
    return vp9_decode_tile_conformant(r, mi_row_start, mi_row_end,
                                      mi_col_start, mi_col_end, probs, pf,
                                      NULL, NULL, NULL, NULL, 0,
                                      NULL, NULL, NULL);
}
