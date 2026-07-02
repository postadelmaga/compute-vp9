/**
 * compute-vp9 — VP9 compressed header parser (spec §6.3)
 *
 * Reads the bool-coded probability-update partition that sits between the
 * uncompressed header and the tile data, applying the updates onto the
 * frame context. Field order mirrors libvpx read_compressed_header().
 */
#include "vp9_entropy.h"
#include "vp9_bitstream.h"
#include "vp9_default_probs.h"

#define DIFF_UPDATE_PROB 252
#define MV_UPDATE_PROB   252
#define VP9_MAX_PROB     255

/* ── Sub-exponential prob delta coding (libvpx vp9_dsubexp.c) ───────────── */

static int inv_recenter_nonneg(int v, int m)
{
    if (v > 2 * m) return v;
    return (v & 1) ? m - ((v + 1) >> 1) : m + (v >> 1);
}

static int decode_term_subexp(vpx_reader *r)
{
    if (!vpx_read_bit(r)) return vpx_read_literal(r, 4);
    if (!vpx_read_bit(r)) return vpx_read_literal(r, 4) + 16;
    if (!vpx_read_bit(r)) return vpx_read_literal(r, 5) + 32;

    int v = vpx_read_literal(r, 7);
    if (v < 65) return v + 64;
    return (v << 1) - 1 + vpx_read_bit(r);
}

static uint8_t inv_remap_prob(int v, int m)
{
    v = vp9_inv_map_table[v];
    m--;
    if ((m << 1) <= VP9_MAX_PROB) {
        return (uint8_t)(1 + inv_recenter_nonneg(v, m));
    }
    return (uint8_t)(VP9_MAX_PROB - inv_recenter_nonneg(v, VP9_MAX_PROB - 1 - m));
}

static void diff_update_prob(vpx_reader *r, uint8_t *p)
{
    if (vpx_read(r, DIFF_UPDATE_PROB)) {
        int delp = decode_term_subexp(r);
        *p = inv_remap_prob(delp, *p);
    }
}

static void update_mv_probs(vpx_reader *r, uint8_t *p, int n)
{
    for (int i = 0; i < n; i++) {
        if (vpx_read(r, MV_UPDATE_PROB)) {
            p[i] = (uint8_t)((vpx_read_literal(r, 7) << 1) | 1);
        }
    }
}

/* ── Header sections ────────────────────────────────────────────────────── */

static int read_tx_mode(vpx_reader *r)
{
    int mode = vpx_read_literal(r, 2);
    if (mode == VP9_TX_ALLOW_32X32) mode += vpx_read_bit(r);
    return mode;
}

static void read_coef_probs(vpx_reader *r, vp9_entropy_probs_t *fc, int tx_mode)
{
    static const int biggest_tx[5] = { 0, 1, 2, 3, 3 };
    int max_tx_size = biggest_tx[tx_mode];

    for (int tx = 0; tx <= max_tx_size; tx++) {
        if (!vpx_read_bit(r)) continue;
        for (int i = 0; i < PLANE_TYPES; i++)
            for (int j = 0; j < REF_TYPES; j++)
                for (int k = 0; k < COEF_BANDS; k++)
                    for (int l = 0; l < (k == 0 ? 3 : COEFF_CONTEXTS); l++)
                        for (int m = 0; m < UNCONSTRAINED_NODES; m++)
                            diff_update_prob(r, &fc->coef_probs[tx][i][j][k][l][m]);
    }
}

static void read_mv_probs(vpx_reader *r, vp9_entropy_probs_t *fc, bool allow_hp)
{
    update_mv_probs(r, fc->mv_joint_probs, 3);

    for (int i = 0; i < 2; i++) {
        update_mv_probs(r, &fc->mv_sign_probs[i], 1);
        update_mv_probs(r, fc->mv_class_probs[i], 10);
        update_mv_probs(r, fc->mv_class0_probs[i], 1);
        update_mv_probs(r, fc->mv_bits_probs[i], 10);
    }

    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++)
            update_mv_probs(r, fc->mv_class0_fr_probs[i][j], 3);
        update_mv_probs(r, fc->mv_fr_probs[i], 3);
    }

    if (allow_hp) {
        for (int i = 0; i < 2; i++) {
            update_mv_probs(r, &fc->mv_class0_hp_probs[i], 1);
            update_mv_probs(r, &fc->mv_hp_probs[i], 1);
        }
    }
}

int vp9_parse_compressed_header(const vp9_frame_header_t *hdr,
                                const uint8_t *data, size_t size,
                                vp9_entropy_probs_t *fc)
{
    vpx_reader r;
    if (vpx_reader_init(&r, data, size)) return -1;  /* marker bit must be 0 */

    bool lossless = hdr->base_qindex == 0 && hdr->y_dc_delta_q == 0 &&
                    hdr->uv_dc_delta_q == 0 && hdr->uv_ac_delta_q == 0;

    fc->tx_mode = lossless ? VP9_TX_ONLY_4X4 : (uint8_t)read_tx_mode(&r);
    if (fc->tx_mode == VP9_TX_MODE_SELECT) {
        for (int i = 0; i < 2; i++)
            diff_update_prob(&r, &fc->tx8_probs[i][0]);
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 2; j++)
                diff_update_prob(&r, &fc->tx16_probs[i][j]);
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 3; j++)
                diff_update_prob(&r, &fc->tx32_probs[i][j]);
    }

    read_coef_probs(&r, fc, fc->tx_mode);

    for (int i = 0; i < 3; i++)
        diff_update_prob(&r, &fc->skip_probs[i]);

    bool intra_only_frame = hdr->frame_type == VP9_FRAME_KEY || hdr->intra_only;
    if (!intra_only_frame) {
        for (int i = 0; i < 7; i++)
            for (int j = 0; j < 3; j++)
                diff_update_prob(&r, &fc->inter_mode_probs[i][j]);

        if (hdr->interp_filter == VP9_FILTER_SWITCHABLE) {
            for (int i = 0; i < 4; i++)
                for (int j = 0; j < 2; j++)
                    diff_update_prob(&r, &fc->switchable_interp_probs[i][j]);
        }

        for (int i = 0; i < 4; i++)
            diff_update_prob(&r, &fc->intra_inter_probs[i]);

        /* read_frame_reference_mode */
        bool compound_ok =
            hdr->ref_frame_sign_bias[1] != hdr->ref_frame_sign_bias[2] ||
            hdr->ref_frame_sign_bias[1] != hdr->ref_frame_sign_bias[3];
        int ref_mode = VP9_SINGLE_REFERENCE;
        if (compound_ok) {
            ref_mode = vpx_read_bit(&r)
                ? (vpx_read_bit(&r) ? VP9_REFERENCE_MODE_SELECT
                                    : VP9_COMPOUND_REFERENCE)
                : VP9_SINGLE_REFERENCE;
        }
        fc->reference_mode = (uint8_t)ref_mode;

        if (ref_mode == VP9_REFERENCE_MODE_SELECT)
            for (int i = 0; i < 5; i++)
                diff_update_prob(&r, &fc->comp_inter_probs[i]);
        if (ref_mode != VP9_COMPOUND_REFERENCE)
            for (int i = 0; i < 5; i++)
                for (int j = 0; j < 2; j++)
                    diff_update_prob(&r, &fc->single_ref_probs[i][j]);
        if (ref_mode != VP9_SINGLE_REFERENCE)
            for (int i = 0; i < 5; i++)
                diff_update_prob(&r, &fc->comp_ref_probs[i]);

        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 9; j++)
                diff_update_prob(&r, &fc->y_mode_probs[i][j]);

        for (int i = 0; i < 16; i++)
            for (int j = 0; j < 3; j++)
                diff_update_prob(&r, &fc->partition_probs[i][j]);

        read_mv_probs(&r, fc, hdr->allow_high_precision_mv);
    }

    return vpx_reader_has_error(&r) ? -1 : 0;
}
