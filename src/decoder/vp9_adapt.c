/**
 * compute-vp9 — backward probability adaptation (libvpx vp9_entropymode.c,
 * vp9_entropymv.c, vp9_entropy.c, vpx_dsp/prob.c)
 *
 * After each decoded frame (when !error_resilient && !frame_parallel), the
 * frame context is re-derived by merging the PRE-frame context with the
 * frame's symbol counts. Bit-exactness of later frames depends on this.
 */
#include "vp9_entropy.h"
#include <string.h>

#define MODE_MV_COUNT_SAT 20
#define MODE_MV_MAX_UPDATE_FACTOR 128
#define COEF_COUNT_SAT 24
#define COEF_MAX_UPDATE_FACTOR 112
#define COEF_COUNT_SAT_AFTER_KEY 24
#define COEF_MAX_UPDATE_FACTOR_AFTER_KEY 128

void vp9_counts_accumulate(vp9_counts_t *a, const vp9_counts_t *b)
{
    unsigned int *pa = (unsigned int *)a;
    const unsigned int *pb = (const unsigned int *)b;
    size_t n = sizeof(*a) / sizeof(unsigned int);
    for (size_t i = 0; i < n; i++) pa[i] += pb[i];
}

/* ── merge primitives (vpx_dsp/prob.h) ──────────────────────────────────── */

static uint8_t clip_prob(int p) { return p > 255 ? 255 : (p < 1 ? 1 : (uint8_t)p); }

static uint8_t get_prob(unsigned int num, unsigned int den)
{
    if (den == 0) return 128;
    return clip_prob((int)(((int64_t)num * 256 + (den >> 1)) / den));
}

static uint8_t weighted_prob(uint8_t p1, uint8_t p2, int factor)
{
    return (uint8_t)((p1 * (256 - factor) + p2 * factor + 128) >> 8);
}

static uint8_t merge_probs(uint8_t pre_prob, const unsigned int ct[2],
                           unsigned int count_sat, unsigned int max_update_factor)
{
    uint8_t prob = get_prob(ct[0], ct[0] + ct[1]);
    unsigned int count = ct[0] + ct[1];
    if (count > count_sat) count = count_sat;
    unsigned int factor = max_update_factor * count / count_sat;
    return weighted_prob(pre_prob, prob, (int)factor);
}

static uint8_t mode_mv_merge_probs(uint8_t pre_prob, const unsigned int ct[2])
{
    return merge_probs(pre_prob, ct, MODE_MV_COUNT_SAT, MODE_MV_MAX_UPDATE_FACTOR);
}

/* Recursive tree merge (vpx_dsp/prob.c) */
static unsigned int tree_merge_impl(unsigned int i, const int8_t *tree,
                                    const uint8_t *pre_probs,
                                    const unsigned int *counts, uint8_t *probs)
{
    const int l = tree[i];
    const unsigned int left_count =
        (l <= 0) ? counts[-l] : tree_merge_impl(l, tree, pre_probs, counts, probs);
    const int r = tree[i + 1];
    const unsigned int right_count =
        (r <= 0) ? counts[-r] : tree_merge_impl(r, tree, pre_probs, counts, probs);
    const unsigned int ct[2] = { left_count, right_count };
    probs[i >> 1] = mode_mv_merge_probs(pre_probs[i >> 1], ct);
    return left_count + right_count;
}

static void tree_merge(const int8_t *tree, const uint8_t *pre_probs,
                       const unsigned int *counts, uint8_t *probs)
{
    tree_merge_impl(0, tree, pre_probs, counts, probs);
}

/* Trees (leaf indices relative to each syntax element) */
static const int8_t intra_mode_tree[] = {
    -0, 2, -9, 4, -1, 6, 8, 12, -2, 10, -4, -5, -3, 14, -8, 16, -6, -7
};
static const int8_t inter_mode_tree[] = { -2, 2, -0, 4, -1, -3 };
static const int8_t partition_tree[] = { -0, 2, -1, 4, -2, -3 };
static const int8_t switchable_tree[] = { -0, 2, -1, -2 };
static const int8_t mv_joint_tree[] = { -0, 2, -1, 4, -2, -3 };
static const int8_t mv_class_tree[] = {
    -0, 2, -1, 4, 6, 8, -2, -3, 10, 12, -4, -5, -6, 14, 16, 18, -7, -8, -9, -10
};
static const int8_t mv_class0_tree[] = { -0, -1 };
static const int8_t mv_fp_tree[] = { -0, 2, -1, 4, -2, -3 };

/* ── coefficient adaptation ─────────────────────────────────────────────── */

static void adapt_coef_probs_tx(vp9_entropy_probs_t *fc,
                                const vp9_entropy_probs_t *pre_fc,
                                const vp9_counts_t *counts, int tx,
                                unsigned int count_sat, unsigned int update_factor)
{
    for (int i = 0; i < PLANE_TYPES; i++) {
        for (int j = 0; j < REF_TYPES; j++) {
            for (int k = 0; k < COEF_BANDS; k++) {
                for (int l = 0; l < (k == 0 ? 3 : COEFF_CONTEXTS); l++) {
                    const unsigned int n0 = counts->coef[tx][i][j][k][l][0];
                    const unsigned int n1 = counts->coef[tx][i][j][k][l][1];
                    const unsigned int n2 = counts->coef[tx][i][j][k][l][2];
                    const unsigned int neob = counts->coef[tx][i][j][k][l][3];
                    const unsigned int eob_total = counts->eob_branch[tx][i][j][k][l];
                    const unsigned int branch_ct[3][2] = {
                        { neob, eob_total - neob },
                        { n0, n1 + n2 },
                        { n1, n2 },
                    };
                    for (int m = 0; m < UNCONSTRAINED_NODES; m++)
                        fc->coef_probs[tx][i][j][k][l][m] =
                            merge_probs(pre_fc->coef_probs[tx][i][j][k][l][m],
                                        branch_ct[m], count_sat, update_factor);
                }
            }
        }
    }
}

void vp9_adapt_probs(vp9_entropy_probs_t *fc, const vp9_entropy_probs_t *pre_fc,
                     const vp9_counts_t *counts, int intra_only,
                     int last_was_key, int allow_hp, int interp_switchable)
{
    unsigned int count_sat, update_factor;
    if (intra_only) {
        count_sat = COEF_COUNT_SAT;
        update_factor = COEF_MAX_UPDATE_FACTOR;
    } else if (last_was_key) {
        count_sat = COEF_COUNT_SAT_AFTER_KEY;
        update_factor = COEF_MAX_UPDATE_FACTOR_AFTER_KEY;
    } else {
        count_sat = COEF_COUNT_SAT;
        update_factor = COEF_MAX_UPDATE_FACTOR;
    }
    for (int tx = 0; tx < TX_SIZES; tx++)
        adapt_coef_probs_tx(fc, pre_fc, counts, tx, count_sat, update_factor);

    if (intra_only) return;

    /* vp9_adapt_mode_probs */
    for (int i = 0; i < 4; i++)
        fc->intra_inter_probs[i] =
            mode_mv_merge_probs(pre_fc->intra_inter_probs[i], counts->intra_inter[i]);
    for (int i = 0; i < 5; i++)
        fc->comp_inter_probs[i] =
            mode_mv_merge_probs(pre_fc->comp_inter_probs[i], counts->comp_inter[i]);
    for (int i = 0; i < 5; i++)
        fc->comp_ref_probs[i] =
            mode_mv_merge_probs(pre_fc->comp_ref_probs[i], counts->comp_ref[i]);
    for (int i = 0; i < 5; i++)
        for (int j = 0; j < 2; j++)
            fc->single_ref_probs[i][j] =
                mode_mv_merge_probs(pre_fc->single_ref_probs[i][j],
                                    counts->single_ref[i][j]);

    for (int i = 0; i < 7; i++)
        tree_merge(inter_mode_tree, pre_fc->inter_mode_probs[i],
                   counts->inter_mode[i], fc->inter_mode_probs[i]);
    for (int i = 0; i < 4; i++)
        tree_merge(intra_mode_tree, pre_fc->y_mode_probs[i],
                   counts->y_mode[i], fc->y_mode_probs[i]);
    for (int i = 0; i < 10; i++)
        tree_merge(intra_mode_tree, pre_fc->uv_mode_probs[i],
                   counts->uv_mode[i], fc->uv_mode_probs[i]);
    for (int i = 0; i < 16; i++)
        tree_merge(partition_tree, pre_fc->partition_probs[i],
                   counts->partition[i], fc->partition_probs[i]);

    if (interp_switchable) {
        for (int i = 0; i < 4; i++)
            tree_merge(switchable_tree, pre_fc->switchable_interp_probs[i],
                       counts->switchable_interp[i], fc->switchable_interp_probs[i]);
    }

    if (fc->tx_mode == VP9_TX_MODE_SELECT) {
        for (int i = 0; i < 2; i++) {
            unsigned int b8[1][2], b16[2][2], b32[3][2];
            b8[0][0] = counts->tx8[i][0];
            b8[0][1] = counts->tx8[i][1];
            b16[0][0] = counts->tx16[i][0];
            b16[0][1] = counts->tx16[i][1] + counts->tx16[i][2];
            b16[1][0] = counts->tx16[i][1];
            b16[1][1] = counts->tx16[i][2];
            b32[0][0] = counts->tx32[i][0];
            b32[0][1] = counts->tx32[i][1] + counts->tx32[i][2] + counts->tx32[i][3];
            b32[1][0] = counts->tx32[i][1];
            b32[1][1] = counts->tx32[i][2] + counts->tx32[i][3];
            b32[2][0] = counts->tx32[i][2];
            b32[2][1] = counts->tx32[i][3];

            fc->tx8_probs[i][0] = mode_mv_merge_probs(pre_fc->tx8_probs[i][0], b8[0]);
            for (int j = 0; j < 2; j++)
                fc->tx16_probs[i][j] = mode_mv_merge_probs(pre_fc->tx16_probs[i][j], b16[j]);
            for (int j = 0; j < 3; j++)
                fc->tx32_probs[i][j] = mode_mv_merge_probs(pre_fc->tx32_probs[i][j], b32[j]);
        }
    }

    for (int i = 0; i < 3; i++)
        fc->skip_probs[i] = mode_mv_merge_probs(pre_fc->skip_probs[i], counts->skip[i]);

    /* vp9_adapt_mv_probs */
    tree_merge(mv_joint_tree, pre_fc->mv_joint_probs, counts->mv_joints,
               fc->mv_joint_probs);
    for (int i = 0; i < 2; i++) {
        fc->mv_sign_probs[i] =
            mode_mv_merge_probs(pre_fc->mv_sign_probs[i], counts->mv_comp[i].sign);
        tree_merge(mv_class_tree, pre_fc->mv_class_probs[i],
                   counts->mv_comp[i].classes, fc->mv_class_probs[i]);
        tree_merge(mv_class0_tree, pre_fc->mv_class0_probs[i],
                   counts->mv_comp[i].class0, fc->mv_class0_probs[i]);
        for (int j = 0; j < 10; j++)
            fc->mv_bits_probs[i][j] =
                mode_mv_merge_probs(pre_fc->mv_bits_probs[i][j],
                                    counts->mv_comp[i].bits[j]);
        for (int j = 0; j < 2; j++)
            tree_merge(mv_fp_tree, pre_fc->mv_class0_fr_probs[i][j],
                       counts->mv_comp[i].class0_fp[j], fc->mv_class0_fr_probs[i][j]);
        tree_merge(mv_fp_tree, pre_fc->mv_fr_probs[i],
                   counts->mv_comp[i].fp, fc->mv_fr_probs[i]);
        if (allow_hp) {
            fc->mv_class0_hp_probs[i] =
                mode_mv_merge_probs(pre_fc->mv_class0_hp_probs[i],
                                    counts->mv_comp[i].class0_hp);
            fc->mv_hp_probs[i] =
                mode_mv_merge_probs(pre_fc->mv_hp_probs[i], counts->mv_comp[i].hp);
        }
    }
}
