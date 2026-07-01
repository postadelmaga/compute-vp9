/**
 * compute-vp9 — CPU reconstruction engine implementation
 */
#include "vp9_reconstruct.h"
#include "vp9_frame.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define CLAMP(x, l, h) (((x) < (l)) ? (l) : (((x) > (h)) ? (h) : (x)))

/* ── Generic 2D Inverse DCT ──────────────────────────────────────────────── */
static void vp9_idct_2d(const int16_t *in, int16_t *out, int N)
{
    double M[32][32];
    double temp[32][32];
    double out_d[32][32];

    double scale_zero = sqrt(1.0 / N);
    double scale_other = sqrt(2.0 / N);

    /* Build IDCT transform matrix */
    for (int i = 0; i < N; i++) {
        double scale = (i == 0) ? scale_zero : scale_other;
        for (int j = 0; j < N; j++) {
            M[i][j] = scale * cos(((2.0 * j + 1.0) * i * 3.141592653589793) / (2.0 * N));
        }
    }

    /* Temp = M^T * In */
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double sum = 0.0;
            for (int k = 0; k < N; k++) {
                sum += M[k][i] * in[k * N + j];
            }
            temp[i][j] = sum;
        }
    }

    /* Out = Temp * M */
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double sum = 0.0;
            for (int k = 0; k < N; k++) {
                sum += temp[i][k] * M[k][j];
            }
            out_d[i][j] = sum;
        }
    }

    /* Round and clamp */
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double val = out_d[i][j];
            int32_t rounded = (int32_t)(val + (val >= 0 ? 0.5 : -0.5));
            out[i * N + j] = (int16_t)CLAMP(rounded, -32768, 32767);
        }
    }
}

/* ── Intra Predictors ────────────────────────────────────────────────────── */
static void intra_predict(uint8_t *dst, int stride, int w, int h,
                          const uint8_t *above, const uint8_t *left,
                          uint8_t top_left, int mode)
{
    if (mode == 0) { /* DC_PRED */
        uint32_t sum = 0;
        for (int i = 0; i < w; i++) sum += above[i];
        for (int i = 0; i < h; i++) sum += left[i];
        uint8_t dc = (uint8_t)((sum + ((w + h) >> 1)) / (w + h));
        for (int r = 0; r < h; r++) {
            memset(dst + r * stride, dc, w);
        }
    } else if (mode == 1) { /* V_PRED */
        for (int r = 0; r < h; r++) {
            memcpy(dst + r * stride, above, w);
        }
    } else if (mode == 2) { /* H_PRED */
        for (int r = 0; r < h; r++) {
            memset(dst + r * stride, left[r], w);
        }
    } else { /* TM_PRED */
        for (int r = 0; r < h; r++) {
            for (int c = 0; c < w; c++) {
                int val = above[c] + left[r] - top_left;
                dst[r * stride + c] = (uint8_t)CLAMP(val, 0, 255);
            }
        }
    }
}

/* ── Motion Compensation (Bilinear Interpolation) ─────────────────────────── */
static void motion_compensate(uint8_t *dst, int dst_stride,
                              const uint8_t *ref, int ref_stride,
                              int w, int h, int block_x, int block_y,
                              int mv_x, int mv_y, int ref_w, int ref_h)
{
    /* MV is in 1/8-pel units */
    int ref_x8 = block_x * 8 + mv_x;
    int ref_y8 = block_y * 8 + mv_y;

    int int_x = ref_x8 >> 3;
    int int_y = ref_y8 >> 3;
    int frac_x = ref_x8 & 7;
    int frac_y = ref_y8 & 7;

    for (int r = 0; r < h; r++) {
        for (int c = 0; c < w; c++) {
            int px = int_x + c;
            int py = int_y + r;

            int px0 = CLAMP(px,     0, ref_w - 1);
            int px1 = CLAMP(px + 1, 0, ref_w - 1);
            int py0 = CLAMP(py,     0, ref_h - 1);
            int py1 = CLAMP(py + 1, 0, ref_h - 1);

            uint8_t p00 = ref[py0 * ref_stride + px0];
            uint8_t p10 = ref[py0 * ref_stride + px1];
            uint8_t p01 = ref[py1 * ref_stride + px0];
            uint8_t p11 = ref[py1 * ref_stride + px1];

            /* Bilinear interpolation */
            uint32_t top = (p00 * (8 - frac_x) + p10 * frac_x + 4) >> 3;
            uint32_t bot = (p01 * (8 - frac_x) + p11 * frac_x + 4) >> 3;
            uint32_t predicted = (top * (8 - frac_y) + bot * frac_y + 4) >> 3;

            dst[r * dst_stride + c] = (uint8_t)predicted;
        }
    }
}

/* ── Loop Deblocking Filter ──────────────────────────────────────────────── */
static void apply_deblocking_filter(cvp9_frame_info_t *f, uint8_t level, uint8_t sharpness)
{
    if (level == 0) return;

    /* Derivation of threshold limit */
    uint32_t threshold = level;
    if (sharpness > 0) {
        threshold >>= (sharpness > 4) ? 2 : 1;
        if (threshold > 9 - sharpness) threshold = 9 - sharpness;
    }
    if (threshold == 0) threshold = 1;

    /* Horizontal boundaries pass */
    for (uint32_t y = 8; y < f->height; y += 8) {
        for (uint32_t x = 0; x < f->width; x++) {
            int p1_off = (y - 2) * f->stride_y + x;
            int p0_off = (y - 1) * f->stride_y + x;
            int q0_off = (y + 0) * f->stride_y + x;
            int q1_off = (y + 1) * f->stride_y + x;

            int p1 = f->plane_y[p1_off];
            int p0 = f->plane_y[p0_off];
            int q0 = f->plane_y[q0_off];
            int q1 = f->plane_y[q1_off];

            if (abs(p0 - q0) * 2 + abs(p1 - q1) / 2 <= (int)threshold) {
                int delta = CLAMP((3 * (q0 - p0) + 4) >> 3, -(int)threshold, (int)threshold);
                f->plane_y[p0_off] = (uint8_t)CLAMP(p0 + delta, 0, 255);
                f->plane_y[q0_off] = (uint8_t)CLAMP(q0 - delta, 0, 255);
                f->plane_y[p1_off] = (uint8_t)CLAMP(p1 + delta / 2, 0, 255);
                f->plane_y[q1_off] = (uint8_t)CLAMP(q1 - delta / 2, 0, 255);
            }
        }
    }

    /* Vertical boundaries pass */
    for (uint32_t x = 8; x < f->width; x += 8) {
        for (uint32_t y = 0; y < f->height; y++) {
            int p1_off = y * f->stride_y + x - 2;
            int p0_off = y * f->stride_y + x - 1;
            int q0_off = y * f->stride_y + x + 0;
            int q1_off = y * f->stride_y + x + 1;

            int p1 = f->plane_y[p1_off];
            int p0 = f->plane_y[p0_off];
            int q0 = f->plane_y[q0_off];
            int q1 = f->plane_y[q1_off];

            if (abs(p0 - q0) * 2 + abs(p1 - q1) / 2 <= (int)threshold) {
                int delta = CLAMP((3 * (q0 - p0) + 4) >> 3, -(int)threshold, (int)threshold);
                f->plane_y[p0_off] = (uint8_t)CLAMP(p0 + delta, 0, 255);
                f->plane_y[q0_off] = (uint8_t)CLAMP(q0 - delta, 0, 255);
                f->plane_y[p1_off] = (uint8_t)CLAMP(p1 + delta / 2, 0, 255);
                f->plane_y[q1_off] = (uint8_t)CLAMP(q1 - delta / 2, 0, 255);
            }
        }
    }
}

/* ── Main Reconstruction Logic ───────────────────────────────────────────── */
bool vp9_reconstruct_frame(const vp9_parsed_frame_t *pf,
                           const cvp9_frame_info_t *ref_frames[8],
                           cvp9_frame_info_t *out_frame)
{
    if (!cvp9_frame_alloc(out_frame, pf->hdr.width, pf->hdr.height)) return false;

    uint8_t above_luma[4096];
    uint8_t left_luma[4096];
    uint8_t above_chroma[2048];
    uint8_t left_chroma[2048];
    memset(above_luma, 127, sizeof(above_luma));
    memset(left_luma, 127, sizeof(left_luma));
    memset(above_chroma, 127, sizeof(above_chroma));
    memset(left_chroma, 127, sizeof(left_chroma));

    /* Process block by block */
    for (uint32_t b = 0; b < pf->num_blocks; b++) {
        const vp9_macroblock_info_t *block = &pf->blocks[b];

        int bx = block->x;
        int by = block->y;
        int bw = block->width;
        int bh = block->height;

        /* Clamp block parameters to frame size */
        if (bx + bw > (int)pf->hdr.width) bw = pf->hdr.width - bx;
        if (by + bh > (int)pf->hdr.height) bh = pf->hdr.height - by;
        if (bw <= 0 || bh <= 0) continue;

        /* 1. Luma Plane Reconstruction */
        if (block->is_intra) {
            /* Grab neighboring pixels for Y prediction */
            uint8_t top_left = 127;
            if (by > 0) {
                for (int i = 0; i < bw; i++) {
                    above_luma[i] = out_frame->plane_y[(by - 1) * out_frame->stride_y + bx + i];
                }
                if (bx > 0) {
                    top_left = out_frame->plane_y[(by - 1) * out_frame->stride_y + bx - 1];
                }
            } else {
                memset(above_luma, 127, bw);
            }

            if (bx > 0) {
                for (int i = 0; i < bh; i++) {
                    left_luma[i] = out_frame->plane_y[(by + i) * out_frame->stride_y + bx - 1];
                }
            } else {
                memset(left_luma, 127, bh);
            }

            uint8_t *dst_y = &out_frame->plane_y[by * out_frame->stride_y + bx];
            intra_predict(dst_y, out_frame->stride_y, bw, bh, above_luma, left_luma, top_left, block->y_mode);
        } else {
            /* Inter-frame prediction */
            int ref_idx = block->ref_frame[0] - 1;
            const cvp9_frame_info_t *ref = (ref_idx >= 0 && ref_idx < 8) ? ref_frames[ref_idx] : NULL;

            uint8_t *dst_y = &out_frame->plane_y[by * out_frame->stride_y + bx];
            if (ref && ref->plane_y) {
                motion_compensate(dst_y, out_frame->stride_y, ref->plane_y, ref->stride_y,
                                  bw, bh, bx, by, block->mv[0][0], block->mv[0][1],
                                  ref->width, ref->height);
            } else {
                /* Reference missing fallback: fill flat grey */
                for (int r = 0; r < bh; r++) {
                    memset(dst_y + r * out_frame->stride_y, 127, bw);
                }
            }
        }

        /* 2. Chroma Plane Reconstruction (Sub-sampled YUV 4:2:0) */
        int cbx = bx / 2;
        int cby = by / 2;
        int cbw = (bw + 1) / 2;
        int cbh = (bh + 1) / 2;

        if (block->is_intra) {
            uint8_t top_left_u = 127, top_left_v = 127;
            if (cby > 0) {
                for (int i = 0; i < cbw; i++) {
                    above_chroma[i] = out_frame->plane_u[(cby - 1) * out_frame->stride_uv + cbx + i];
                }
                if (cbx > 0) {
                    top_left_u = out_frame->plane_u[(cby - 1) * out_frame->stride_uv + cbx - 1];
                }
            } else {
                memset(above_chroma, 127, cbw);
            }

            if (cbx > 0) {
                for (int i = 0; i < cbh; i++) {
                    left_chroma[i] = out_frame->plane_u[(cby + i) * out_frame->stride_uv + cbx - 1];
                }
            } else {
                memset(left_chroma, 127, cbh);
            }

            uint8_t *dst_u = &out_frame->plane_u[cby * out_frame->stride_uv + cbx];
            intra_predict(dst_u, out_frame->stride_uv, cbw, cbh, above_chroma, left_chroma, top_left_u, block->uv_mode);

            /* Repeat for V plane */
            if (cby > 0) {
                for (int i = 0; i < cbw; i++) {
                    above_chroma[i] = out_frame->plane_v[(cby - 1) * out_frame->stride_uv + cbx + i];
                }
                if (cbx > 0) {
                    top_left_v = out_frame->plane_v[(cby - 1) * out_frame->stride_uv + cbx - 1];
                }
            }
            if (cbx > 0) {
                for (int i = 0; i < cbh; i++) {
                    left_chroma[i] = out_frame->plane_v[(cby + i) * out_frame->stride_uv + cbx - 1];
                }
            }
            uint8_t *dst_v = &out_frame->plane_v[cby * out_frame->stride_uv + cbx];
            intra_predict(dst_v, out_frame->stride_uv, cbw, cbh, above_chroma, left_chroma, top_left_v, block->uv_mode);
        } else {
            int ref_idx = block->ref_frame[0] - 1;
            const cvp9_frame_info_t *ref = (ref_idx >= 0 && ref_idx < 8) ? ref_frames[ref_idx] : NULL;

            uint8_t *dst_u = &out_frame->plane_u[cby * out_frame->stride_uv + cbx];
            uint8_t *dst_v = &out_frame->plane_v[cby * out_frame->stride_uv + cbx];

            if (ref && ref->plane_u && ref->plane_v) {
                /* Scale chroma motion vectors */
                int cmv_x = block->mv[0][0] / 2;
                int cmv_y = block->mv[0][1] / 2;

                motion_compensate(dst_u, out_frame->stride_uv, ref->plane_u, ref->stride_uv,
                                  cbw, cbh, cbx, cby, cmv_x, cmv_y, ref->width / 2, ref->height / 2);
                motion_compensate(dst_v, out_frame->stride_uv, ref->plane_v, ref->stride_uv,
                                  cbw, cbh, cbx, cby, cmv_x, cmv_y, ref->width / 2, ref->height / 2);
            } else {
                for (int r = 0; r < cbh; r++) {
                    memset(dst_u + r * out_frame->stride_uv, 127, cbw);
                    memset(dst_v + r * out_frame->stride_uv, 127, cbw);
                }
            }
        }

        /* 3. Residual Addition (IDCT) */
        if (!block->skip) {
            int tx_size_px = 1 << (block->tx_size + 2);
            int num_tx_x = bw / tx_size_px;
            int num_tx_y = bh / tx_size_px;
            int tx_area = tx_size_px * tx_size_px;

            int16_t idct_out[1024];

            for (int ty = 0; ty < num_tx_y; ty++) {
                for (int tx = 0; tx < num_tx_x; tx++) {
                    uint32_t offset = block->coeff_offset + (ty * num_tx_x + tx) * tx_area;
                    const int16_t *src_coeff = &pf->coeffs[offset];

                    vp9_idct_2d(src_coeff, idct_out, tx_size_px);

                    /* Add residuals back to dst frame */
                    for (int r = 0; r < tx_size_px; r++) {
                        for (int c = 0; c < tx_size_px; c++) {
                            int dst_pixel_idx = (by + ty * tx_size_px + r) * out_frame->stride_y + (bx + tx * tx_size_px + c);
                            int16_t pred_val = out_frame->plane_y[dst_pixel_idx];
                            int16_t residual = idct_out[r * tx_size_px + c];
                            out_frame->plane_y[dst_pixel_idx] = (uint8_t)CLAMP(pred_val + residual, 0, 255);
                        }
                    }
                }
            }
        }
    }

    /* 4. Deblocking Filter */
    apply_deblocking_filter(out_frame, pf->hdr.filter_level, pf->hdr.sharpness_level);

    return true;
}
