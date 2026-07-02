/**
 * compute-vp9 — synthetic VP9 packet generator (spec-conformant headers)
 *
 * Emits keyframe/interframe packets whose uncompressed header follows the
 * VP9 spec (§6.2), a dummy 4-byte compressed header, and zero-filled tile
 * payloads. Shared by the benchmark and the unit tests.
 */
#pragma once

#include <stdint.h>
#include <stddef.h>
#include <string.h>

typedef struct {
    uint8_t *buf;
    size_t pos;
    int bit;
} vp9_synth_bw_t;

static void synth_write_bit(vp9_synth_bw_t *bw, int bit)
{
    if (bw->bit < 0) {
        bw->bit = 7;
        bw->pos++;
    }
    if (bit) {
        bw->buf[bw->pos] |= (1 << bw->bit);
    } else {
        bw->buf[bw->pos] &= ~(1 << bw->bit);
    }
    bw->bit--;
}

static void synth_write_bits(vp9_synth_bw_t *bw, uint32_t val, int n)
{
    for (int i = n - 1; i >= 0; i--) {
        synth_write_bit(bw, (val >> i) & 1);
    }
}

static void synth_align_byte(vp9_synth_bw_t *bw)
{
    if (bw->bit < 7) {
        bw->bit = 7;
        bw->pos++;
    }
}

/* tile_info per spec: increment bits from min_log2 up to the requested
 * log2_tile_cols, then a terminator when below max_log2 */
static void synth_write_tile_info(vp9_synth_bw_t *bw, uint32_t width, int log2_tile_cols)
{
    int mi_cols = ((int)width + 7) >> 3;
    int sb64_cols = (mi_cols + 7) >> 3;

    int min_log2 = 0;
    while ((64 << min_log2) < sb64_cols) min_log2++;
    int max_log2 = 1;
    while ((sb64_cols >> max_log2) >= 4) max_log2++;
    max_log2--;

    if (log2_tile_cols < min_log2) log2_tile_cols = min_log2;
    if (log2_tile_cols > max_log2) log2_tile_cols = max_log2;

    for (int i = min_log2; i < log2_tile_cols; i++) synth_write_bit(bw, 1);
    if (log2_tile_cols < max_log2) synth_write_bit(bw, 0);
    synth_write_bit(bw, 0);  /* log2_tile_rows = 0 */
}

/* Zero-filled tile payload sized so an all-zeros conformant decode (every
 * bool reads 0: NONE partitions, DC modes, immediate EOBs) stays inside the
 * buffer: ~18 bytes per 64x64 superblock, padded generously. */
static size_t synth_tile_payload(uint32_t width, uint32_t height, int log2_tile_cols)
{
    size_t sb_cols = ((width + 63) / 64 >> log2_tile_cols) + 1;
    size_t sb_rows = (height + 63) / 64;
    return 64 + sb_cols * sb_rows * 32;
}

/* Buffer size needed for a synthetic packet */
static size_t vp9_synth_bufsize(uint32_t width, uint32_t height)
{
    return 4096 + ((size_t)(width + 63) / 64) * ((height + 63) / 64) * 40;
}

/* Common tail: frame context, loop filter, quant, segmentation, tiles,
 * header_size, dummy compressed header, zero tile payloads */
static void synth_write_tail(vp9_synth_bw_t *bw, uint32_t width, uint32_t height,
                             int log2_tile_cols)
{
    synth_write_bit(bw, 0);        /* refresh_frame_context */
    synth_write_bit(bw, 0);        /* frame_parallel_decoding_mode */
    synth_write_bits(bw, 0, 2);    /* frame_context_idx */

    /* loop_filter_params */
    synth_write_bits(bw, 32, 6);   /* filter_level */
    synth_write_bits(bw, 0, 3);    /* sharpness */
    synth_write_bit(bw, 0);        /* loop_filter_delta_enabled */

    /* quantization_params */
    synth_write_bits(bw, 128, 8);  /* base_q_idx */
    synth_write_bit(bw, 0);        /* y_dc delta coded */
    synth_write_bit(bw, 0);        /* uv_dc delta coded */
    synth_write_bit(bw, 0);        /* uv_ac delta coded */

    synth_write_bit(bw, 0);        /* segmentation_enabled */

    synth_write_tile_info(bw, width, log2_tile_cols);

    synth_write_bits(bw, 4, 16);   /* header_size_in_bytes (compressed hdr) */
    synth_align_byte(bw);

    /* Dummy compressed header (4 bytes) */
    bw->buf[bw->pos++] = 0x00;
    bw->buf[bw->pos++] = 0x00;
    bw->buf[bw->pos++] = 0x00;
    bw->buf[bw->pos++] = 0x00;

    /* Tile payloads: non-last tiles carry a 4-byte size header */
    int tiles = 1 << log2_tile_cols;
    size_t payload = synth_tile_payload(width, height, log2_tile_cols);
    for (int t = 0; t < tiles; t++) {
        if (t != tiles - 1) {
            bw->buf[bw->pos++] = (uint8_t)(payload >> 24);
            bw->buf[bw->pos++] = (uint8_t)(payload >> 16);
            bw->buf[bw->pos++] = (uint8_t)(payload >> 8);
            bw->buf[bw->pos++] = (uint8_t)payload;
        }
        memset(bw->buf + bw->pos, 0, payload);
        bw->pos += payload;
    }
}

static size_t vp9_synth_keyframe(uint8_t *buf, uint32_t width, uint32_t height,
                                 int log2_tile_cols)
{
    memset(buf, 0, 1024);
    vp9_synth_bw_t bw = { .buf = buf, .pos = 0, .bit = 7 };

    synth_write_bits(&bw, 2, 2);   /* frame_marker */
    synth_write_bit(&bw, 0);       /* profile_low_bit */
    synth_write_bit(&bw, 0);       /* profile_high_bit */
    synth_write_bit(&bw, 0);       /* show_existing_frame */
    synth_write_bit(&bw, 0);       /* frame_type = KEY */
    synth_write_bit(&bw, 1);       /* show_frame */
    synth_write_bit(&bw, 0);       /* error_resilient */

    synth_write_bits(&bw, 0x498342, 24);  /* sync code */

    /* color_config (profile 0) */
    synth_write_bits(&bw, 1, 3);   /* color_space = BT.601 */
    synth_write_bit(&bw, 0);       /* color_range */

    synth_write_bits(&bw, width - 1, 16);
    synth_write_bits(&bw, height - 1, 16);
    synth_write_bit(&bw, 0);       /* render_and_frame_size_different */

    synth_write_tail(&bw, width, height, log2_tile_cols);
    return bw.pos;
}

static size_t vp9_synth_interframe(uint8_t *buf, uint32_t width, uint32_t height,
                                   int log2_tile_cols)
{
    memset(buf, 0, 1024);
    vp9_synth_bw_t bw = { .buf = buf, .pos = 0, .bit = 7 };

    synth_write_bits(&bw, 2, 2);   /* frame_marker */
    synth_write_bit(&bw, 0);       /* profile_low_bit */
    synth_write_bit(&bw, 0);       /* profile_high_bit */
    synth_write_bit(&bw, 0);       /* show_existing_frame */
    synth_write_bit(&bw, 1);       /* frame_type = NON-KEY */
    synth_write_bit(&bw, 1);       /* show_frame */
    synth_write_bit(&bw, 0);       /* error_resilient */

    synth_write_bits(&bw, 0, 2);   /* reset_frame_context */
    synth_write_bits(&bw, 0x01, 8); /* refresh_frame_flags: slot 0 */
    for (int i = 0; i < 3; i++) {
        synth_write_bits(&bw, 0, 3);  /* ref_frame_idx[i] */
        synth_write_bit(&bw, 0);      /* ref_frame_sign_bias */
    }

    /* frame_size_with_refs: no ref match → explicit size */
    synth_write_bit(&bw, 0);
    synth_write_bit(&bw, 0);
    synth_write_bit(&bw, 0);
    synth_write_bits(&bw, width - 1, 16);
    synth_write_bits(&bw, height - 1, 16);
    synth_write_bit(&bw, 0);       /* render_and_frame_size_different */

    synth_write_bit(&bw, 0);       /* allow_high_precision_mv */
    synth_write_bit(&bw, 0);       /* is_filter_switchable */
    synth_write_bits(&bw, 1, 2);   /* literal 1 = EIGHTTAP */

    synth_write_tail(&bw, width, height, log2_tile_cols);
    return bw.pos;
}
