/**
 * compute-vp9 — internal VP9 bitstream parser
 */
#pragma once

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

/* VP9 frame types */
typedef enum {
    VP9_FRAME_KEY      = 0,
    VP9_FRAME_NON_KEY  = 1,
    VP9_FRAME_INTRA_ONLY = 2,
    VP9_FRAME_S        = 3,
} vp9_frame_type_t;

/* VP9 color spaces */
typedef enum {
    VP9_CS_UNKNOWN   = 0,
    VP9_CS_BT601     = 1,
    VP9_CS_BT709     = 2,
    VP9_CS_SMPTE170  = 3,
    VP9_CS_SMPTE240  = 4,
    VP9_CS_BT2020    = 5,
    VP9_CS_SRGB      = 7,
} vp9_color_space_t;

/* VP9 frame header (parsed from bitstream) */
typedef struct {
    vp9_frame_type_t   frame_type;
    uint32_t           width;
    uint32_t           height;
    uint8_t            profile;
    uint8_t            bit_depth;
    vp9_color_space_t  color_space;
    bool               show_frame;
    bool               error_resilient;
    bool               frame_parallel;
    uint8_t            log2_tile_cols;
    uint8_t            log2_tile_rows;

    /* Quantization */
    int16_t            base_qindex;   /* 0..255 (int16 to survive delta_q math) */
    int8_t             y_dc_delta_q;
    int8_t             uv_dc_delta_q;
    int8_t             uv_ac_delta_q;

    /* Loop filter */
    uint8_t            filter_level;
    uint8_t            sharpness_level;

    /* Reference frames */
    uint8_t            refresh_frame_flags;  /* bitmask: ref slots updated by this frame */
    uint8_t            ref_frame_idx[3];     /* slot for LAST / GOLDEN / ALTREF */
    uint8_t            ref_frame_sign_bias[4];

    /* Segmentation */
    bool               segmentation_enabled;
} vp9_frame_header_t;

/* Bit reader context */
typedef struct {
    const uint8_t *data;
    size_t         size;
    size_t         pos;      /* byte position */
    int            bit;      /* current bit offset within byte */
} vp9_bitreader_t;

/* ── Parser API ─────────────────────────────────────────────────────────── */

void    vp9_bitreader_init(vp9_bitreader_t *br, const uint8_t *data, size_t size);
uint32_t vp9_read_bits(vp9_bitreader_t *br, int n);
bool    vp9_read_bit(vp9_bitreader_t *br);
int     vp9_parse_frame_header(vp9_bitreader_t *br, vp9_frame_header_t *hdr);
