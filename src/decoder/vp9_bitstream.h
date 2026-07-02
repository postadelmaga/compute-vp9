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

/* VP9 interpolation filters (header order after literal remap) */
typedef enum {
    VP9_FILTER_EIGHTTAP        = 0,
    VP9_FILTER_EIGHTTAP_SMOOTH = 1,
    VP9_FILTER_EIGHTTAP_SHARP  = 2,
    VP9_FILTER_BILINEAR        = 3,
    VP9_FILTER_SWITCHABLE      = 4,
} vp9_interp_filter_t;

/* VP9 frame header (parsed from bitstream) */
typedef struct {
    vp9_frame_type_t   frame_type;
    uint32_t           width;
    uint32_t           height;
    uint32_t           render_width;
    uint32_t           render_height;
    uint8_t            profile;
    uint8_t            bit_depth;
    vp9_color_space_t  color_space;
    bool               color_range;
    uint8_t            subsampling_x;
    uint8_t            subsampling_y;
    bool               show_frame;
    bool               show_existing_frame;
    uint8_t            frame_to_show_map_idx;
    bool               intra_only;
    uint8_t            reset_frame_context;
    bool               error_resilient;
    bool               allow_high_precision_mv;
    vp9_interp_filter_t interp_filter;
    bool               refresh_frame_context;
    bool               frame_parallel;
    uint8_t            frame_context_idx;
    uint8_t            log2_tile_cols;
    uint8_t            log2_tile_rows;

    /* Byte layout: uncompressed header size (aligned) and the size of the
     * compressed header that follows it (header_size_in_bytes) */
    uint32_t           uncompressed_header_bytes;
    uint16_t           first_partition_size;

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
/* ref_width/ref_height: dimensions of the active reference frames, needed
 * when an inter frame inherits its size (frame_size_with_refs) — tile_info
 * limits depend on the frame width. Pass 0 if unknown. */
int     vp9_parse_frame_header(vp9_bitreader_t *br, vp9_frame_header_t *hdr,
                               uint32_t ref_width, uint32_t ref_height);
