#include "vp9_bitstream.h"
#include <string.h>

/* ── Bit reader ─────────────────────────────────────────────────────────── */

void vp9_bitreader_init(vp9_bitreader_t *br, const uint8_t *data, size_t size)
{
    br->data = data;
    br->size = size;
    br->pos  = 0;
    br->bit  = 7;
}

bool vp9_read_bit(vp9_bitreader_t *br)
{
    if (br->pos >= br->size) return 0;
    bool v = (br->data[br->pos] >> br->bit) & 1;
    if (--br->bit < 0) { br->bit = 7; br->pos++; }
    return v;
}

uint32_t vp9_read_bits(vp9_bitreader_t *br, int n)
{
    uint32_t v = 0;
    for (int i = n - 1; i >= 0; i--)
        v |= (uint32_t)vp9_read_bit(br) << i;
    return v;
}

/* ── Frame header parser (spec-conformant uncompressed header, §6.2) ────── */

static const uint8_t VP9_SYNC_CODE[3] = {0x49, 0x83, 0x42};

/* su(n): n-bit magnitude followed by a sign bit */
static int read_inv_signed(vp9_bitreader_t *br, int n)
{
    int value = (int)vp9_read_bits(br, n);
    return vp9_read_bit(br) ? -value : value;
}

/* delta_q(): update flag + su(4) */
static int read_delta_q(vp9_bitreader_t *br)
{
    return vp9_read_bit(br) ? read_inv_signed(br, 4) : 0;
}

static int parse_color_config(vp9_bitreader_t *br, vp9_frame_header_t *hdr)
{
    if (hdr->profile >= 2) {
        hdr->bit_depth = vp9_read_bit(br) ? 12 : 10;
    } else {
        hdr->bit_depth = 8;
    }
    hdr->color_space = vp9_read_bits(br, 3);
    if (hdr->color_space != VP9_CS_SRGB) {
        hdr->color_range = vp9_read_bit(br);
        if (hdr->profile == 1 || hdr->profile == 3) {
            hdr->subsampling_x = vp9_read_bit(br);
            hdr->subsampling_y = vp9_read_bit(br);
            vp9_read_bit(br);  /* reserved_zero */
        } else {
            hdr->subsampling_x = 1;
            hdr->subsampling_y = 1;
        }
    } else {
        hdr->color_range = 1;
        if (hdr->profile == 1 || hdr->profile == 3) {
            hdr->subsampling_x = 0;
            hdr->subsampling_y = 0;
            vp9_read_bit(br);  /* reserved_zero */
        } else {
            return -1;  /* RGB requires profile 1 or 3 */
        }
    }
    return 0;
}

static void parse_frame_size(vp9_bitreader_t *br, vp9_frame_header_t *hdr)
{
    hdr->width  = vp9_read_bits(br, 16) + 1;
    hdr->height = vp9_read_bits(br, 16) + 1;
}

static void parse_render_size(vp9_bitreader_t *br, vp9_frame_header_t *hdr)
{
    if (vp9_read_bit(br)) {  /* render_and_frame_size_different */
        hdr->render_width  = vp9_read_bits(br, 16) + 1;
        hdr->render_height = vp9_read_bits(br, 16) + 1;
    } else {
        hdr->render_width  = hdr->width;
        hdr->render_height = hdr->height;
    }
}

static void parse_loop_filter_params(vp9_bitreader_t *br, vp9_frame_header_t *hdr)
{
    hdr->filter_level    = vp9_read_bits(br, 6);
    hdr->sharpness_level = vp9_read_bits(br, 3);
    if (vp9_read_bit(br)) {          /* loop_filter_delta_enabled */
        if (vp9_read_bit(br)) {      /* loop_filter_delta_update */
            for (int i = 0; i < 4; i++)
                if (vp9_read_bit(br)) read_inv_signed(br, 6);  /* ref deltas */
            for (int i = 0; i < 2; i++)
                if (vp9_read_bit(br)) read_inv_signed(br, 6);  /* mode deltas */
        }
    }
}

static void parse_quantization_params(vp9_bitreader_t *br, vp9_frame_header_t *hdr)
{
    hdr->base_qindex   = (int16_t)vp9_read_bits(br, 8);
    hdr->y_dc_delta_q  = (int8_t)read_delta_q(br);
    hdr->uv_dc_delta_q = (int8_t)read_delta_q(br);
    hdr->uv_ac_delta_q = (int8_t)read_delta_q(br);
}

static void parse_segmentation_params(vp9_bitreader_t *br, vp9_frame_header_t *hdr)
{
    static const int feature_bits[4]   = { 8, 6, 2, 0 };
    static const int feature_signed[4] = { 1, 1, 0, 0 };

    hdr->segmentation_enabled = vp9_read_bit(br);
    if (!hdr->segmentation_enabled) return;

    if (vp9_read_bit(br)) {          /* segmentation_update_map */
        for (int i = 0; i < 7; i++)
            if (vp9_read_bit(br)) vp9_read_bits(br, 8);  /* tree probs */
        if (vp9_read_bit(br)) {      /* segmentation_temporal_update */
            for (int i = 0; i < 3; i++)
                if (vp9_read_bit(br)) vp9_read_bits(br, 8);  /* pred probs */
        }
    }

    if (vp9_read_bit(br)) {          /* segmentation_update_data */
        vp9_read_bit(br);            /* segmentation_abs_or_delta_update */
        for (int seg = 0; seg < 8; seg++) {
            for (int f = 0; f < 4; f++) {
                if (vp9_read_bit(br)) {  /* feature_enabled */
                    if (feature_bits[f] > 0) vp9_read_bits(br, feature_bits[f]);
                    if (feature_signed[f]) vp9_read_bit(br);
                }
            }
        }
    }
}

static void parse_tile_info(vp9_bitreader_t *br, vp9_frame_header_t *hdr)
{
    int mi_cols = ((int)hdr->width + 7) >> 3;
    int sb64_cols = (mi_cols + 7) >> 3;

    /* MAX_TILE_WIDTH_B64 = 64, MIN_TILE_WIDTH_B64 = 4 */
    int min_log2 = 0;
    while ((64 << min_log2) < sb64_cols) min_log2++;
    int max_log2 = 1;
    while ((sb64_cols >> max_log2) >= 4) max_log2++;
    max_log2--;

    hdr->log2_tile_cols = min_log2;
    while (hdr->log2_tile_cols < max_log2) {
        if (!vp9_read_bit(br)) break;
        hdr->log2_tile_cols++;
    }
    hdr->log2_tile_rows = vp9_read_bit(br) ? 1 + vp9_read_bit(br) : 0;
}

int vp9_parse_frame_header(vp9_bitreader_t *br, vp9_frame_header_t *hdr,
                           uint32_t ref_width, uint32_t ref_height)
{
    memset(hdr, 0, sizeof(*hdr));

    /* Frame marker */
    if (vp9_read_bits(br, 2) != 0x2) return -1;

    hdr->profile = vp9_read_bit(br);
    hdr->profile |= vp9_read_bit(br) << 1;
    if (hdr->profile == 3 && vp9_read_bit(br)) return -1; /* reserved_zero */

    if (vp9_read_bit(br)) {  /* show_existing_frame */
        hdr->show_existing_frame = true;
        hdr->frame_to_show_map_idx = vp9_read_bits(br, 3);
        return 0;
    }

    hdr->frame_type  = vp9_read_bit(br);
    hdr->show_frame  = vp9_read_bit(br);
    hdr->error_resilient = vp9_read_bit(br);

    if (hdr->frame_type == VP9_FRAME_KEY) {
        for (int i = 0; i < 3; i++)
            if (vp9_read_bits(br, 8) != VP9_SYNC_CODE[i]) return -1;

        if (parse_color_config(br, hdr) != 0) return -1;
        parse_frame_size(br, hdr);
        parse_render_size(br, hdr);

        hdr->refresh_frame_flags = 0xFF;
        hdr->allow_high_precision_mv = false;
    } else {
        if (!hdr->show_frame)
            hdr->intra_only = vp9_read_bit(br);

        if (!hdr->error_resilient)
            hdr->reset_frame_context = vp9_read_bits(br, 2);

        if (hdr->intra_only) {
            for (int i = 0; i < 3; i++)
                if (vp9_read_bits(br, 8) != VP9_SYNC_CODE[i]) return -1;

            if (hdr->profile > 0) {
                if (parse_color_config(br, hdr) != 0) return -1;
            } else {
                hdr->bit_depth = 8;
                hdr->color_space = VP9_CS_BT601;
                hdr->subsampling_x = 1;
                hdr->subsampling_y = 1;
            }
            hdr->refresh_frame_flags = vp9_read_bits(br, 8);
            parse_frame_size(br, hdr);
            parse_render_size(br, hdr);
        } else {
            hdr->refresh_frame_flags = vp9_read_bits(br, 8);
            for (int i = 0; i < 3; i++) {
                hdr->ref_frame_idx[i] = vp9_read_bits(br, 3);
                hdr->ref_frame_sign_bias[i + 1] = vp9_read_bit(br);
            }

            /* frame_size_with_refs: inherit a ref's size or read it.
             * tile_info below depends on the width, so the caller-provided
             * reference dimensions are required to stay bit-aligned. */
            bool found_ref = false;
            for (int i = 0; i < 3; i++) {
                if (vp9_read_bit(br)) {
                    found_ref = true;
                    hdr->width = ref_width;
                    hdr->height = ref_height;
                    break;
                }
            }
            if (!found_ref) parse_frame_size(br, hdr);
            parse_render_size(br, hdr);

            hdr->allow_high_precision_mv = vp9_read_bit(br);

            /* read_interpolation_filter */
            static const vp9_interp_filter_t literal_to_filter[4] = {
                VP9_FILTER_EIGHTTAP_SMOOTH, VP9_FILTER_EIGHTTAP,
                VP9_FILTER_EIGHTTAP_SHARP, VP9_FILTER_BILINEAR
            };
            if (vp9_read_bit(br)) {
                hdr->interp_filter = VP9_FILTER_SWITCHABLE;
            } else {
                hdr->interp_filter = literal_to_filter[vp9_read_bits(br, 2)];
            }
        }
    }

    if (!hdr->error_resilient) {
        hdr->refresh_frame_context = vp9_read_bit(br);
        hdr->frame_parallel = vp9_read_bit(br);
    }
    hdr->frame_context_idx = vp9_read_bits(br, 2);

    parse_loop_filter_params(br, hdr);
    parse_quantization_params(br, hdr);
    parse_segmentation_params(br, hdr);
    parse_tile_info(br, hdr);

    hdr->first_partition_size = (uint16_t)vp9_read_bits(br, 16);
    if (hdr->first_partition_size == 0) return -1;

    /* Byte-align: the compressed header starts at the next byte boundary */
    hdr->uncompressed_header_bytes = (uint32_t)br->pos + (br->bit != 7 ? 1 : 0);

    if (br->pos > br->size) return -1;  /* ran past the end of the buffer */

    return 0;
}
