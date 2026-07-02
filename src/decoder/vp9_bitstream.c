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

/* ── Frame header parser ────────────────────────────────────────────────── */

static const uint8_t VP9_SYNC_CODE[3] = {0x49, 0x83, 0x42};

int vp9_parse_frame_header(vp9_bitreader_t *br, vp9_frame_header_t *hdr)
{
    memset(hdr, 0, sizeof(*hdr));

    /* Frame marker */
    if (vp9_read_bits(br, 2) != 0x2) return -1;

    hdr->profile = vp9_read_bit(br);
    hdr->profile |= vp9_read_bit(br) << 1;
    if (hdr->profile == 3) vp9_read_bit(br); /* reserved_zero */

    /* Show existing frame */
    if (vp9_read_bit(br)) return 0; /* show_existing_frame */

    hdr->frame_type  = vp9_read_bit(br);
    hdr->show_frame  = vp9_read_bit(br);
    hdr->error_resilient = vp9_read_bit(br);

    if (hdr->frame_type == VP9_FRAME_KEY) {
        /* Sync code */
        for (int i = 0; i < 3; i++)
            if (vp9_read_bits(br, 8) != VP9_SYNC_CODE[i]) return -1;

        /* Color config */
        if (hdr->profile >= 2) {
            hdr->bit_depth = vp9_read_bit(br) ? 12 : 10;
        } else {
            hdr->bit_depth = 8;
        }
        hdr->color_space = vp9_read_bits(br, 3);

        /* Frame size */
        hdr->width  = vp9_read_bits(br, 16) + 1;
        hdr->height = vp9_read_bits(br, 16) + 1;

        /* Keyframes reset the whole reference pool */
        hdr->refresh_frame_flags = 0xFF;
    } else {
        /* Reference management */
        hdr->refresh_frame_flags = vp9_read_bits(br, 8);
        for (int i = 0; i < 3; i++)
            hdr->ref_frame_idx[i] = vp9_read_bits(br, 3);
    }

    /* Quantization */
    hdr->base_qindex    = (int16_t)vp9_read_bits(br, 8);
    hdr->y_dc_delta_q   = vp9_read_bit(br) ? (int8_t)vp9_read_bits(br, 5) : 0;
    hdr->uv_dc_delta_q  = vp9_read_bit(br) ? (int8_t)vp9_read_bits(br, 5) : 0;
    hdr->uv_ac_delta_q  = vp9_read_bit(br) ? (int8_t)vp9_read_bits(br, 5) : 0;

    /* Loop filter */
    hdr->filter_level    = vp9_read_bits(br, 6);
    hdr->sharpness_level = vp9_read_bits(br, 3);

    /* Tile info */
    hdr->log2_tile_cols = 0;
    while (vp9_read_bit(br)) hdr->log2_tile_cols++;
    hdr->log2_tile_rows = vp9_read_bit(br) ? (vp9_read_bit(br) ? 2 : 1) : 0;

    return 0;
}
