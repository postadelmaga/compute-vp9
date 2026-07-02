/**
 * compute-vp9 — validate the uncompressed-header parser against a real
 * VP9 stream (IVF container, e.g. produced by ffmpeg/libvpx).
 *
 * Splits VP9 superframes, parses every sub-frame's uncompressed header and
 * checks structural invariants (first frame is a keyframe, sizes match the
 * IVF header, header offsets stay inside the packet).
 *
 * Usage: test_ivf_header <file.ivf>
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../src/decoder/vp9_bitstream.h"
#include "../src/decoder/vp9_entropy.h"

static uint32_t rd_le32(const uint8_t *p) {
    return p[0] | (p[1] << 8) | (p[2] << 16) | ((uint32_t)p[3] << 24);
}
static uint16_t rd_le16(const uint8_t *p) { return p[0] | (p[1] << 8); }

static int g_failures = 0;
#define CHECK(cond, ...) do { \
    if (!(cond)) { printf("  FAIL: " __VA_ARGS__); printf("\n"); g_failures++; } \
} while (0)

/* Split a (possibly) superframe packet into sub-frames; returns count */
static int split_superframe(const uint8_t *data, size_t size,
                            const uint8_t *frames[8], size_t sizes[8])
{
    if (size < 1) return 0;
    uint8_t marker = data[size - 1];
    if ((marker & 0xE0) == 0xC0) {
        int count = (marker & 0x7) + 1;
        int mag = ((marker >> 3) & 0x3) + 1;
        size_t index_size = 2 + (size_t)mag * count;
        if (size >= index_size && data[size - index_size] == marker) {
            const uint8_t *idx = data + size - index_size + 1;
            size_t off = 0;
            for (int i = 0; i < count; i++) {
                size_t fsz = 0;
                for (int b = 0; b < mag; b++) fsz |= (size_t)idx[i * mag + b] << (8 * b);
                if (off + fsz > size - index_size) return 0;
                frames[i] = data + off;
                sizes[i] = fsz;
                off += fsz;
            }
            return count;
        }
    }
    frames[0] = data;
    sizes[0] = size;
    return 1;
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "usage: %s <file.ivf>\n", argv[0]);
        return 2;
    }

    FILE *f = fopen(argv[1], "rb");
    if (!f) { perror(argv[1]); return 2; }

    uint8_t ivf[32];
    if (fread(ivf, 1, 32, f) != 32 || memcmp(ivf, "DKIF", 4) != 0 ||
        memcmp(ivf + 8, "VP90", 4) != 0) {
        fprintf(stderr, "not a VP9 IVF file\n");
        fclose(f);
        return 2;
    }
    uint32_t ivf_w = rd_le16(ivf + 12), ivf_h = rd_le16(ivf + 14);
    printf("IVF: %ux%u, %u frames declared\n", ivf_w, ivf_h, rd_le32(ivf + 24));

    uint8_t *buf = malloc(1 << 22);
    int packet_no = 0, frame_no = 0, keyframes = 0;
    uint32_t last_w = ivf_w, last_h = ivf_h;

    /* VP9 frame contexts, managed like the decoder does */
    static vp9_entropy_probs_t fctx[4], fprobs;
    for (int i = 0; i < 4; i++) vp9_entropy_probs_init(&fctx[i]);

    for (;;) {
        uint8_t fh[12];
        if (fread(fh, 1, 12, f) != 12) break;
        uint32_t psize = rd_le32(fh);
        if (psize > (1u << 22) || fread(buf, 1, psize, f) != psize) break;

        const uint8_t *frames[8];
        size_t sizes[8];
        int n = split_superframe(buf, psize, frames, sizes);
        CHECK(n > 0, "packet %d: bad superframe index", packet_no);

        for (int i = 0; i < n; i++) {
            vp9_bitreader_t br;
            vp9_frame_header_t hdr;
            vp9_bitreader_init(&br, frames[i], sizes[i]);
            int rc = vp9_parse_frame_header(&br, &hdr, last_w, last_h);

            CHECK(rc == 0, "frame %d: header parse failed", frame_no);
            if (rc != 0) { frame_no++; continue; }

            if (hdr.show_existing_frame) {
                printf("  frame %2d: show_existing idx=%d\n", frame_no,
                       hdr.frame_to_show_map_idx);
                frame_no++;
                continue;
            }

            if (frame_no == 0) {
                CHECK(hdr.frame_type == VP9_FRAME_KEY, "frame 0 not a keyframe");
                keyframes++;
            }
            if (hdr.frame_type == VP9_FRAME_KEY) {
                CHECK(hdr.width == ivf_w && hdr.height == ivf_h,
                      "frame %d: size %ux%u != IVF %ux%u",
                      frame_no, hdr.width, hdr.height, ivf_w, ivf_h);
            }
            if (hdr.width) { last_w = hdr.width; last_h = hdr.height; }
            CHECK(hdr.first_partition_size > 0, "frame %d: zero compressed header",
                  frame_no);
            CHECK(hdr.uncompressed_header_bytes + hdr.first_partition_size <= sizes[i],
                  "frame %d: header overruns packet (%u+%u > %zu)",
                  frame_no, hdr.uncompressed_header_bytes,
                  hdr.first_partition_size, sizes[i]);

            /* Structural cross-check: walk the tile size headers implied by
             * log2_tile_cols/rows — they must chain exactly to packet end */
            {
                size_t off = (size_t)hdr.uncompressed_header_bytes + hdr.first_partition_size;
                int tiles = (1 << hdr.log2_tile_cols) * (1 << hdr.log2_tile_rows);
                int ok = 1;
                for (int t = 0; t < tiles - 1 && ok; t++) {
                    if (off + 4 > sizes[i]) { ok = 0; break; }
                    uint32_t tsz = ((uint32_t)frames[i][off] << 24) |
                                   ((uint32_t)frames[i][off + 1] << 16) |
                                   ((uint32_t)frames[i][off + 2] << 8) |
                                   frames[i][off + 3];
                    off += 4 + tsz;
                    if (off > sizes[i]) ok = 0;
                }
                CHECK(ok && off <= sizes[i],
                      "frame %d: tile sizes inconsistent with 2^%u cols",
                      frame_no, hdr.log2_tile_cols);
            }

            /* Compressed header: select/reset context, parse the updates */
            if (hdr.frame_type == VP9_FRAME_KEY || hdr.error_resilient ||
                hdr.reset_frame_context == 3) {
                for (int c = 0; c < 4; c++) vp9_entropy_probs_init(&fctx[c]);
            } else if (hdr.reset_frame_context == 2) {
                vp9_entropy_probs_init(&fctx[hdr.frame_context_idx & 3]);
            }
            fprobs = fctx[hdr.frame_context_idx & 3];
            int chdr_rc = vp9_parse_compressed_header(
                &hdr, frames[i] + hdr.uncompressed_header_bytes,
                hdr.first_partition_size, &fprobs);
            CHECK(chdr_rc == 0, "frame %d: compressed header parse failed", frame_no);
            if (hdr.refresh_frame_context) {
                fctx[hdr.frame_context_idx & 3] = fprobs;
            }

            printf("  frame %2d: %s%s %ux%u q=%d lf=%u tiles=2^%u refs=[%u,%u,%u] "
                   "refresh=0x%02X hdr=%u+%u/%zuB tx_mode=%d refmode=%d chdr=%s\n",
                   frame_no,
                   hdr.frame_type == VP9_FRAME_KEY ? "KEY" : "INTER",
                   hdr.show_frame ? "" : "(hidden)",
                   hdr.width, hdr.height, hdr.base_qindex, hdr.filter_level,
                   hdr.log2_tile_cols,
                   hdr.ref_frame_idx[0], hdr.ref_frame_idx[1], hdr.ref_frame_idx[2],
                   hdr.refresh_frame_flags,
                   hdr.uncompressed_header_bytes, hdr.first_partition_size, sizes[i],
                   fprobs.tx_mode, fprobs.reference_mode,
                   chdr_rc == 0 ? "ok" : "FAIL");
            frame_no++;
        }
        packet_no++;
    }

    free(buf);
    fclose(f);

    printf("%d packets, %d frames parsed, %d failures\n",
           packet_no, frame_no, g_failures);
    if (frame_no == 0) g_failures++;
    return g_failures ? 1 : 0;
}
