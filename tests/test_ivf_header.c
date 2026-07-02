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
#include "../src/decoder/vp9_tile.h"

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
    static vp9_counts_t counts;
    for (int i = 0; i < 4; i++) vp9_entropy_probs_init(&fctx[i]);

    /* Previous-frame MV state */
    int8_t *prev_ref0 = NULL, *prev_ref1 = NULL, *cur_ref0 = NULL, *cur_ref1 = NULL;
    int16_t *prev_mvs = NULL, *cur_mvs = NULL;
    size_t mi_alloc = 0;
    int last_show = 0, last_key = 1, last_intra_only = 0;
    uint32_t prev_w = 0, prev_h = 0;

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

            /* Conformant entropy decode of every frame's tiles: the bool
             * decoder must consume ~exactly each tile and cover the MI grid.
             * Frames 0-1 are strict (no backward adaptation involved yet);
             * later frames report drift until adaptation is implemented. */
            int strict = 1;
            if (!hdr.segmentation_enabled && hdr.width > 0) {
                vp9_parsed_frame_t *tpf = vp9_parsed_frame_alloc(hdr.width, hdr.height);
                if (tpf) {
                    memcpy(&tpf->hdr, &hdr, sizeof(hdr));
                    size_t mi_count = (size_t)tpf->mi_grid_width * tpf->mi_grid_height;
                    if (mi_count > mi_alloc) {
                        free(prev_ref0); free(prev_ref1); free(prev_mvs);
                        free(cur_ref0); free(cur_ref1); free(cur_mvs);
                        prev_ref0 = calloc(mi_count, 1);
                        prev_ref1 = calloc(mi_count, 1);
                        prev_mvs = calloc(mi_count * 4, sizeof(int16_t));
                        cur_ref0 = calloc(mi_count, 1);
                        cur_ref1 = calloc(mi_count, 1);
                        cur_mvs = calloc(mi_count * 4, sizeof(int16_t));
                        mi_alloc = mi_count;
                    }
                    int use_prev = !hdr.error_resilient && hdr.width == prev_w &&
                                   hdr.height == prev_h && !last_intra_only &&
                                   last_show && !last_key;
                    memset(&counts, 0, sizeof(counts));
                    int mi_cols = ((int)hdr.width + 7) >> 3;
                    int mi_rows = ((int)hdr.height + 7) >> 3;
                    int sb_cols = (mi_cols + 7) >> 3, sb_rows = (mi_rows + 7) >> 3;
                    int tcols = 1 << hdr.log2_tile_cols, trows = 1 << hdr.log2_tile_rows;
                    size_t off = (size_t)hdr.uncompressed_header_bytes + hdr.first_partition_size;
                    int rc_all = 0;

                    for (int tr = 0; tr < trows && !rc_all; tr++) {
                        int r0 = ((tr * sb_rows) >> hdr.log2_tile_rows) << 3;
                        int r1 = (((tr + 1) * sb_rows) >> hdr.log2_tile_rows) << 3;
                        if (r1 > mi_rows) r1 = mi_rows;
                        for (int tc = 0; tc < tcols && !rc_all; tc++) {
                            int c0 = ((tc * sb_cols) >> hdr.log2_tile_cols) << 3;
                            int c1 = (((tc + 1) * sb_cols) >> hdr.log2_tile_cols) << 3;
                            if (c1 > mi_cols) c1 = mi_cols;

                            size_t tsize;
                            if (tr == trows - 1 && tc == tcols - 1) {
                                tsize = sizes[i] - off;
                            } else {
                                tsize = ((size_t)frames[i][off] << 24) |
                                        ((size_t)frames[i][off + 1] << 16) |
                                        ((size_t)frames[i][off + 2] << 8) |
                                        frames[i][off + 3];
                                off += 4;
                            }

                            vpx_reader rd;
                            if (vpx_reader_init(&rd, frames[i] + off, tsize)) {
                                rc_all = -1;
                                break;
                            }
                            int trc = vp9_decode_tile_conformant(&rd, r0, r1, c0, c1,
                                    &fprobs, tpf, &counts,
                                    prev_ref0, prev_ref1, prev_mvs, use_prev,
                                    cur_ref0, cur_ref1, cur_mvs);
                            size_t consumed =
                                (size_t)(vpx_reader_find_end(&rd) - (frames[i] + off));
                            if (strict) {
                                CHECK(trc == 0, "frame %d tile %d,%d: decode failed",
                                      frame_no, tr, tc);
                                CHECK(consumed <= tsize && consumed + 8 >= tsize,
                                      "frame %d tile %d,%d: consumed %zu of %zu bytes",
                                      frame_no, tr, tc, consumed, tsize);
                            } else if (trc != 0 || consumed > tsize || consumed + 8 < tsize) {
                                printf("           frame %d tile %d,%d: drift (rc=%d, %zu of %zu bytes)\n",
                                       frame_no, tr, tc, trc, consumed, tsize);
                            }
                            if (trc) rc_all = -1;
                            off += tsize;
                        }
                    }

                    /* Backward adaptation, then context refresh (libvpx order) */
                    if (!rc_all && !hdr.error_resilient && !hdr.frame_parallel) {
                        int intra_only_f = hdr.frame_type == VP9_FRAME_KEY || hdr.intra_only;
                        vp9_adapt_probs(&fprobs, &fctx[hdr.frame_context_idx & 3],
                                        &counts, intra_only_f, last_key,
                                        hdr.allow_high_precision_mv,
                                        hdr.interp_filter == VP9_FILTER_SWITCHABLE);
                    }
                    if (hdr.refresh_frame_context) {
                        fctx[hdr.frame_context_idx & 3] = fprobs;
                    }

                    /* Swap prev-frame MV state */
                    { int8_t *t0 = prev_ref0; prev_ref0 = cur_ref0; cur_ref0 = t0; }
                    { int8_t *t1 = prev_ref1; prev_ref1 = cur_ref1; cur_ref1 = t1; }
                    { int16_t *tm = prev_mvs; prev_mvs = cur_mvs; cur_mvs = tm; }
                    prev_w = hdr.width; prev_h = hdr.height;
                    last_show = hdr.show_frame;
                    last_key = hdr.frame_type == VP9_FRAME_KEY;
                    last_intra_only = hdr.intra_only;

                    if (!rc_all) {
                        uint32_t covered = 0;
                        for (uint32_t m = 0; m < tpf->mi_grid_width * tpf->mi_grid_height; m++)
                            covered += tpf->mi_block_grid[m] != 0;
                        if (strict)
                            CHECK(covered == tpf->mi_grid_width * tpf->mi_grid_height,
                                  "frame %d: MI coverage %u/%u", frame_no, covered,
                                  tpf->mi_grid_width * tpf->mi_grid_height);
                        printf("           tile decode: %u blocks, %u coeffs, MI %u/%u\n",
                               tpf->num_blocks, tpf->num_coeffs, covered,
                               tpf->mi_grid_width * tpf->mi_grid_height);
                    }
                    vp9_parsed_frame_free(tpf);
                }
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
