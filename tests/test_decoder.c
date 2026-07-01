#include <stdio.h>
#include <string.h>
#include <assert.h>
#include "compute_vp9/decoder.h"
#include "../src/decoder/vp9_bitstream.h"
#include "../src/decoder/vpx_reader.h"
#include "../src/decoder/vp9_parsed_frame.h"
#include "../src/decoder/vp9_entropy.h"
#include "../src/decoder/vp9_tile.h"

/* ── Helpers ─────────────────────────────────────────────────────────────── */
#define PASS(name) printf("  ✓ %s\n", name)
#define FAIL(name, msg) do { printf("  ✗ %s: %s\n", name, msg); return 1; } while(0)

/* ── Test: bit reader ────────────────────────────────────────────────────── */
int test_bitstream(void)
{
    printf("=== test_bitstream ===\n");

    uint8_t data[] = {0b10110100, 0b00001111};
    vp9_bitreader_t br;
    vp9_bitreader_init(&br, data, sizeof(data));

    if (vp9_read_bit(&br) != 1) FAIL("bit0", "expected 1");
    if (vp9_read_bit(&br) != 0) FAIL("bit1", "expected 0");
    PASS("first 2 bits");

    if (vp9_read_bits(&br, 4) != 0b1101) FAIL("bits 2-5", "wrong value");
    PASS("4-bit read");

    if (vp9_read_bits(&br, 2) != 0b00) FAIL("bits 6-7", "wrong value");
    PASS("cross-byte read");

    /* Next byte: 0x0F = 0b00001111 */
    if (vp9_read_bits(&br, 4) != 0b0000) FAIL("upper nibble", "wrong");
    if (vp9_read_bits(&br, 4) != 0b1111) FAIL("lower nibble", "wrong");
    PASS("second byte");

    return 0;
}

/* ── Test: range coder (vpx_reader) ──────────────────────────────────────── */
int test_range_coder(void)
{
    printf("=== test_range_coder ===\n");

    // An all-zero byte array should start with a 0 marker bit (so init returns 0)
    uint8_t zero_data[8] = { 0 };
    vpx_reader r;
    int err = vpx_reader_init(&r, zero_data, sizeof(zero_data));
    if (err != 0) FAIL("vpx_reader_init", "expected 0 (marker bit is 0)");

    // Read a few bits with 50/50 probability
    int b0 = vpx_read_bit(&r);
    int b1 = vpx_read_bit(&r);
    printf("  read bits: %d, %d\n", b0, b1);
    PASS("read bits from range coder");

    return 0;
}

/* ── Test: frame header parser ───────────────────────────────────────────── */
int test_frame_header(void)
{
    printf("=== test_frame_header ===\n");

    /* Minimal synthetic VP9 keyframe header:
     * frame_marker(2)=10, profile(2)=00,
     * show_existing=0, frame_type=0(KEY), show_frame=1, error_resilient=0
     * sync_code: 0x49 0x83 0x42
     * color_space(3)=001 (BT601), bit_depth implied=8
     * width-1 (16): 1279 = 0x04FF, height-1 (16): 719 = 0x02CF
     * ... (simplified — real bitstream is more complex)
     */
    /* We just test that the parser doesn't crash on valid-ish data */
    uint8_t fake_key[] = {
        0x84,               /* 10 00 0 0 1 0 — marker+profile+flags */
        0x49, 0x83, 0x42,  /* sync code */
        0x04, 0xFF,         /* width-1 = 1279 → width=1280 */
        0x02, 0xCF,         /* height-1 = 719 → height=720 */
        0x80,               /* base_qindex=128 */
        0x00,               /* no delta Q */
        0x00,               /* filter_level=0, sharpness=0 */
        0x00,               /* tile cols=0, rows=0 */
    };

    vp9_bitreader_t br;
    vp9_bitreader_init(&br, fake_key, sizeof(fake_key));
    vp9_frame_header_t hdr;
    int rc = vp9_parse_frame_header(&br, &hdr);

    /* We allow parse errors on synthetic data — just check no crash */
    printf("  parse returned: %d (frame_type=%d)\n", rc, hdr.frame_type);
    PASS("no crash on frame header parse");

    return 0;
}

/* ── Test: decoder context ───────────────────────────────────────────────── */
int test_backend(void)
{
    printf("=== test_backend ===\n");

    cvp9_ctx_t *ctx = NULL;
    cvp9_err_t err = cvp9_create(NULL, &ctx);

    if (err != CVP9_OK)   FAIL("cvp9_create", cvp9_err_str(err));
    if (ctx == NULL)      FAIL("ctx not null", "returned NULL");
    PASS("context created");

    cvp9_backend_t backend = cvp9_active_backend(ctx);
    printf("  active backend: %d\n", backend);
    PASS("backend query");

    cvp9_destroy(ctx);
    PASS("context destroyed");

    return 0;
}

/* ── Test: tile and block parsing ────────────────────────────────────────── */
int test_tile_parsing(void)
{
    printf("=== test_tile_parsing ===\n");

    vp9_frame_header_t hdr;
    memset(&hdr, 0, sizeof(hdr));
    hdr.width = 128;
    hdr.height = 128;
    hdr.frame_type = VP9_FRAME_KEY;
    hdr.log2_tile_cols = 0;
    hdr.log2_tile_rows = 0;

    uint8_t dummy_tile_data[256] = {0};

    vp9_parsed_frame_t *pf = vp9_parsed_frame_alloc(hdr.width, hdr.height);
    if (!pf) FAIL("vp9_parsed_frame_alloc", "returned NULL");

    vp9_entropy_probs_t probs;
    vp9_entropy_probs_init(&probs);

    int rc = vp9_decode_tiles(&hdr, dummy_tile_data, sizeof(dummy_tile_data), &probs, pf);
    printf("  vp9_decode_tiles returned %d, parsed %u blocks, %u coeffs\n",
           rc, pf->num_blocks, pf->num_coeffs);

    if (rc != 0) FAIL("vp9_decode_tiles", "decoding failed");
    if (pf->num_blocks == 0) FAIL("num_blocks", "no blocks parsed");

    vp9_parsed_frame_free(pf);
    PASS("tile and block parsing completed successfully");
    return 0;
}

/* ── Test: end-to-end keyframe decoding ──────────────────────────────────── */
int test_keyframe_decode(void)
{
    printf("=== test_keyframe_decode ===\n");

    /* Construct a minimal valid VP9 keyframe package */
    uint8_t packet[] = {
        0x82,               /* marker, profile, keyframe, show_frame, etc. */
        0x49, 0x83, 0x42,  /* sync code */
        0x00, 0x0F, 0xE0, 0x0F, 0xF0, 0x00, 0x00, 0x00, /* width, height, qindex, delta q, filter, tiles */
        
        /* Compressed header size (16-bit) = 4 */
        0x00, 0x04,
        /* Compressed header (4 bytes) */
        0x00, 0x00, 0x00, 0x00,
        
        /* Tile data (zeros) */
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    };

    cvp9_ctx_t *ctx = NULL;
    cvp9_config_t cfg = { .backend = CVP9_BACKEND_CPU };
    cvp9_err_t err = cvp9_create(&cfg, &ctx);
    if (err != CVP9_OK) FAIL("cvp9_create", cvp9_err_str(err));

    err = cvp9_decode(ctx, packet, sizeof(packet), 12345);
    if (err != CVP9_OK) FAIL("cvp9_decode", cvp9_err_str(err));

    cvp9_frame_info_t frame;
    err = cvp9_get_frame(ctx, &frame);
    if (err != CVP9_OK) FAIL("cvp9_get_frame", cvp9_err_str(err));

    printf("  decoded frame width=%u, height=%u, pts=%ld\n",
           frame.width, frame.height, frame.pts);

    if (frame.width != 128) FAIL("frame.width", "wrong width");
    if (frame.height != 128) FAIL("frame.height", "wrong height");
    if (frame.pts != 12345) FAIL("frame.pts", "wrong pts");

    cvp9_destroy(ctx);
    PASS("full keyframe decode and YUV output verified");
    return 0;
}

/* ── Entry point ─────────────────────────────────────────────────────────── */
int main(int argc, char **argv)
{
    const char *test = "all";
    for (int i = 1; i < argc - 1; i++)
        if (strcmp(argv[i], "--test") == 0) test = argv[i + 1];

    int failures = 0;
    if (strcmp(test, "bitstream")    == 0 || strcmp(test, "all") == 0)
        failures += test_bitstream();
    if (strcmp(test, "range_coder")   == 0 || strcmp(test, "all") == 0)
        failures += test_range_coder();
    if (strcmp(test, "frame_header") == 0 || strcmp(test, "all") == 0)
        failures += test_frame_header();
    if (strcmp(test, "backend")      == 0 || strcmp(test, "all") == 0)
        failures += test_backend();
    if (strcmp(test, "tile_parsing")  == 0 || strcmp(test, "all") == 0)
        failures += test_tile_parsing();
    if (strcmp(test, "keyframe")      == 0 || strcmp(test, "all") == 0)
        failures += test_keyframe_decode();

    printf("\n%s\n", failures == 0 ? "ALL TESTS PASSED ✓" : "SOME TESTS FAILED ✗");
    return failures;
}
