/**
 * compute-vp9 — decoding performance benchmark
 *
 * Generates a synthetic VP9-like frame sequence and compares the decoding and
 * reconstruction performance of the CPU fallback and Vulkan compute backend.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "compute_vp9/decoder.h"
#include "../src/decoder/vp9_bitstream.h"
#include "../src/decoder/vp9_parsed_frame.h"

#define BENCHMARK_FRAMES 100
#define FRAME_WIDTH 1280
#define FRAME_HEIGHT 720

/* Helper: get time in milliseconds */
static double get_time_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1000000.0;
}

typedef struct {
    uint8_t *buf;
    size_t pos;
    int bit;
} bitwriter_t;

static void write_bit(bitwriter_t *bw, int bit) {
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

static void write_bits(bitwriter_t *bw, uint32_t val, int n) {
    for (int i = n - 1; i >= 0; i--) {
        write_bit(bw, (val >> i) & 1);
    }
}

static void align_byte(bitwriter_t *bw) {
    if (bw->bit < 7) {
        bw->bit = 7;
        bw->pos++;
    }
}

/* Generates a synthetic VP9 keyframe packet matching our parser's dialect */
static size_t generate_synthetic_keyframe(uint8_t *buf, uint32_t width, uint32_t height)
{
    memset(buf, 0, 1024);
    bitwriter_t bw = { .buf = buf, .pos = 0, .bit = 7 };

    write_bits(&bw, 2, 2); /* frame_marker */
    write_bits(&bw, 0, 2); /* profile */
    write_bit(&bw, 0);     /* show_existing_frame */
    write_bit(&bw, 0);     /* frame_type (KEY) */
    write_bit(&bw, 1);     /* show_frame */
    write_bit(&bw, 0);     /* error_resilient */

    write_bits(&bw, 0x498342, 24); /* sync_code */
    write_bits(&bw, 1, 3);          /* color_space = BT601 */

    write_bits(&bw, width - 1, 16);
    write_bits(&bw, height - 1, 16);

    write_bits(&bw, 128, 8); /* base_qindex */
    write_bit(&bw, 0);       /* y_dc_delta_q */
    write_bit(&bw, 0);       /* uv_dc_delta_q */
    write_bit(&bw, 0);       /* uv_ac_delta_q */

    write_bits(&bw, 32, 6);  /* filter_level */
    write_bits(&bw, 0, 3);   /* sharpness_level */

    write_bit(&bw, 0);       /* log2_tile_cols */
    write_bit(&bw, 0);       /* log2_tile_rows */

    align_byte(&bw);

    /* Write 16-bit compressed header size (big-endian) */
    size_t size_pos = bw.pos;
    buf[size_pos] = 0x00;
    buf[size_pos + 1] = 0x04;
    bw.pos += 2;

    /* Write compressed header payload (4 bytes of zeros) */
    buf[bw.pos++] = 0x00;
    buf[bw.pos++] = 0x00;
    buf[bw.pos++] = 0x00;
    buf[bw.pos++] = 0x00;

    /* Write tile data (64 bytes of zeros) */
    memset(buf + bw.pos, 0, 64);
    bw.pos += 64;

    return bw.pos;
}

/* Generates a synthetic VP9 interframe packet */
static size_t generate_synthetic_interframe(uint8_t *buf, uint32_t width, uint32_t height)
{
    memset(buf, 0, 1024);
    bitwriter_t bw = { .buf = buf, .pos = 0, .bit = 7 };

    write_bits(&bw, 2, 2); /* frame_marker */
    write_bits(&bw, 0, 2); /* profile */
    write_bit(&bw, 0);     /* show_existing_frame */
    write_bit(&bw, 1);     /* frame_type (NON-KEY) */
    write_bit(&bw, 1);     /* show_frame */
    write_bit(&bw, 0);     /* error_resilient */

    write_bits(&bw, 128, 8); /* base_qindex */
    write_bit(&bw, 0);       /* y_dc_delta_q */
    write_bit(&bw, 0);       /* uv_dc_delta_q */
    write_bit(&bw, 0);       /* uv_ac_delta_q */

    write_bits(&bw, 32, 6);  /* filter_level */
    write_bits(&bw, 0, 3);   /* sharpness_level */

    write_bit(&bw, 0);       /* log2_tile_cols */
    write_bit(&bw, 0);       /* log2_tile_rows */

    align_byte(&bw);

    /* Write 16-bit compressed header size (big-endian) */
    size_t size_pos = bw.pos;
    buf[size_pos] = 0x00;
    buf[size_pos + 1] = 0x04;
    bw.pos += 2;

    /* Write compressed header payload (4 bytes of zeros) */
    buf[bw.pos++] = 0x00;
    buf[bw.pos++] = 0x00;
    buf[bw.pos++] = 0x00;
    buf[bw.pos++] = 0x00;

    /* Write tile data (64 bytes of zeros) */
    memset(buf + bw.pos, 0, 64);
    bw.pos += 64;

    return bw.pos;
}

int main(int argc, char **argv)
{
    setbuf(stdout, NULL);

    int width = 1280;
    int height = 720;
    int duration_sec = 0;

    for (int i = 1; i < argc; i++) {
        if ((strcmp(argv[i], "-w") == 0 || strcmp(argv[i], "--width") == 0) && i + 1 < argc) {
            width = atoi(argv[++i]);
        } else if ((strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--height") == 0) && i + 1 < argc) {
            height = atoi(argv[++i]);
        } else if ((strcmp(argv[i], "-d") == 0 || strcmp(argv[i], "--duration") == 0) && i + 1 < argc) {
            duration_sec = atoi(argv[++i]);
        }
    }

    printf("=================================================================\n");
    printf("         compute-vp9 Decoding Performance Benchmark             \n");
    printf("=================================================================\n");
    if (duration_sec > 0) {
        printf("Resolution: %dx%d | Duration: %d seconds\n\n", width, height, duration_sec);
    } else {
        printf("Resolution: %dx%d | Frames: %d\n\n", width, height, BENCHMARK_FRAMES);
    }

    uint8_t *keyframe_buf = malloc(1024);
    uint8_t *interframe_buf = malloc(1024);
    if (!keyframe_buf || !interframe_buf) {
        printf("ERROR: Out of memory allocating packet buffers\n");
        free(keyframe_buf);
        free(interframe_buf);
        return 1;
    }

    size_t keyframe_size = generate_synthetic_keyframe(keyframe_buf, width, height);
    size_t interframe_size = generate_synthetic_interframe(interframe_buf, width, height);

    /* ─── Test 1: CPU Fallback Backend ─────────────────────────────────────── */
    printf("Running CPU Software Fallback Backend...\n");
    cvp9_ctx_t *cpu_ctx = NULL;
    cvp9_config_t cpu_cfg = { .backend = CVP9_BACKEND_CPU };
    if (cvp9_create(&cpu_cfg, &cpu_ctx) != CVP9_OK) {
        printf("ERROR: Failed to create CPU decoder context\n");
        free(keyframe_buf);
        free(interframe_buf);
        return 1;
    }

    double cpu_start = get_time_ms();
    int cpu_decoded_count = 0;
    int f = 0;

    if (duration_sec > 0) {
        double dur_ms = duration_sec * 1000.0;
        while ((get_time_ms() - cpu_start) < dur_ms) {
            cvp9_err_t err;
            if (f == 0) {
                err = cvp9_decode(cpu_ctx, keyframe_buf, keyframe_size, f);
            } else {
                err = cvp9_decode(cpu_ctx, interframe_buf, interframe_size, f);
            }

            if (err == CVP9_OK) {
                cvp9_frame_info_t out_frame;
                if (cvp9_get_frame(cpu_ctx, &out_frame) == CVP9_OK) {
                    cpu_decoded_count++;
                }
            }
            f++;
        }
    } else {
        for (f = 0; f < BENCHMARK_FRAMES; f++) {
            cvp9_err_t err;
            if (f == 0) {
                err = cvp9_decode(cpu_ctx, keyframe_buf, keyframe_size, f);
            } else {
                err = cvp9_decode(cpu_ctx, interframe_buf, interframe_size, f);
            }

            if (err == CVP9_OK) {
                cvp9_frame_info_t out_frame;
                if (cvp9_get_frame(cpu_ctx, &out_frame) == CVP9_OK) {
                    cpu_decoded_count++;
                }
            }
        }
    }
    double cpu_end = get_time_ms();
    double cpu_time = cpu_end - cpu_start;
    double cpu_fps = (double)cpu_decoded_count / (cpu_time / 1000.0);

    printf("  Decoded frames: %d\n", cpu_decoded_count);
    printf("  Time taken:     %.2f ms\n", cpu_time);
    printf("  Throughput:     %.2f FPS\n\n", cpu_fps);

    cvp9_destroy(cpu_ctx);

    /* ─── Test 2: Vulkan Compute Backend ───────────────────────────────────── */
    printf("Running Vulkan Compute GPU Backend...\n");
    cvp9_ctx_t *vk_ctx = NULL;
    cvp9_config_t vk_cfg = { .backend = CVP9_BACKEND_VULKAN };
    if (cvp9_create(&vk_cfg, &vk_ctx) != CVP9_OK) {
        printf("ERROR: Failed to create Vulkan decoder context\n");
        free(keyframe_buf);
        free(interframe_buf);
        return 1;
    }

    if (cvp9_active_backend(vk_ctx) != CVP9_BACKEND_VULKAN) {
        printf("WARNING: Vulkan backend not active, skipping Vulkan benchmark\n\n");
        cvp9_destroy(vk_ctx);
        free(keyframe_buf);
        free(interframe_buf);
        return 0;
    }

    double vk_start = get_time_ms();
    int vk_decoded_count = 0;
    f = 0;
    
    double vk_max_frame_ms = 0;
    int vk_frame_drops_60fps = 0;

    /* Note: the Vulkan backend pipelines up to a few frames in flight —
     * cvp9_get_frame returns CVP9_ERR_AGAIN while the oldest frame is still
     * on the GPU, so we count delivered frames and drain the pipeline with
     * cvp9_get_frame_sync at the end. */
    if (duration_sec > 0) {
        double dur_ms = duration_sec * 1000.0;
        while ((get_time_ms() - vk_start) < dur_ms) {
            double frame_start_time = get_time_ms();
            cvp9_err_t err;
            if (f == 0) {
                err = cvp9_decode(vk_ctx, keyframe_buf, keyframe_size, f);
            } else {
                err = cvp9_decode(vk_ctx, interframe_buf, interframe_size, f);
            }

            if (err == CVP9_OK) {
                cvp9_frame_info_t out_frame;
                cvp9_err_t get_err = cvp9_get_frame(vk_ctx, &out_frame);
                if (get_err == CVP9_OK) {
                    vk_decoded_count++;
                } else if (get_err != CVP9_ERR_AGAIN) {
                    printf("[benchmark] cvp9_get_frame failed at frame %d: %s\n", f, cvp9_err_str(get_err));
                }
            } else {
                printf("[benchmark] cvp9_decode failed at frame %d: %s\n", f, cvp9_err_str(err));
            }
            double frame_end_time = get_time_ms();
            double f_time = frame_end_time - frame_start_time;
            if (f_time > vk_max_frame_ms) vk_max_frame_ms = f_time;
            if (f_time > 16.666) vk_frame_drops_60fps++;
            f++;
        }
    } else {
        for (f = 0; f < BENCHMARK_FRAMES; f++) {
            double frame_start_time = get_time_ms();
            cvp9_err_t err;
            if (f == 0) {
                err = cvp9_decode(vk_ctx, keyframe_buf, keyframe_size, f);
            } else {
                err = cvp9_decode(vk_ctx, interframe_buf, interframe_size, f);
            }

            if (err == CVP9_OK) {
                cvp9_frame_info_t out_frame;
                cvp9_err_t get_err = cvp9_get_frame(vk_ctx, &out_frame);
                if (get_err == CVP9_OK) {
                    vk_decoded_count++;
                } else if (get_err != CVP9_ERR_AGAIN) {
                    printf("[benchmark] cvp9_get_frame failed at frame %d: %s\n", f, cvp9_err_str(get_err));
                }
            } else {
                printf("[benchmark] cvp9_decode failed at frame %d: %s\n", f, cvp9_err_str(err));
            }
            double frame_end_time = get_time_ms();
            double f_time = frame_end_time - frame_start_time;
            if (f_time > vk_max_frame_ms) vk_max_frame_ms = f_time;
            if (f_time > 16.666) vk_frame_drops_60fps++;
        }
    }

    /* Drain frames still in flight in the GPU pipeline */
    {
        cvp9_frame_info_t out_frame;
        while (cvp9_get_frame_sync(vk_ctx, &out_frame) == CVP9_OK) {
            vk_decoded_count++;
        }
    }
    double vk_end = get_time_ms();
    double vk_time = vk_end - vk_start;
    double vk_fps = (double)vk_decoded_count / (vk_time / 1000.0);

    printf("  Decoded frames: %d\n", vk_decoded_count);
    printf("  Time taken:     %.2f ms\n", vk_time);
    printf("  Throughput:     %.2f FPS\n", vk_fps);
    printf("  Max latency:    %.2f ms/frame\n", vk_max_frame_ms);
    printf("  Frame drops:    %d (slower than 60fps limit)\n\n", vk_frame_drops_60fps);

    cvp9_destroy(vk_ctx);
    free(keyframe_buf);
    free(interframe_buf);

    /* ─── Summary comparison ───────────────────────────────────────────────── */
    printf("=================================================================\n");
    printf("                     Benchmark Results Summary                   \n");
    printf("=================================================================\n");
    printf("CPU Fallback Backend:   %.2f FPS (1.0x)\n", cpu_fps);
    printf("Vulkan GPU Backend:     %.2f FPS (%.2fx speedup)\n", vk_fps, vk_fps / cpu_fps);
    printf("=================================================================\n");

    return 0;
}
