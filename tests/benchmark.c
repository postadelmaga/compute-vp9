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
#include "vp9_synth.h"

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

int main(int argc, char **argv)
{
    setbuf(stdout, NULL);

    int width = 1280;
    int height = 720;
    int duration_sec = 0;
    int log2_tile_cols = 0;

    for (int i = 1; i < argc; i++) {
        if ((strcmp(argv[i], "-w") == 0 || strcmp(argv[i], "--width") == 0) && i + 1 < argc) {
            width = atoi(argv[++i]);
        } else if ((strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--height") == 0) && i + 1 < argc) {
            height = atoi(argv[++i]);
        } else if ((strcmp(argv[i], "-d") == 0 || strcmp(argv[i], "--duration") == 0) && i + 1 < argc) {
            duration_sec = atoi(argv[++i]);
        } else if ((strcmp(argv[i], "-c") == 0 || strcmp(argv[i], "--tile-cols") == 0) && i + 1 < argc) {
            log2_tile_cols = atoi(argv[++i]);  /* log2: 1 = 2 tiles, 2 = 4 tiles */
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

    size_t synth_bufsize = vp9_synth_bufsize(width, height);
    uint8_t *keyframe_buf = malloc(synth_bufsize);
    uint8_t *interframe_buf = malloc(synth_bufsize);
    if (!keyframe_buf || !interframe_buf) {
        printf("ERROR: Out of memory allocating packet buffers\n");
        free(keyframe_buf);
        free(interframe_buf);
        return 1;
    }

    size_t keyframe_size = vp9_synth_keyframe(keyframe_buf, width, height, log2_tile_cols);
    size_t interframe_size = vp9_synth_interframe(interframe_buf, width, height, log2_tile_cols);
    if (log2_tile_cols > 0) {
        printf("Tile columns: %d (parsed in parallel)\n", 1 << log2_tile_cols);
    }

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
