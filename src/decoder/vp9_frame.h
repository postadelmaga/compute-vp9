/**
 * compute-vp9 — frame buffer allocation and reference management
 */
#pragma once

#include "compute_vp9/decoder.h"
#include <stdbool.h>

/**
 * Allocates YUV 4:2:0 plane buffers for a frame of given dimensions.
 */
bool cvp9_frame_alloc(cvp9_frame_info_t *f, uint32_t width, uint32_t height);

/**
 * Releases memory allocated for a frame's planes.
 */
void cvp9_frame_free(cvp9_frame_info_t *f);

/**
 * Deep copies frame content from src to dst.
 */
bool cvp9_frame_copy(cvp9_frame_info_t *dst, const cvp9_frame_info_t *src);
