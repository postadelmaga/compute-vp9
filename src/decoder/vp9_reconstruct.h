/**
 * compute-vp9 — CPU reconstruction engine (Intra, Inter, IDCT, Loop Filter)
 */
#pragma once

#include "vp9_parsed_frame.h"
#include "compute_vp9/decoder.h"

/**
 * Reconstructs a frame on the CPU using the parsed block metadata and coefficients.
 * Performs intra-prediction, motion-compensation (inter), IDCT residual addition,
 * and deblocking loop filtering.
 */
bool vp9_reconstruct_frame(const vp9_parsed_frame_t *pf,
                           const cvp9_frame_info_t *ref_frames[8],
                           cvp9_frame_info_t *out_frame);
