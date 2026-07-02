/**
 * Port of libvpx boolean coder (range decoder)
 *
 * Copyright (c) 2010 The WebM project authors. All Rights Reserved.
 * Licensed under BSD-style license.
 */
#pragma once

#include <stddef.h>
#include <stdint.h>
#include <limits.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef size_t BD_VALUE;

#define BD_VALUE_SIZE ((int)sizeof(BD_VALUE) * CHAR_BIT)
#define LOTS_OF_BITS 0x40000000

typedef struct {
  BD_VALUE value;
  unsigned int range;
  int count;
  const uint8_t *buffer_end;
  const uint8_t *buffer;
} vpx_reader;

int vpx_reader_init(vpx_reader *r, const uint8_t *buffer, size_t size);
void vpx_reader_fill(vpx_reader *r);
const uint8_t *vpx_reader_find_end(vpx_reader *r);

static inline int vpx_reader_has_error(vpx_reader *r) {
  /* LOTS_OF_BITS marks end-of-stream reads: an error is only flagged once
   * the decoder consumed more (virtual) bits than the stream ever had */
  return r->count > BD_VALUE_SIZE && r->count < LOTS_OF_BITS;
}

// 256-byte normalization table used by the range decoder
static const uint8_t vpx_norm[256] = {
  0, 7, 6, 6, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4,
  3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
  2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
  2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
};

/* Canonical libvpx vpx_read: refill happens at entry when the buffered bit
 * budget went negative, and normalization always runs (vpx_norm is 0 for
 * range >= 128) — bit-exact with the reference decoder. */
static inline int vpx_read(vpx_reader *r, int prob) {
  unsigned int bit = 0;
  BD_VALUE value;
  BD_VALUE bigsplit;
  int count;
  unsigned int range;
  unsigned int split = (r->range * prob + (256 - prob)) >> 8;

  if (r->count < 0) vpx_reader_fill(r);

  value = r->value;
  count = r->count;

  bigsplit = (BD_VALUE)split << (BD_VALUE_SIZE - 8);

  range = split;

  if (value >= bigsplit) {
    range = r->range - split;
    value = value - bigsplit;
    bit = 1;
  }

  {
    const unsigned char shift = vpx_norm[(unsigned char)range];
    range <<= shift;
    value <<= shift;
    count -= shift;
  }

  r->value = value;
  r->count = count;
  r->range = range;

  return bit;
}

static inline int vpx_read_bit(vpx_reader *r) {
  return vpx_read(r, 128); // 50% probability
}

static inline int vpx_read_literal(vpx_reader *r, int bits) {
  int literal = 0;
  for (int bit = bits - 1; bit >= 0; bit--) {
    literal |= vpx_read_bit(r) << bit;
  }
  return literal;
}

#ifdef __cplusplus
}
#endif
