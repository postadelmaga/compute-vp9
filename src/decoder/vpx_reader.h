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
  // If count has accumulated too many shifts and we've read past the end, error out.
  return r->count > BD_VALUE_SIZE && r->buffer >= r->buffer_end;
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

static inline int vpx_read(vpx_reader *r, int prob) {
  unsigned int split = (r->range * prob + (256 - prob)) >> 8;
  BD_VALUE value = r->value;
  int count = r->count;
  unsigned int range = r->range;
  BD_VALUE bigsplit = (BD_VALUE)split << (BD_VALUE_SIZE - 8);
  int bit;

  if (value >= bigsplit) {
    range -= split;
    value -= bigsplit;
    bit = 1;
  } else {
    range = split;
    bit = 0;
  }

  if (range >= 128) {
    r->value = value;
    r->range = range;
    return bit;
  }

  // Normalization
  int shift = vpx_norm[range];
  range <<= shift;
  value <<= shift;
  count -= shift;

  if (count < 0) {
    // Fill the buffer
    const uint8_t *buffer = r->buffer;
    const uint8_t *buffer_end = r->buffer_end;
    int loop_end = 0 - count;
    int fill_shift = BD_VALUE_SIZE - 8 - loop_end;

    while (loop_end >= 8 && buffer < buffer_end) {
      value |= (BD_VALUE)*buffer++ << fill_shift;
      loop_end -= 8;
      fill_shift += 8;
      count += 8;
    }
    if (loop_end >= 0 && buffer < buffer_end) {
      value |= (BD_VALUE)*buffer++ << fill_shift;
      count += 8;
    }
    r->buffer = buffer;
  }

  r->value = value;
  r->range = range;
  r->count = count;

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
