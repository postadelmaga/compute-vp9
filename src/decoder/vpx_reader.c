/**
 * Port of libvpx range decoder implementation
 */
#include "vpx_reader.h"
#include <string.h>

int vpx_reader_init(vpx_reader *r, const uint8_t *buffer, size_t size) {
  if (size && !buffer) {
    return 1;
  } else {
    r->buffer_end = buffer + size;
    r->buffer = buffer;
    r->value = 0;
    r->count = -8;
    r->range = 255;
    vpx_reader_fill(r);
    return vpx_read_bit(r) != 0;  // marker bit (must be 0 in VP9)
  }
}

void vpx_reader_fill(vpx_reader *r) {
  const uint8_t *buffer = r->buffer;
  const uint8_t *buffer_end = r->buffer_end;
  BD_VALUE value = r->value;
  int count = r->count;
  int shift = BD_VALUE_SIZE - 8 - (count + 8);

  while (shift >= 0 && buffer < buffer_end) {
    value |= (BD_VALUE)*buffer++ << shift;
    shift -= 8;
    count += 8;
  }

  r->buffer = buffer;
  r->value = value;
  r->count = count;
}

const uint8_t *vpx_reader_find_end(vpx_reader *r) {
  // Return position where reader has parsed up to.
  // We subtract the unused bits still buffered in 'value'.
  int unused_bits = (r->count & 7) + (r->count >> 3) * 8;
  if (unused_bits < 0) unused_bits = 0;
  return r->buffer - (unused_bits >> 3);
}
