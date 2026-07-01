/**
 * compute-vp9 — frame allocation and reference management implementation
 */
#include "vp9_frame.h"
#include <stdlib.h>
#include <string.h>

bool cvp9_frame_alloc(cvp9_frame_info_t *f, uint32_t width, uint32_t height)
{
    f->width = width;
    f->height = height;
    f->stride_y = width;
    f->stride_uv = (width + 1) / 2;

    uint32_t size_y = f->stride_y * height;
    uint32_t size_uv = f->stride_uv * ((height + 1) / 2);

    f->plane_y = malloc(size_y);
    f->plane_u = malloc(size_uv);
    f->plane_v = malloc(size_uv);

    if (!f->plane_y || !f->plane_u || !f->plane_v) {
        cvp9_frame_free(f);
        return false;
    }

    /* Initialize to mid-grey YUV (Y=128, U=128, V=128) */
    memset(f->plane_y, 128, size_y);
    memset(f->plane_u, 128, size_uv);
    memset(f->plane_v, 128, size_uv);
    f->pts = 0;

    return true;
}

void cvp9_frame_free(cvp9_frame_info_t *f)
{
    if (!f) return;
    free(f->plane_y);
    free(f->plane_u);
    free(f->plane_v);
    f->plane_y = NULL;
    f->plane_u = NULL;
    f->plane_v = NULL;
}

bool cvp9_frame_copy(cvp9_frame_info_t *dst, const cvp9_frame_info_t *src)
{
    if (!dst || !src) return false;
    
    cvp9_frame_free(dst);
    if (!cvp9_frame_alloc(dst, src->width, src->height)) return false;

    uint32_t size_y = src->stride_y * src->height;
    uint32_t size_uv = src->stride_uv * ((src->height + 1) / 2);

    memcpy(dst->plane_y, src->plane_y, size_y);
    memcpy(dst->plane_u, src->plane_u, size_uv);
    memcpy(dst->plane_v, src->plane_v, size_uv);
    dst->pts = src->pts;

    return true;
}
