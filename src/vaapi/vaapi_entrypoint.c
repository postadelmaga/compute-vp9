#ifdef ENABLE_VAAPI
/**
 * compute-vp9 — VA-API driver entry point
 */
#include <va/va_backend.h>
#include <va/va_drmcommon.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include "compute_vp9/decoder.h"
#include "decoder/vp9_frame.h"

/* DRM fourccs (avoid a libdrm header dependency) */
#define CVP9_DRM_FORMAT_R8   0x20203852  /* 'R8  ' */
#define CVP9_DRM_FORMAT_GR88 0x38385247  /* 'GR88' */
#define CVP9_DRM_FORMAT_NV12 0x3231564E  /* 'NV12' */
#define CVP9_DRM_MOD_LINEAR  0

#define CVP9_VA_VERSION_MAJOR 0
#define CVP9_VA_VERSION_MINOR 1

#define MAX_SURFACES 32
#define MAX_BUFFERS 128
#define MAX_IMAGES 32
#define MAX_PENDING 8

/* ── Driver private structures ───────────────────────────────────────────── */
typedef struct {
    VABufferType type;
    uint32_t     size;
    uint32_t     num_elements;
    void        *data;
} cvp9_buffer_t;

/* A render-target surface: CPU I420 planes (canonical storage) plus an
 * optional NV12 DMA-BUF backing so clients (Chrome) can import the surface
 * zero-copy via vaExportSurfaceHandle. */
typedef struct {
    cvp9_frame_info_t    frame;
    cvp9_export_buffer_t exp;
    int                  has_export;
    int                  nv12_valid;  /* exp.mapped holds the latest frame */
} cvp9_surface_t;

typedef struct {
    cvp9_ctx_t        *decoder;
    cvp9_surface_t     surfaces[MAX_SURFACES];
    cvp9_buffer_t     *buffers[MAX_BUFFERS];

    /* Render context state */
    VASurfaceID        current_render_target;
    uint8_t           *bitstream_buffer;
    size_t             bitstream_size;

    /* GPU-pipelined frames awaiting delivery: decode is submitted in
     * vaEndPicture but the result is copied to its render target only when
     * ready (opportunistically) or in vaSyncSurface (blocking), so several
     * frames overlap on CPU+GPU at the cost of a few frames of latency. */
    VASurfaceID        pending_targets[MAX_PENDING];
    uint8_t            pending_gpu_direct[MAX_PENDING];  /* GPU wrote NV12 into the surface already */
    int                pending_head;
    int                pending_count;

    /* VA-API parameters */
    VADecPictureParameterBufferVP9 pic_param;
    VASliceParameterBufferVP9     slice_param;
    bool                           has_pic_param;
    bool                           has_slice_param;
} cvp9_driver_data_t;

#define GET_DRIVER(ctx) ((cvp9_driver_data_t *)((ctx)->pDriverData))

/* ── Profiles and Entrypoints ────────────────────────────────────────────── */
static const VAProfile supported_profiles[] = {
    VAProfileVP9Profile0,
    VAProfileVP9Profile2,
};

static VAStatus cvp9_QueryConfigProfiles(
        VADriverContextP ctx,
        VAProfile *profiles,
        int *num_profiles)
{
    memcpy(profiles, supported_profiles, sizeof(supported_profiles));
    *num_profiles = sizeof(supported_profiles) / sizeof(VAProfile);
    return VA_STATUS_SUCCESS;
}

static VAStatus cvp9_QueryConfigEntrypoints(
        VADriverContextP ctx,
        VAProfile profile,
        VAEntrypoint *entrypoints,
        int *num_entrypoints)
{
    switch (profile) {
    case VAProfileVP9Profile0:
    case VAProfileVP9Profile2:
        entrypoints[0] = VAEntrypointVLD;
        *num_entrypoints = 1;
        return VA_STATUS_SUCCESS;
    default:
        *num_entrypoints = 0;
        return VA_STATUS_ERROR_UNSUPPORTED_PROFILE;
    }
}

static VAStatus cvp9_CreateConfig(
        VADriverContextP ctx,
        VAProfile profile,
        VAEntrypoint entrypoint,
        VAConfigAttrib *attrib_list,
        int num_attribs,
        VAConfigID *config_id)
{
    (void)attrib_list; (void)num_attribs;
    *config_id = (VAConfigID)profile;
    return VA_STATUS_SUCCESS;
}

static VAStatus cvp9_DestroyConfig(VADriverContextP ctx, VAConfigID config_id)
{
    (void)ctx; (void)config_id;
    return VA_STATUS_SUCCESS;
}

/* ── Surface Management ──────────────────────────────────────────────────── */

/* Re-pack the surface's I420 planes into its NV12 DMA-BUF backing so
 * importers (Chrome's compositor) see the updated frame */
static void cvp9_surface_write_nv12(cvp9_surface_t *s)
{
    if (!s->has_export || !s->frame.plane_y) return;

    const cvp9_frame_info_t *f = &s->frame;
    uint32_t w = f->width, h = f->height;
    uint32_t cw = (w + 1) / 2, ch = (h + 1) / 2;
    uint8_t *dst_y = s->exp.mapped;
    uint8_t *dst_uv = dst_y + (size_t)w * h;

    for (uint32_t r = 0; r < h; r++) {
        memcpy(dst_y + (size_t)r * w, f->plane_y + (size_t)r * f->stride_y, w);
    }
    for (uint32_t r = 0; r < ch; r++) {
        const uint8_t *u = f->plane_u + (size_t)r * f->stride_uv;
        const uint8_t *v = f->plane_v + (size_t)r * f->stride_uv;
        uint8_t *uv = dst_uv + (size_t)r * 2 * cw;
        for (uint32_t c = 0; c < cw; c++) {
            uv[2 * c]     = u[c];
            uv[2 * c + 1] = v[c];
        }
    }
}

static VAStatus cvp9_create_surfaces_common(
        cvp9_driver_data_t *drv,
        unsigned int width, unsigned int height,
        unsigned int num_surfaces, VASurfaceID *surfaces)
{
    for (unsigned int i = 0; i < num_surfaces; i++) {
        int slot = -1;
        for (int s = 0; s < MAX_SURFACES; s++) {
            if (!drv->surfaces[s].frame.plane_y) { slot = s; break; }
        }
        if (slot == -1) return VA_STATUS_ERROR_ALLOCATION_FAILED;

        cvp9_surface_t *surf = &drv->surfaces[slot];
        if (!cvp9_frame_alloc(&surf->frame, width, height))
            return VA_STATUS_ERROR_ALLOCATION_FAILED;

        /* NV12 DMA-BUF backing for zero-copy import by the client */
        uint32_t cw = (width + 1) / 2, ch = (height + 1) / 2;
        uint64_t nv12_size = (uint64_t)width * height + 2ull * cw * ch;
        surf->has_export =
            (cvp9_export_buffer_alloc(drv->decoder, nv12_size, &surf->exp) == CVP9_OK &&
             surf->exp.fd >= 0);

        surfaces[i] = (VASurfaceID)(slot + 1);
    }
    return VA_STATUS_SUCCESS;
}

static VAStatus cvp9_CreateSurfaces(
        VADriverContextP ctx,
        int width, int height, int format,
        int num_surfaces, VASurfaceID *surfaces)
{
    cvp9_driver_data_t *drv = GET_DRIVER(ctx);
    if (!drv) return VA_STATUS_ERROR_INVALID_CONTEXT;
    (void)format;
    return cvp9_create_surfaces_common(drv, width, height, num_surfaces, surfaces);
}

static VAStatus cvp9_CreateSurfaces2(
        VADriverContextP ctx,
        unsigned int format,
        unsigned int width, unsigned int height,
        VASurfaceID *surfaces, unsigned int num_surfaces,
        VASurfaceAttrib *attrib_list, unsigned int num_attribs)
{
    cvp9_driver_data_t *drv = GET_DRIVER(ctx);
    if (!drv) return VA_STATUS_ERROR_INVALID_CONTEXT;

    /* Accept the pixel formats we can serve (NV12 via export, I420 via
     * vaGetImage); reject anything else */
    for (unsigned int i = 0; attrib_list && i < num_attribs; i++) {
        if (attrib_list[i].type == VASurfaceAttribPixelFormat &&
            (attrib_list[i].flags & VA_SURFACE_ATTRIB_SETTABLE)) {
            uint32_t fourcc = (uint32_t)attrib_list[i].value.value.i;
            if (fourcc != 0 && fourcc != VA_FOURCC_NV12 && fourcc != VA_FOURCC_I420) {
                return VA_STATUS_ERROR_UNSUPPORTED_RT_FORMAT;
            }
        }
    }
    (void)format;
    return cvp9_create_surfaces_common(drv, width, height, num_surfaces, surfaces);
}

static VAStatus cvp9_DestroySurfaces(
        VADriverContextP ctx,
        VASurfaceID *surfaces, int num_surfaces)
{
    cvp9_driver_data_t *drv = GET_DRIVER(ctx);
    if (!drv) return VA_STATUS_ERROR_INVALID_CONTEXT;

    for (int i = 0; i < num_surfaces; i++) {
        int slot = (int)surfaces[i] - 1;
        if (slot >= 0 && slot < MAX_SURFACES) {
            cvp9_frame_free(&drv->surfaces[slot].frame);
            if (drv->surfaces[slot].has_export) {
                cvp9_export_buffer_free(drv->decoder, &drv->surfaces[slot].exp);
                drv->surfaces[slot].has_export = 0;
            }
        }
    }
    return VA_STATUS_SUCCESS;
}

/* ── Context Management ──────────────────────────────────────────────────── */
static VAStatus cvp9_CreateContext(
        VADriverContextP ctx,
        VAConfigID config_id,
        int picture_width, int picture_height,
        int flag,
        VASurfaceID *render_targets, int num_render_targets,
        VAContextID *context)
{
    (void)picture_width; (void)picture_height; (void)flag;
    (void)render_targets; (void)num_render_targets; (void)config_id;
    *context = (VAContextID)1;
    return VA_STATUS_SUCCESS;
}

static VAStatus cvp9_DestroyContext(VADriverContextP ctx, VAContextID context)
{
    (void)ctx; (void)context;
    return VA_STATUS_SUCCESS;
}

/* ── Buffer Management ───────────────────────────────────────────────────── */
static VAStatus cvp9_CreateBuffer(
        VADriverContextP ctx,
        VAContextID context,
        VABufferType type,
        unsigned int size,
        unsigned int num_elements,
        void *data,
        VABufferID *buf_id)
{
    cvp9_driver_data_t *drv = GET_DRIVER(ctx);
    if (!drv) return VA_STATUS_ERROR_INVALID_CONTEXT;

    int slot = -1;
    for (int i = 0; i < MAX_BUFFERS; i++) {
        if (!drv->buffers[i]) { slot = i; break; }
    }
    if (slot == -1) return VA_STATUS_ERROR_ALLOCATION_FAILED;

    cvp9_buffer_t *buf = calloc(1, sizeof(*buf));
    if (!buf) return VA_STATUS_ERROR_ALLOCATION_FAILED;

    buf->type = type;
    buf->size = size;
    buf->num_elements = num_elements;
    buf->data = malloc(size * num_elements);
    if (!buf->data) {
        free(buf);
        return VA_STATUS_ERROR_ALLOCATION_FAILED;
    }

    if (data) {
        memcpy(buf->data, data, size * num_elements);
    } else {
        memset(buf->data, 0, size * num_elements);
    }

    drv->buffers[slot] = buf;
    *buf_id = (VABufferID)(slot + 1);
    return VA_STATUS_SUCCESS;
}

static VAStatus cvp9_DestroyBuffer(VADriverContextP ctx, VABufferID buf_id)
{
    cvp9_driver_data_t *drv = GET_DRIVER(ctx);
    if (!drv) return VA_STATUS_ERROR_INVALID_CONTEXT;

    int slot = (int)buf_id - 1;
    if (slot < 0 || slot >= MAX_BUFFERS || !drv->buffers[slot]) {
        return VA_STATUS_ERROR_INVALID_BUFFER;
    }

    free(drv->buffers[slot]->data);
    free(drv->buffers[slot]);
    drv->buffers[slot] = NULL;
    return VA_STATUS_SUCCESS;
}

static VAStatus cvp9_MapBuffer(
        VADriverContextP ctx,
        VABufferID buf_id,
        void **pbuf)
{
    cvp9_driver_data_t *drv = GET_DRIVER(ctx);
    if (!drv) return VA_STATUS_ERROR_INVALID_CONTEXT;

    int slot = (int)buf_id - 1;
    if (slot < 0 || slot >= MAX_BUFFERS || !drv->buffers[slot]) {
        return VA_STATUS_ERROR_INVALID_BUFFER;
    }

    *pbuf = drv->buffers[slot]->data;
    return VA_STATUS_SUCCESS;
}

static VAStatus cvp9_UnmapBuffer(VADriverContextP ctx, VABufferID buf_id)
{
    (void)ctx; (void)buf_id;
    return VA_STATUS_SUCCESS;
}

/* ── Picture Decoding Pipeline ───────────────────────────────────────────── */
static VAStatus cvp9_BeginPicture(VADriverContextP ctx,
                                  VAContextID ctx_id,
                                  VASurfaceID render_target)
{
    cvp9_driver_data_t *drv = GET_DRIVER(ctx);
    if (!drv) return VA_STATUS_ERROR_INVALID_CONTEXT;

    drv->current_render_target = render_target;
    
    free(drv->bitstream_buffer);
    drv->bitstream_buffer = NULL;
    drv->bitstream_size = 0;

    drv->has_pic_param = false;
    drv->has_slice_param = false;

    return VA_STATUS_SUCCESS;
}

static VAStatus cvp9_RenderPicture(VADriverContextP ctx,
                                   VAContextID ctx_id,
                                   VABufferID *bufs,
                                   int num_bufs)
{
    cvp9_driver_data_t *drv = GET_DRIVER(ctx);
    if (!drv) return VA_STATUS_ERROR_INVALID_CONTEXT;

    for (int i = 0; i < num_bufs; i++) {
        int slot = (int)bufs[i] - 1;
        if (slot < 0 || slot >= MAX_BUFFERS || !drv->buffers[slot]) continue;

        cvp9_buffer_t *buf = drv->buffers[slot];
        if (buf->type == VASliceDataBufferType) {
            size_t new_size = drv->bitstream_size + buf->size * buf->num_elements;
            uint8_t *new_buf = realloc(drv->bitstream_buffer, new_size);
            if (!new_buf) return VA_STATUS_ERROR_ALLOCATION_FAILED;

            memcpy(new_buf + drv->bitstream_size, buf->data, buf->size * buf->num_elements);
            drv->bitstream_buffer = new_buf;
            drv->bitstream_size = new_size;
        } else if (buf->type == VAPictureParameterBufferType) {
            if (buf->size >= sizeof(VADecPictureParameterBufferVP9) && buf->data) {
                memcpy(&drv->pic_param, buf->data, sizeof(VADecPictureParameterBufferVP9));
                drv->has_pic_param = true;
            }
        } else if (buf->type == VASliceParameterBufferType) {
            if (buf->size >= sizeof(VASliceParameterBufferVP9) && buf->data) {
                memcpy(&drv->slice_param, buf->data, sizeof(VASliceParameterBufferVP9));
                drv->has_slice_param = true;
            }
        }
    }
    return VA_STATUS_SUCCESS;
}

/* Deliver one decoded frame to its queued render target. When the GPU
 * already packed NV12 straight into the surface's DMA-BUF (gpu_direct),
 * there is nothing left to copy. */
static void cvp9_deliver_frame(cvp9_driver_data_t *drv, const cvp9_frame_info_t *frame)
{
    if (drv->pending_count == 0) return;
    int target = (int)drv->pending_targets[drv->pending_head] - 1;
    int gpu_direct = drv->pending_gpu_direct[drv->pending_head];
    drv->pending_head = (drv->pending_head + 1) % MAX_PENDING;
    drv->pending_count--;
    if (target >= 0 && target < MAX_SURFACES) {
        cvp9_surface_t *surf = &drv->surfaces[target];
        if (gpu_direct) {
            surf->nv12_valid = 1;
        } else {
            cvp9_frame_copy(&surf->frame, frame);
            cvp9_surface_write_nv12(surf);
            surf->nv12_valid = surf->has_export;
        }
    }
}

/* Copy every already-finished frame to its render target without blocking */
static void cvp9_drain_ready_frames(cvp9_driver_data_t *drv)
{
    cvp9_frame_info_t frame;
    while (drv->pending_count > 0 &&
           cvp9_get_frame(drv->decoder, &frame) == CVP9_OK) {
        cvp9_deliver_frame(drv, &frame);
    }
}

static bool cvp9_surface_pending(const cvp9_driver_data_t *drv, VASurfaceID surface)
{
    for (int i = 0; i < drv->pending_count; i++) {
        if (drv->pending_targets[(drv->pending_head + i) % MAX_PENDING] == surface)
            return true;
    }
    return false;
}

static VAStatus cvp9_EndPicture(VADriverContextP ctx, VAContextID ctx_id)
{
    cvp9_driver_data_t *drv = GET_DRIVER(ctx);
    if (!drv) return VA_STATUS_ERROR_INVALID_CONTEXT;
    if (drv->bitstream_size == 0) return VA_STATUS_SUCCESS;

    /* Direct-to-surface: if the render target has a DMA-BUF backing, ask
     * the GPU to pack NV12 straight into it as part of the decode */
    int gpu_direct = 0;
    {
        int target = (int)drv->current_render_target - 1;
        if (target >= 0 && target < MAX_SURFACES &&
            drv->surfaces[target].has_export) {
            gpu_direct = (cvp9_set_render_target(drv->decoder,
                              &drv->surfaces[target].exp) == CVP9_OK);
        }
    }

    /* Submit decode; if the GPU pipeline is full, deliver the oldest frame
     * first (blocking) and retry */
    cvp9_err_t err;
    for (;;) {
        if (drv->has_pic_param && drv->has_slice_param) {
            err = cvp9_decode_vaapi(drv->decoder, drv->bitstream_buffer, drv->bitstream_size, 0,
                                     &drv->pic_param, &drv->slice_param);
        } else {
            err = cvp9_decode(drv->decoder, drv->bitstream_buffer, drv->bitstream_size, 0);
        }
        if (err != CVP9_ERR_AGAIN) break;

        cvp9_frame_info_t frame;
        if (cvp9_get_frame_sync(drv->decoder, &frame) != CVP9_OK) break;
        cvp9_deliver_frame(drv, &frame);
    }
    if (err != CVP9_OK) {
        cvp9_set_render_target(drv->decoder, NULL);
        fprintf(stderr, "[compute-vp9] VA-API decode failed: %d\n", err);
        return VA_STATUS_ERROR_DECODING_ERROR;
    }

    /* Queue the render target for asynchronous delivery */
    if (drv->pending_count < MAX_PENDING) {
        int wr = (drv->pending_head + drv->pending_count) % MAX_PENDING;
        drv->pending_targets[wr] = drv->current_render_target;
        drv->pending_gpu_direct[wr] = (uint8_t)gpu_direct;
        drv->pending_count++;
    }

    /* Opportunistically deliver whatever the GPU already finished */
    cvp9_drain_ready_frames(drv);

    return VA_STATUS_SUCCESS;
}

static VAStatus cvp9_SyncSurface(VADriverContextP ctx, VASurfaceID surface)
{
    cvp9_driver_data_t *drv = GET_DRIVER(ctx);
    if (!drv) return VA_STATUS_ERROR_INVALID_CONTEXT;

    while (cvp9_surface_pending(drv, surface)) {
        cvp9_frame_info_t frame;
        if (cvp9_get_frame_sync(drv->decoder, &frame) != CVP9_OK) {
            /* Decoder lost the pipeline (e.g. resolution change) — drop */
            drv->pending_count = 0;
            break;
        }
        cvp9_deliver_frame(drv, &frame);
    }
    return VA_STATUS_SUCCESS;
}

/* ── Image & Mapping Management ──────────────────────────────────────────── */
static VAStatus cvp9_CreateImage(
        VADriverContextP ctx,
        VAImageFormat *format,
        int width,
        int height,
        VAImage *image)
{
    cvp9_driver_data_t *drv = GET_DRIVER(ctx);
    if (!drv) return VA_STATUS_ERROR_INVALID_CONTEXT;

    uint32_t size_y = width * height;
    uint32_t size_uv = ((width + 1) / 2) * ((height + 1) / 2);
    uint32_t total_size = size_y + 2 * size_uv;

    VABufferID buf_id;
    VAStatus status = cvp9_CreateBuffer(ctx, 0, VAImageBufferType, total_size, 1, NULL, &buf_id);
    if (status != VA_STATUS_SUCCESS) return status;

    image->image_id = (VAImageID)buf_id;
    image->format = *format;
    image->width = width;
    image->height = height;
    image->buf = buf_id;
    image->data_size = total_size;
    image->num_planes = 3;
    image->pitches[0] = width;
    image->pitches[1] = (width + 1) / 2;
    image->pitches[2] = (width + 1) / 2;
    image->offsets[0] = 0;
    image->offsets[1] = size_y;
    image->offsets[2] = size_y + size_uv;

    return VA_STATUS_SUCCESS;
}

static VAStatus cvp9_DestroyImage(VADriverContextP ctx, VAImageID image)
{
    return cvp9_DestroyBuffer(ctx, (VABufferID)image);
}

static VAStatus cvp9_GetImage(
        VADriverContextP ctx,
        VASurfaceID surface,
        int x, int y,
        unsigned int width, unsigned int height,
        VAImageID image_id)
{
    cvp9_driver_data_t *drv = GET_DRIVER(ctx);
    if (!drv) return VA_STATUS_ERROR_INVALID_CONTEXT;

    int surf_slot = (int)surface - 1;
    if (surf_slot < 0 || surf_slot >= MAX_SURFACES) return VA_STATUS_ERROR_INVALID_SURFACE;
    cvp9_surface_t *surf = &drv->surfaces[surf_slot];
    cvp9_frame_info_t *surf_frame = &surf->frame;
    if (!surf_frame->plane_y) return VA_STATUS_ERROR_INVALID_SURFACE;

    int buf_slot = (int)image_id - 1;
    if (buf_slot < 0 || buf_slot >= MAX_BUFFERS || !drv->buffers[buf_slot]) {
        return VA_STATUS_ERROR_INVALID_IMAGE;
    }

    cvp9_buffer_t *img_buf = drv->buffers[buf_slot];
    uint8_t *dst = img_buf->data;

    uint32_t w = surf_frame->width;
    uint32_t h = surf_frame->height;
    uint32_t size_y = w * h;
    uint32_t size_uv = ((w + 1) / 2) * ((h + 1) / 2);

    /* Clamp sizes to destination bounds */
    if (width > w) width = w;
    if (height > h) height = h;

    uint32_t ch = (height + 1) / 2;
    uint32_t cw = (width + 1) / 2;

    if (surf->nv12_valid && surf->has_export) {
        /* Frame lives in the surface's NV12 DMA-BUF (GPU-direct delivery) */
        const uint8_t *src_y = surf->exp.mapped;
        const uint8_t *src_uv = src_y + size_y;
        uint32_t scw = (w + 1) / 2;

        for (uint32_t r = 0; r < height; r++) {
            memcpy(dst + r * width, src_y + r * w, width);
        }
        for (uint32_t r = 0; r < ch; r++) {
            const uint8_t *uv = src_uv + r * 2 * scw;
            uint8_t *du = dst + size_y + r * cw;
            uint8_t *dv = dst + size_y + size_uv + r * cw;
            for (uint32_t c = 0; c < cw; c++) {
                du[c] = uv[2 * c];
                dv[c] = uv[2 * c + 1];
            }
        }
        return VA_STATUS_SUCCESS;
    }

    /* Copy Y plane */
    for (uint32_t r = 0; r < height; r++) {
        memcpy(dst + r * width, surf_frame->plane_y + r * surf_frame->stride_y, width);
    }
    /* Copy U plane */
    for (uint32_t r = 0; r < ch; r++) {
        memcpy(dst + size_y + r * cw, surf_frame->plane_u + r * surf_frame->stride_uv, cw);
    }
    /* Copy V plane */
    for (uint32_t r = 0; r < ch; r++) {
        memcpy(dst + size_y + size_uv + r * cw, surf_frame->plane_v + r * surf_frame->stride_uv, cw);
    }

    return VA_STATUS_SUCCESS;
}

/* ── Driver Initialization and Teardown ───────────────────────────────────── */
static VAStatus cvp9_Terminate(VADriverContextP ctx)
{
    cvp9_driver_data_t *drv = GET_DRIVER(ctx);
    if (drv) {
        for (int i = 0; i < MAX_SURFACES; i++) {
            cvp9_frame_free(&drv->surfaces[i].frame);
            if (drv->surfaces[i].has_export && drv->decoder) {
                cvp9_export_buffer_free(drv->decoder, &drv->surfaces[i].exp);
            }
        }

        if (drv->decoder) cvp9_destroy(drv->decoder);

        for (int i = 0; i < MAX_BUFFERS; i++) {
            if (drv->buffers[i]) {
                free(drv->buffers[i]->data);
                free(drv->buffers[i]);
            }
        }

        free(drv->bitstream_buffer);
        free(drv);
        ctx->pDriverData = NULL;
    }
    return VA_STATUS_SUCCESS;
}

static VAStatus cvp9_QueryConfigAttributes(VADriverContextP ctx, VAConfigID config_id, VAProfile *profile, VAEntrypoint *entrypoint, VAConfigAttrib *attrib_list, int *num_attribs)
{
    VAProfile prof = (VAProfile)config_id;
    if (prof != VAProfileVP9Profile0 && prof != VAProfileVP9Profile2) {
        return VA_STATUS_ERROR_INVALID_CONFIG;
    }
    *profile = prof;
    *entrypoint = VAEntrypointVLD;

    attrib_list[0].type = VAConfigAttribRTFormat;
    if (prof == VAProfileVP9Profile2) {
        attrib_list[0].value = VA_RT_FORMAT_YUV420_10;
    } else {
        attrib_list[0].value = VA_RT_FORMAT_YUV420;
    }
    *num_attribs = 1;
    return VA_STATUS_SUCCESS;
}

static VAStatus cvp9_GetConfigAttributes(VADriverContextP ctx, VAProfile profile, VAEntrypoint entrypoint, VAConfigAttrib *attrib_list, int num_attribs)
{
    if (profile != VAProfileVP9Profile0 && profile != VAProfileVP9Profile2) {
        return VA_STATUS_ERROR_UNSUPPORTED_PROFILE;
    }
    if (entrypoint != VAEntrypointVLD) {
        return VA_STATUS_ERROR_UNSUPPORTED_ENTRYPOINT;
    }

    for (int i = 0; i < num_attribs; i++) {
        switch (attrib_list[i].type) {
        case VAConfigAttribRTFormat:
            if (profile == VAProfileVP9Profile2) {
                attrib_list[i].value = VA_RT_FORMAT_YUV420_10;
            } else {
                attrib_list[i].value = VA_RT_FORMAT_YUV420;
            }
            break;
        default:
            attrib_list[i].value = VA_ATTRIB_NOT_SUPPORTED;
            break;
        }
    }
    return VA_STATUS_SUCCESS;
}
/* Chrome's VaapiWrapper::FillProfileInfo_Locked requires this to enumerate
 * supported surface formats and resolution limits for each profile. */
static VAStatus cvp9_QuerySurfaceAttributes(
        VADriverContextP ctx, VAConfigID config,
        VASurfaceAttrib *attrib_list, unsigned int *num_attribs)
{
    (void)ctx; (void)config;
    enum { CVP9_NUM_SURFACE_ATTRIBS = 7 };

    if (!num_attribs) return VA_STATUS_ERROR_INVALID_PARAMETER;
    if (!attrib_list) {  /* size query */
        *num_attribs = CVP9_NUM_SURFACE_ATTRIBS;
        return VA_STATUS_SUCCESS;
    }
    if (*num_attribs < CVP9_NUM_SURFACE_ATTRIBS) {
        *num_attribs = CVP9_NUM_SURFACE_ATTRIBS;
        return VA_STATUS_ERROR_MAX_NUM_EXCEEDED;
    }

    unsigned int i = 0;

    attrib_list[i].type = VASurfaceAttribPixelFormat;
    attrib_list[i].flags = VA_SURFACE_ATTRIB_GETTABLE | VA_SURFACE_ATTRIB_SETTABLE;
    attrib_list[i].value.type = VAGenericValueTypeInteger;
    attrib_list[i].value.value.i = VA_FOURCC_NV12;
    i++;

    attrib_list[i].type = VASurfaceAttribPixelFormat;
    attrib_list[i].flags = VA_SURFACE_ATTRIB_GETTABLE | VA_SURFACE_ATTRIB_SETTABLE;
    attrib_list[i].value.type = VAGenericValueTypeInteger;
    attrib_list[i].value.value.i = VA_FOURCC_I420;
    i++;

    attrib_list[i].type = VASurfaceAttribMinWidth;
    attrib_list[i].flags = VA_SURFACE_ATTRIB_GETTABLE;
    attrib_list[i].value.type = VAGenericValueTypeInteger;
    attrib_list[i].value.value.i = 16;
    i++;

    attrib_list[i].type = VASurfaceAttribMinHeight;
    attrib_list[i].flags = VA_SURFACE_ATTRIB_GETTABLE;
    attrib_list[i].value.type = VAGenericValueTypeInteger;
    attrib_list[i].value.value.i = 16;
    i++;

    attrib_list[i].type = VASurfaceAttribMaxWidth;
    attrib_list[i].flags = VA_SURFACE_ATTRIB_GETTABLE;
    attrib_list[i].value.type = VAGenericValueTypeInteger;
    attrib_list[i].value.value.i = 8192;
    i++;

    attrib_list[i].type = VASurfaceAttribMaxHeight;
    attrib_list[i].flags = VA_SURFACE_ATTRIB_GETTABLE;
    attrib_list[i].value.type = VAGenericValueTypeInteger;
    attrib_list[i].value.value.i = 8192;
    i++;

    attrib_list[i].type = VASurfaceAttribMemoryType;
    attrib_list[i].flags = VA_SURFACE_ATTRIB_GETTABLE | VA_SURFACE_ATTRIB_SETTABLE;
    attrib_list[i].value.type = VAGenericValueTypeInteger;
    attrib_list[i].value.value.i = VA_SURFACE_ATTRIB_MEM_TYPE_VA;
    i++;

    *num_attribs = i;
    return VA_STATUS_SUCCESS;
}

/* Zero-copy surface export as DRM PRIME 2 (NV12 DMA-BUF) — the path
 * Chrome's VaapiVideoDecoder uses to hand decoded frames to the compositor */
static VAStatus cvp9_ExportSurfaceHandle(
        VADriverContextP ctx, VASurfaceID surface_id,
        uint32_t mem_type, uint32_t flags, void *descriptor)
{
    cvp9_driver_data_t *drv = GET_DRIVER(ctx);
    if (!drv) return VA_STATUS_ERROR_INVALID_CONTEXT;

    if (mem_type != VA_SURFACE_ATTRIB_MEM_TYPE_DRM_PRIME_2)
        return VA_STATUS_ERROR_UNSUPPORTED_MEMORY_TYPE;

    int slot = (int)surface_id - 1;
    if (slot < 0 || slot >= MAX_SURFACES || !drv->surfaces[slot].frame.plane_y)
        return VA_STATUS_ERROR_INVALID_SURFACE;

    cvp9_surface_t *s = &drv->surfaces[slot];
    if (!s->has_export) return VA_STATUS_ERROR_OPERATION_FAILED;

    uint32_t w = s->frame.width, h = s->frame.height;
    uint32_t cw = (w + 1) / 2;
    uint32_t pitch_y = w, pitch_uv = 2 * cw;
    uint32_t offset_uv = w * h;

    int fd = dup(s->exp.fd);
    if (fd < 0) return VA_STATUS_ERROR_OPERATION_FAILED;

    VADRMPRIMESurfaceDescriptor *d = descriptor;
    memset(d, 0, sizeof(*d));
    d->fourcc = VA_FOURCC_NV12;
    d->width = w;
    d->height = h;
    d->num_objects = 1;
    d->objects[0].fd = fd;
    d->objects[0].size = s->exp.size;
    d->objects[0].drm_format_modifier = CVP9_DRM_MOD_LINEAR;

    if (flags & VA_EXPORT_SURFACE_COMPOSED_LAYERS) {
        d->num_layers = 1;
        d->layers[0].drm_format = CVP9_DRM_FORMAT_NV12;
        d->layers[0].num_planes = 2;
        d->layers[0].object_index[0] = 0;
        d->layers[0].object_index[1] = 0;
        d->layers[0].offset[0] = 0;
        d->layers[0].offset[1] = offset_uv;
        d->layers[0].pitch[0] = pitch_y;
        d->layers[0].pitch[1] = pitch_uv;
    } else {
        d->num_layers = 2;
        d->layers[0].drm_format = CVP9_DRM_FORMAT_R8;
        d->layers[0].num_planes = 1;
        d->layers[0].object_index[0] = 0;
        d->layers[0].offset[0] = 0;
        d->layers[0].pitch[0] = pitch_y;
        d->layers[1].drm_format = CVP9_DRM_FORMAT_GR88;
        d->layers[1].num_planes = 1;
        d->layers[1].object_index[0] = 0;
        d->layers[1].offset[0] = offset_uv;
        d->layers[1].pitch[0] = pitch_uv;
    }

    return VA_STATUS_SUCCESS;
}

static VAStatus cvp9_BufferSetNumElements(VADriverContextP ctx, VABufferID buf_id, unsigned int num_elements) { return VA_STATUS_SUCCESS; }
static VAStatus cvp9_QuerySurfaceStatus(VADriverContextP ctx, VASurfaceID surface, VASurfaceStatus *status)
{
    cvp9_driver_data_t *drv = GET_DRIVER(ctx);
    if (drv) cvp9_drain_ready_frames(drv);
    *status = (drv && cvp9_surface_pending(drv, surface)) ? VASurfaceRendering : VASurfaceReady;
    return VA_STATUS_SUCCESS;
}
static VAStatus cvp9_QueryImageFormats(VADriverContextP ctx, VAImageFormat *format_list, int *num_formats) {
    if (format_list) {
        format_list[0].fourcc = VA_FOURCC_I420;
        format_list[0].byte_order = VA_LSB_FIRST;
        format_list[0].bits_per_pixel = 12;
    }
    *num_formats = 1;
    return VA_STATUS_SUCCESS;
}
static VAStatus cvp9_DeriveImage(VADriverContextP ctx, VASurfaceID surface, VAImage *image) { return VA_STATUS_ERROR_UNIMPLEMENTED; }
static VAStatus cvp9_SetImagePalette(VADriverContextP ctx, VAImageID image, unsigned char *palette) { return VA_STATUS_ERROR_UNIMPLEMENTED; }
static VAStatus cvp9_PutImage(VADriverContextP ctx, VASurfaceID surface, VAImageID image, int src_x, int src_y, unsigned int src_width, unsigned int src_height, int dest_x, int dest_y, unsigned int dest_width, unsigned int dest_height) { return VA_STATUS_ERROR_UNIMPLEMENTED; }
static VAStatus cvp9_QuerySubpictureFormats(VADriverContextP ctx, VAImageFormat *format_list, unsigned int *flags, unsigned int *num_formats) { return VA_STATUS_ERROR_UNIMPLEMENTED; }
static VAStatus cvp9_CreateSubpicture(VADriverContextP ctx, VAImageID image, VASubpictureID *subpicture) { return VA_STATUS_ERROR_UNIMPLEMENTED; }
static VAStatus cvp9_DestroySubpicture(VADriverContextP ctx, VASubpictureID subpicture) { return VA_STATUS_ERROR_UNIMPLEMENTED; }
static VAStatus cvp9_SetSubpictureImage(VADriverContextP ctx, VASubpictureID subpicture, VAImageID image) { return VA_STATUS_ERROR_UNIMPLEMENTED; }
static VAStatus cvp9_SetSubpictureChromakey(VADriverContextP ctx, VASubpictureID subpicture, unsigned int chromakey_min, unsigned int chromakey_max, unsigned int chromakey_mask) { return VA_STATUS_ERROR_UNIMPLEMENTED; }
static VAStatus cvp9_SetSubpictureGlobalAlpha(VADriverContextP ctx, VASubpictureID subpicture, float global_alpha) { return VA_STATUS_ERROR_UNIMPLEMENTED; }
static VAStatus cvp9_AssociateSubpicture(VADriverContextP ctx, VASubpictureID subpicture, VASurfaceID *target_surfaces, int num_surfaces, short src_x, short src_y, unsigned short src_width, unsigned short src_height, short dest_x, short dest_y, unsigned short dest_width, unsigned short dest_height, unsigned int flags) { return VA_STATUS_ERROR_UNIMPLEMENTED; }
static VAStatus cvp9_DeassociateSubpicture(VADriverContextP ctx, VASubpictureID subpicture, VASurfaceID *target_surfaces, int num_surfaces) { return VA_STATUS_ERROR_UNIMPLEMENTED; }
static VAStatus cvp9_QueryDisplayAttributes(VADriverContextP ctx, VADisplayAttribute *attr_list, int *num_attributes) { return VA_STATUS_ERROR_UNIMPLEMENTED; }
static VAStatus cvp9_GetDisplayAttributes(VADriverContextP ctx, VADisplayAttribute *attr_list, int num_attributes) { return VA_STATUS_ERROR_UNIMPLEMENTED; }
static VAStatus cvp9_SetDisplayAttributes(VADriverContextP ctx, VADisplayAttribute *attr_list, int num_attributes) { return VA_STATUS_ERROR_UNIMPLEMENTED; }

VAStatus __vaDriverInit_0_40(VADriverContextP ctx)
{
    fprintf(stderr, "[compute-vp9] VA-API driver v%d.%d initializing\n",
            CVP9_VA_VERSION_MAJOR, CVP9_VA_VERSION_MINOR);

    cvp9_driver_data_t *drv = calloc(1, sizeof(*drv));
    if (!drv) return VA_STATUS_ERROR_ALLOCATION_FAILED;

    cvp9_config_t cfg = { .backend = CVP9_BACKEND_AUTO };
    cvp9_err_t err = cvp9_create(&cfg, &drv->decoder);
    if (err != CVP9_OK) {
        fprintf(stderr, "[compute-vp9] Failed to initialize decoder context: %s\n",
                cvp9_err_str(err));
        free(drv);
        return VA_STATUS_ERROR_OPERATION_FAILED;
    }

    ctx->pDriverData = drv;

    /* Initialize core limits required by libva initialization */
    ctx->max_profiles      = 10;
    ctx->max_entrypoints   = 10;
    ctx->max_attributes    = 10;
    ctx->max_image_formats = 10;
    ctx->max_subpic_formats = 10;

    /* Populate VTable */
    struct VADriverVTable *vtable = ctx->vtable;
    vtable->vaTerminate              = cvp9_Terminate;
    vtable->vaQueryConfigProfiles    = cvp9_QueryConfigProfiles;
    vtable->vaQueryConfigEntrypoints = cvp9_QueryConfigEntrypoints;
    vtable->vaCreateConfig           = cvp9_CreateConfig;
    vtable->vaDestroyConfig          = cvp9_DestroyConfig;
    vtable->vaCreateSurfaces         = cvp9_CreateSurfaces;
    vtable->vaDestroySurfaces        = cvp9_DestroySurfaces;
    vtable->vaCreateContext          = cvp9_CreateContext;
    vtable->vaDestroyContext         = cvp9_DestroyContext;
    vtable->vaCreateBuffer           = cvp9_CreateBuffer;
    vtable->vaDestroyBuffer          = cvp9_DestroyBuffer;
    vtable->vaMapBuffer              = cvp9_MapBuffer;
    vtable->vaUnmapBuffer            = cvp9_UnmapBuffer;
    vtable->vaBeginPicture           = cvp9_BeginPicture;
    vtable->vaRenderPicture          = cvp9_RenderPicture;
    vtable->vaEndPicture             = cvp9_EndPicture;
    vtable->vaSyncSurface            = cvp9_SyncSurface;
    vtable->vaCreateImage            = cvp9_CreateImage;
    vtable->vaDestroyImage           = cvp9_DestroyImage;
    vtable->vaGetImage               = cvp9_GetImage;

    /* Entrypoints required by Chrome's VaapiWrapper */
    vtable->vaQuerySurfaceAttributes = cvp9_QuerySurfaceAttributes;
    vtable->vaCreateSurfaces2        = cvp9_CreateSurfaces2;
    vtable->vaExportSurfaceHandle    = cvp9_ExportSurfaceHandle;

    /* Populating additional vtable items to satisfy strict libva check */
    vtable->vaQueryConfigAttributes  = cvp9_QueryConfigAttributes;
    vtable->vaGetConfigAttributes    = cvp9_GetConfigAttributes;
    vtable->vaBufferSetNumElements   = cvp9_BufferSetNumElements;
    vtable->vaQuerySurfaceStatus     = cvp9_QuerySurfaceStatus;
    vtable->vaQueryImageFormats      = cvp9_QueryImageFormats;
    vtable->vaDeriveImage            = cvp9_DeriveImage;
    vtable->vaSetImagePalette        = cvp9_SetImagePalette;
    vtable->vaPutImage               = cvp9_PutImage;
    vtable->vaQuerySubpictureFormats = cvp9_QuerySubpictureFormats;
    vtable->vaCreateSubpicture       = cvp9_CreateSubpicture;
    vtable->vaDestroySubpicture      = cvp9_DestroySubpicture;
    vtable->vaSetSubpictureImage     = cvp9_SetSubpictureImage;
    vtable->vaSetSubpictureChromakey = cvp9_SetSubpictureChromakey;
    vtable->vaSetSubpictureGlobalAlpha = cvp9_SetSubpictureGlobalAlpha;
    vtable->vaAssociateSubpicture    = cvp9_AssociateSubpicture;
    vtable->vaDeassociateSubpicture  = cvp9_DeassociateSubpicture;
    vtable->vaQueryDisplayAttributes = cvp9_QueryDisplayAttributes;
    vtable->vaGetDisplayAttributes   = cvp9_GetDisplayAttributes;
    vtable->vaSetDisplayAttributes   = cvp9_SetDisplayAttributes;

    ctx->version_major = CVP9_VA_VERSION_MAJOR;
    ctx->version_minor = CVP9_VA_VERSION_MINOR;
    ctx->str_vendor    = "compute-vp9 " __DATE__;

    fprintf(stderr, "[compute-vp9] Initialized — backend: %s\n",
            (cvp9_active_backend(drv->decoder) == CVP9_BACKEND_VULKAN) ? "Vulkan compute" : "CPU fallback");

    return VA_STATUS_SUCCESS;
}

VAStatus __vaDriverInit_1_0(VADriverContextP ctx)
{
    return __vaDriverInit_0_40(ctx);
}
#endif /* ENABLE_VAAPI */
