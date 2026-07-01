#ifdef ENABLE_VAAPI
/**
 * compute-vp9 — VA-API driver entry point
 *
 * Exposes the decoder as a VA-API driver plugin (.so loaded by libva).
 * Chrome/Firefox query this via LIBVA_DRIVER_NAME=compute_vp9.
 *
 * VA-API driver interface reference:
 *   https://github.com/intel/libva/blob/master/va/va_backend.h
 */
#include <va/va_backend.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "compute_vp9/decoder.h"

#define CVP9_VA_VERSION_MAJOR 0
#define CVP9_VA_VERSION_MINOR 1

/* ── Driver private data ─────────────────────────────────────────────────── */
typedef struct {
    cvp9_ctx_t *decoder;
} cvp9_driver_data_t;

#define GET_DRIVER(ctx) ((cvp9_driver_data_t *)((ctx)->pDriverData))

/* ── Profile/entrypoint support table ───────────────────────────────────── */
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

/* ── Surface management ─────────────────────────────────────────────────── */
static VAStatus cvp9_CreateSurfaces(
        VADriverContextP ctx,
        int width, int height, int format,
        int num_surfaces, VASurfaceID *surfaces)
{
    for (int i = 0; i < num_surfaces; i++)
        surfaces[i] = (VASurfaceID)(uintptr_t)(i + 1);
    return VA_STATUS_SUCCESS;
}

static VAStatus cvp9_DestroySurfaces(
        VADriverContextP ctx,
        VASurfaceID *surfaces, int num_surfaces)
{
    (void)surfaces; (void)num_surfaces;
    return VA_STATUS_SUCCESS;
}

/* ── Context management ─────────────────────────────────────────────────── */
static VAStatus cvp9_CreateContext(
        VADriverContextP ctx,
        VAConfigID config_id,
        int picture_width, int picture_height,
        int flag,
        VASurfaceID *render_targets, int num_render_targets,
        VAContextID *context)
{
    cvp9_driver_data_t *drv = GET_DRIVER(ctx);
    (void)drv;
    *context = (VAContextID)1;
    return VA_STATUS_SUCCESS;
}

static VAStatus cvp9_DestroyContext(VADriverContextP ctx, VAContextID context)
{
    (void)ctx; (void)context;
    return VA_STATUS_SUCCESS;
}

/* ── Decode pipeline stubs (to be implemented) ──────────────────────────── */
static VAStatus cvp9_BeginPicture(VADriverContextP ctx,
        VAContextID ctx_id, VASurfaceID render_target)
{ return VA_STATUS_SUCCESS; }

static VAStatus cvp9_RenderPicture(VADriverContextP ctx,
        VAContextID ctx_id, VABufferID *bufs, int num_bufs)
{ return VA_STATUS_SUCCESS; }

static VAStatus cvp9_EndPicture(VADriverContextP ctx, VAContextID ctx_id)
{ return VA_STATUS_SUCCESS; }

static VAStatus cvp9_SyncSurface(VADriverContextP ctx, VASurfaceID surface)
{ return VA_STATUS_SUCCESS; }

/* ── Driver init / teardown ─────────────────────────────────────────────── */
static VAStatus cvp9_Terminate(VADriverContextP ctx)
{
    cvp9_driver_data_t *drv = GET_DRIVER(ctx);
    if (drv) {
        if (drv->decoder) cvp9_destroy(drv->decoder);
        free(drv);
        ctx->pDriverData = NULL;
    }
    return VA_STATUS_SUCCESS;
}

VAStatus __vaDriverInit_0_40(VADriverContextP ctx)
{
    fprintf(stderr, "[compute-vp9] VA-API driver v%d.%d initializing\n",
            CVP9_VA_VERSION_MAJOR, CVP9_VA_VERSION_MINOR);

    cvp9_driver_data_t *drv = calloc(1, sizeof(*drv));
    if (!drv) return VA_STATUS_ERROR_ALLOCATION_FAILED;

    cvp9_config_t cfg = { .backend = CVP9_BACKEND_AUTO };
    cvp9_err_t err = cvp9_create(&cfg, &drv->decoder);
    if (err != CVP9_OK) {
        fprintf(stderr, "[compute-vp9] Failed to initialize decoder: %s\n",
                cvp9_err_str(err));
        free(drv);
        return VA_STATUS_ERROR_OPERATION_FAILED;
    }

    ctx->pDriverData = drv;

    /* Fill vtable */
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
    vtable->vaBeginPicture           = cvp9_BeginPicture;
    vtable->vaRenderPicture          = cvp9_RenderPicture;
    vtable->vaEndPicture             = cvp9_EndPicture;
    vtable->vaSyncSurface            = cvp9_SyncSurface;

    ctx->version_major = CVP9_VA_VERSION_MAJOR;
    ctx->version_minor = CVP9_VA_VERSION_MINOR;
    ctx->str_vendor    = "compute-vp9 " __DATE__;

    fprintf(stderr, "[compute-vp9] Initialized — backend: %s\n",
            drv->decoder ? "Vulkan compute" : "CPU fallback");

    return VA_STATUS_SUCCESS;
}

#endif /* ENABLE_VAAPI */
