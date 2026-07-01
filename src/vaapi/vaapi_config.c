/**
 * compute-vp9 — VA-API configuration stubs
 */
#ifdef ENABLE_VAAPI
#include <va/va_backend.h>

VAStatus vaapi_config_dummy(void)
{
    return VA_STATUS_SUCCESS;
}
#endif
