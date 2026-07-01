cmake_minimum_required(VERSION 3.20)

project(compute-vp9
    VERSION 0.1.0
    DESCRIPTION "VP9 hardware-accelerated decode via Vulkan compute + VA-API backend"
    LANGUAGES C CXX
)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ─── Options ────────────────────────────────────────────────────────────────
option(ENABLE_VULKAN  "Enable Vulkan compute backend"  ON)
option(ENABLE_OPENCL  "Enable OpenCL compute backend"  OFF)
option(ENABLE_VAAPI   "Enable VA-API driver interface"  ON)
option(ENABLE_TESTS   "Build test suite"                ON)
option(ENABLE_DOCS    "Build documentation"             OFF)

# ─── Dependencies ───────────────────────────────────────────────────────────
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

find_package(PkgConfig REQUIRED)

if(ENABLE_VAAPI)
    pkg_check_modules(LIBVA REQUIRED libva)
    pkg_check_modules(LIBVA_DRM REQUIRED libva-drm)
endif()

if(ENABLE_VULKAN)
    find_package(Vulkan REQUIRED)
    find_program(GLSLC glslc HINTS Vulkan::glslc)
    if(NOT GLSLC)
        message(WARNING "glslc not found — shaders will not be compiled automatically")
    endif()
endif()

if(ENABLE_OPENCL)
    find_package(OpenCL REQUIRED)
endif()

# ─── Library target ─────────────────────────────────────────────────────────
add_library(compute_vp9 SHARED
    src/decoder/vp9_decoder.c
    src/decoder/vp9_bitstream.c
    src/decoder/vpx_reader.c
    src/decoder/vp9_entropy.c
    src/decoder/vp9_tile.c
    src/decoder/vp9_frame.c
    src/decoder/vp9_parsed_frame.c
    src/decoder/vp9_reconstruct.c
)

target_include_directories(compute_vp9
    PUBLIC  include src
)
target_link_libraries(compute_vp9 PRIVATE m)

# ─── Shader compilation (after target exists) ────────────────────────────────
if(ENABLE_VULKAN AND GLSLC)
    include(cmake/CompileShaders.cmake)
    compile_shaders(VP9_SHADERS
        shaders/vp9_idct.comp.glsl
        shaders/vp9_mc.comp.glsl
        shaders/vp9_intra_pred.comp.glsl
        shaders/vp9_loopfilter.comp.glsl
    )
    add_dependencies(compute_vp9 VP9_SHADERS)
endif()

if(ENABLE_VULKAN)
    target_sources(compute_vp9 PRIVATE
        src/backend/vulkan/vulkan_backend.c
        src/backend/vulkan/vulkan_utils.c
        src/backend/vulkan/shader_registry.c
    )
    target_link_libraries(compute_vp9 PRIVATE Vulkan::Vulkan)
    target_compile_definitions(compute_vp9 PRIVATE ENABLE_VULKAN)
endif()

if(ENABLE_OPENCL)
    target_sources(compute_vp9 PRIVATE
        src/backend/opencl/opencl_backend.c
    )
    target_link_libraries(compute_vp9 PRIVATE OpenCL::OpenCL)
    target_compile_definitions(compute_vp9 PRIVATE ENABLE_OPENCL)
endif()

if(ENABLE_VAAPI)
    target_sources(compute_vp9 PRIVATE
        src/vaapi/vaapi_driver.c
        src/vaapi/vaapi_surface.c
        src/vaapi/vaapi_config.c
    )
    target_include_directories(compute_vp9 PRIVATE ${LIBVA_INCLUDE_DIRS})
    target_link_libraries(compute_vp9 PRIVATE ${LIBVA_LIBRARIES} ${LIBVA_DRM_LIBRARIES})
    target_compile_definitions(compute_vp9 PUBLIC ENABLE_VAAPI)
endif()

set_target_properties(compute_vp9 PROPERTIES
    VERSION   ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    OUTPUT_NAME "compute_vp9"
)

# ─── VA-API driver plugin ────────────────────────────────────────────────────
if(ENABLE_VAAPI)
    add_library(compute_vp9_drv_video MODULE
        src/vaapi/vaapi_entrypoint.c
    )
    target_link_libraries(compute_vp9_drv_video PRIVATE compute_vp9 ${LIBVA_LIBRARIES})
    target_include_directories(compute_vp9_drv_video PRIVATE include ${LIBVA_INCLUDE_DIRS})
    set_target_properties(compute_vp9_drv_video PROPERTIES
        PREFIX ""
        OUTPUT_NAME "compute_vp9_drv_video"
    )
    install(TARGETS compute_vp9_drv_video
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/dri
    )
endif()

# ─── Tests ──────────────────────────────────────────────────────────────────
if(ENABLE_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# ─── Install ────────────────────────────────────────────────────────────────
include(GNUInstallDirs)
install(TARGETS compute_vp9
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
