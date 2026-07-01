# Function to compile GLSL shaders to SPIR-V
function(compile_shaders TARGET_NAME)
    set(SPIRV_OUTPUTS "")

    foreach(GLSL_FILE ${ARGN})
        get_filename_component(FILE_NAME ${GLSL_FILE} NAME)
        set(SPIRV_FILE "${CMAKE_CURRENT_BINARY_DIR}/shaders/${FILE_NAME}.spv")

        add_custom_command(
            OUTPUT ${SPIRV_FILE}
            COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/shaders"
            COMMAND ${GLSLC} -fshader-stage=compute
                    "${CMAKE_CURRENT_SOURCE_DIR}/${GLSL_FILE}"
                    -o ${SPIRV_FILE}
            DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${GLSL_FILE}"
            COMMENT "Compiling shader: ${FILE_NAME} → ${FILE_NAME}.spv"
            VERBATIM
        )
        list(APPEND SPIRV_OUTPUTS ${SPIRV_FILE})
    endforeach()

    add_custom_target(${TARGET_NAME} DEPENDS ${SPIRV_OUTPUTS})

    # Embed SPIR-V paths as compile definition for shader_registry.c
    string(REPLACE ";" ":" SPIRV_PATH_LIST "${SPIRV_OUTPUTS}")
    set_property(TARGET compute_vp9 APPEND PROPERTY
        COMPILE_DEFINITIONS "SPIRV_OUTPUT_DIR=\"${CMAKE_CURRENT_BINARY_DIR}/shaders\""
    )
endfunction()
