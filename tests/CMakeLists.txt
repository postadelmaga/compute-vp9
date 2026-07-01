cmake_minimum_required(VERSION 3.20)

# ── Test executable ──────────────────────────────────────────────────────────
add_executable(test_decoder
    test_decoder.c
    test_bitstream.c
)

target_link_libraries(test_decoder PRIVATE compute_vp9)
target_include_directories(test_decoder PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

# ── Test cases ───────────────────────────────────────────────────────────────
add_test(NAME test_bitstream_reader  COMMAND test_decoder --test bitstream)
add_test(NAME test_range_coder       COMMAND test_decoder --test range_coder)
add_test(NAME test_frame_header      COMMAND test_decoder --test frame_header)
add_test(NAME test_backend_init      COMMAND test_decoder --test backend)
add_test(NAME test_decode_key_frame  COMMAND test_decoder --test keyframe
         WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
