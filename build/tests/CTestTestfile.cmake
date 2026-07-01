# CMake generated Testfile for 
# Source directory: /mnt/dati/Dev/compute-vp9/tests
# Build directory: /mnt/dati/Dev/compute-vp9/build/tests
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test([=[test_bitstream_reader]=] "/mnt/dati/Dev/compute-vp9/build/tests/test_decoder" "--test" "bitstream")
set_tests_properties([=[test_bitstream_reader]=] PROPERTIES  _BACKTRACE_TRIPLES "/mnt/dati/Dev/compute-vp9/tests/CMakeLists.txt;16;add_test;/mnt/dati/Dev/compute-vp9/tests/CMakeLists.txt;0;")
add_test([=[test_frame_header]=] "/mnt/dati/Dev/compute-vp9/build/tests/test_decoder" "--test" "frame_header")
set_tests_properties([=[test_frame_header]=] PROPERTIES  _BACKTRACE_TRIPLES "/mnt/dati/Dev/compute-vp9/tests/CMakeLists.txt;17;add_test;/mnt/dati/Dev/compute-vp9/tests/CMakeLists.txt;0;")
add_test([=[test_backend_init]=] "/mnt/dati/Dev/compute-vp9/build/tests/test_decoder" "--test" "backend")
set_tests_properties([=[test_backend_init]=] PROPERTIES  _BACKTRACE_TRIPLES "/mnt/dati/Dev/compute-vp9/tests/CMakeLists.txt;18;add_test;/mnt/dati/Dev/compute-vp9/tests/CMakeLists.txt;0;")
add_test([=[test_decode_key_frame]=] "/mnt/dati/Dev/compute-vp9/build/tests/test_decoder" "--test" "keyframe")
set_tests_properties([=[test_decode_key_frame]=] PROPERTIES  WORKING_DIRECTORY "/mnt/dati/Dev/compute-vp9/tests" _BACKTRACE_TRIPLES "/mnt/dati/Dev/compute-vp9/tests/CMakeLists.txt;19;add_test;/mnt/dati/Dev/compute-vp9/tests/CMakeLists.txt;0;")
