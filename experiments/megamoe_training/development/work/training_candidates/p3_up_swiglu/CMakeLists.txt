cmake_minimum_required(VERSION 3.20)
project(megamoe_up_swiglu NONE)
get_filename_component(DEEPEP_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../DeepEP" ABSOLUTE)
set(CATLASS_ROOT "${DEEPEP_ROOT}/third-party/catlass" CACHE PATH "Pinned CATLASS source")
set(ASCEND_HOME_PATH "$ENV{ASCEND_HOME_PATH}" CACHE PATH "Active official CANN installation")
find_program(BISHENG bisheng PATHS "${ASCEND_HOME_PATH}/tools/bisheng_compiler/bin" NO_DEFAULT_PATH REQUIRED)
set(flags -O2 -std=c++17 -xasc --npu-arch=dav-3510
    --cce-auto-infer-kernel-type=false --cce-auto-sync=off --cce-fatobj-link
    -shared -fPIC -DCATLASS_ARCH=3510 -DENABLE_TENSOR_API -DNOT_DYNAMIC_COMPILE
    -Wl,-soname,libmegamoe_up_swiglu.so)
foreach(p "${CATLASS_ROOT}/include" "${ASCEND_HOME_PATH}/include"
    "${ASCEND_HOME_PATH}/pkg_inc" "${ASCEND_HOME_PATH}/include/experiment/runtime"
    "${ASCEND_HOME_PATH}/pkg_inc/runtime" "${ASCEND_HOME_PATH}/include/op_common"
    "${ASCEND_HOME_PATH}/pkg_inc/op_common" "${ASCEND_HOME_PATH}/compiler/ascendc/act"
    "${ASCEND_HOME_PATH}/aarch64-linux/asc/atcos")
    list(APPEND flags "-I${p}")
endforeach()
find_file(api_header ascendc_api_version.h PATHS
    "${ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw/lib"
    "${ASCEND_HOME_PATH}/aarch64-linux/tikcpp/tikcfw/lib" NO_DEFAULT_PATH)
if(api_header)
    file(STRINGS "${api_header}" api_version REGEX "^#define ASCENDC_API_VERSION [0-9]+")
    string(REGEX REPLACE ".*ASCENDC_API_VERSION ([0-9]+).*" "\\1" api_version "${api_version}")
    list(APPEND flags "-DASCENDC_API_VERSION=${api_version}")
endif()
set(library "${CMAKE_CURRENT_BINARY_DIR}/libmegamoe_up_swiglu.so")
add_custom_command(OUTPUT "${library}"
    COMMAND "${BISHENG}" ${flags} "${CMAKE_CURRENT_SOURCE_DIR}/native/up_swiglu.cpp"
        "-L${ASCEND_HOME_PATH}/lib64" -lplatform -ltiling_api -lruntime -lascendcl -lstdc++ -lm -o "${library}"
    DEPENDS native/up_swiglu.cpp VERBATIM)
add_custom_target(megamoe_up_swiglu ALL DEPENDS "${library}")
install(FILES "${library}" DESTINATION p3_up_swiglu)
