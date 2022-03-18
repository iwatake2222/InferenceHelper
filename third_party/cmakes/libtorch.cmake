if(DEFINED ANDROID_ABI)
    set(LIBTORCH_ROOTDIR "${CMAKE_CURRENT_LIST_DIR}/../libtorch_prebuilt/android")
elseif(MSVC_VERSION)
    if(INFERENCE_HELPER_ENABLE_LIBTORCH_CUDA)
        set(LIBTORCH_ROOTDIR "${CMAKE_CURRENT_LIST_DIR}/../libtorch_prebuilt/win-x64-gpu")
    else()
        set(LIBTORCH_ROOTDIR "${CMAKE_CURRENT_LIST_DIR}/../libtorch_prebuilt/win-x64")
    endif()
else()
    if(${BUILD_SYSTEM} STREQUAL "armv7")
        message(FATAL_ERROR "[LIBTORCH] unsupported platform")
    elseif(${BUILD_SYSTEM} STREQUAL "aarch64")
        message(FATAL_ERROR "[LIBTORCH] unsupported platform")
    else()
        if(INFERENCE_HELPER_ENABLE_LIBTORCH_CUDA)
            ### For some reasons, LibTorch+CUDA library doesn't work(it's missing some symbols?), and LibTorch+CPU library works with CUDA option
            # set(LIBTORCH_ROOTDIR "${CMAKE_CURRENT_LIST_DIR}/../libtorch_prebuilt/linux-x64-gpu")
            set(LIBTORCH_ROOTDIR "${CMAKE_CURRENT_LIST_DIR}/../libtorch_prebuilt/linux-x64")
        else()
            set(LIBTORCH_ROOTDIR "${CMAKE_CURRENT_LIST_DIR}/../libtorch_prebuilt/linux-x64")
        endif()
    endif()
endif()

if(DEFINED ANDROID_ABI)
    message(FATAL_ERROR "[libtorch] unsupported platform")
    add_library(LIBTORCH SHARED IMPORTED GLOBAL)
    set_target_properties(
        LIBTORCH
        PROPERTIES IMPORTED_LOCATION
        ${LIBTORCH_ROOTDIR}/jni/${ANDROID_ABI}/libpytorch_vision_jni.so
    )
    set(LIBTORCH_LIB LIBTORCH)
    set(LIBTORCH_INC "${LIBTORCH_ROOTDIR}/headers")
else()
    set(TORCH_CUDA_ARCH_LIST Auto)
    find_package(Caffe2 REQUIRED PATHS "${LIBTORCH_ROOTDIR}")
    find_package(Torch REQUIRED PATHS "${LIBTORCH_ROOTDIR}")

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
    set(LIBTORCH_LIB "${TORCH_LIBRARIES}")
    set(LIBTORCH_INC )

    if(MSVC)
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    add_custom_command(TARGET InferenceHelper
                        POST_BUILD
                        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        ${TORCH_DLLS}
                        ${CMAKE_BINARY_DIR})
    endif(MSVC)
endif()
