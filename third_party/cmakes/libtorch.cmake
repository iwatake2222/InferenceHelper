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
        set(LIBTORCH_ROOTDIR "${CMAKE_CURRENT_LIST_DIR}/../libtorch_prebuilt/linux-aarch64")
    else()
        if(INFERENCE_HELPER_ENABLE_LIBTORCH_CUDA)
            set(LIBTORCH_ROOTDIR "${CMAKE_CURRENT_LIST_DIR}/../libtorch_prebuilt/linux-x64-gpu")
        else()
            set(LIBTORCH_ROOTDIR "${CMAKE_CURRENT_LIST_DIR}/../libtorch_prebuilt/linux-x64")
        endif()
    endif()
endif()

# find_package(Caffe2 REQUIRED PATHS "${LIBTORCH_ROOTDIR}")
find_package(Torch REQUIRED PATHS "${LIBTORCH_ROOTDIR}")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(LIBTORCH_LIB "${TORCH_LIBRARIES}")
set(LIBTORCH_INC )

if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET InferenceHelper
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     ${CMAKE_BINARY_DIR})
endif (MSVC)

# if(DEFINED ANDROID_ABI)
#     add_library(LIBTORCH SHARED IMPORTED GLOBAL)
#     set_target_properties(
#         LIBTORCH
#         PROPERTIES IMPORTED_LOCATION
#         ${LIBTORCH_ROOTDIR}/jni/${ANDROID_ABI}/libonnxruntime.so
#     )
#     set(LIBTORCH_LIB LIBTORCH)
#     set(LIBTORCH_INC "${LIBTORCH_ROOTDIR}/headers")
# else()
#     if(MSVC_VERSION)
#         set(LIBTORCH_LIB ${LIBTORCH_ROOTDIR}/lib/onnxruntime.lib)
#         if(INFERENCE_HELPER_ENABLE_LIBTORCH_CUDA)
#             set(LIBTORCH_LIB ${LIBTORCH_LIB} ${LIBTORCH_ROOTDIR}/lib/onnxruntime_providers_cuda.lib)
#         endif()
#         # Copy dll files into the binary directory
#         file(GLOB DLL_FILES files "${LIBTORCH_ROOTDIR}/lib/*.dll")
#         file(COPY ${DLL_FILES} DESTINATION ${CMAKE_BINARY_DIR})
#         # Ensure to use pre-built onnxruntime.dll rather than that in system folder
#         file(COPY ${DLL_FILES} DESTINATION ${CMAKE_BINARY_DIR}/Debug)
#         file(COPY ${DLL_FILES} DESTINATION ${CMAKE_BINARY_DIR}/RelWithDebInfo)
#         file(COPY ${DLL_FILES} DESTINATION ${CMAKE_BINARY_DIR}/Release)
#     else()
#         set(LIBTORCH_LIB ${LIBTORCH_ROOTDIR}/lib/libonnxruntime.so)
#     endif()

#     set(LIBTORCH_INC "${LIBTORCH_ROOTDIR}/include" "${LIBTORCH_ROOTDIR}/include/onnxruntime/core/session")
# endif()
