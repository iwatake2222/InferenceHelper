if(DEFINED ANDROID_ABI)
    
elseif(MSVC_VERSION)
    if(INFERENCE_HELPER_ENABLE_ONNX_RUNTIME_CUDA)
        set(ONNX_RUNTIME_ROOTDIR "${CMAKE_CURRENT_LIST_DIR}/../onnxruntime_prebuilt/windows-x64-gpu")
    else()
        set(ONNX_RUNTIME_ROOTDIR "${CMAKE_CURRENT_LIST_DIR}/../onnxruntime_prebuilt/windows-x64")
    endif()
    
else()
    if(${BUILD_SYSTEM} STREQUAL "armv7")
        
    elseif(${BUILD_SYSTEM} STREQUAL "aarch64")
        
    else()
        
    endif()
endif()


if(MSVC_VERSION)
    set(ONNX_RUNTIME_LIB ${ONNX_RUNTIME_ROOTDIR}/lib/onnxruntime.lib)
    if(INFERENCE_HELPER_ENABLE_ONNX_RUNTIME_CUDA)
        set(ONNX_RUNTIME_LIB ${ONNX_RUNTIME_LIB} ${ONNX_RUNTIME_ROOTDIR}/lib/onnxruntime_providers_cuda.lib)
    endif()
    # Copy dll files into the binary directory
    file(GLOB DLL_FILES files "${ONNX_RUNTIME_ROOTDIR}/lib/*.dll")
    file(COPY ${DLL_FILES} DESTINATION ${CMAKE_BINARY_DIR})
    # Ensure to use pre-built onnxruntime.dll rather than that in system folder
    file(COPY ${DLL_FILES} DESTINATION ${CMAKE_BINARY_DIR}/Debug)
    file(COPY ${DLL_FILES} DESTINATION ${CMAKE_BINARY_DIR}/RelWithDebInfo)
    file(COPY ${DLL_FILES} DESTINATION ${CMAKE_BINARY_DIR}/Release)
else()

endif()

set(ONNX_RUNTIME_INC "${ONNX_RUNTIME_ROOTDIR}/include" "${ONNX_RUNTIME_ROOTDIR}/include/onnxruntime/core/session")
