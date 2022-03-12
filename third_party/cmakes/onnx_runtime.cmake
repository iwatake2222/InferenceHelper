if(DEFINED ANDROID_ABI)
    
elseif(MSVC_VERSION)
    set(ONNX_RUNTIME_ROOTDIR "${CMAKE_CURRENT_LIST_DIR}/../onnxruntime_prebuilt/windows-x64")
else()
    if(${BUILD_SYSTEM} STREQUAL "armv7")
        
    elseif(${BUILD_SYSTEM} STREQUAL "aarch64")
        
    else()
        
    endif()
endif()


if(MSVC_VERSION)
    set(ONNX_RUNTIME_LIB
        $<$<CONFIG:Debug>:${ONNX_RUNTIME_ROOTDIR}/lib/onnxruntime.lib>
        $<$<CONFIG:RelWithDebInfo>:${ONNX_RUNTIME_ROOTDIR}/lib/onnxruntime.lib>
        $<$<CONFIG:Release>:${ONNX_RUNTIME_ROOTDIR}/lib/onnxruntime.lib>
        $<$<CONFIG:MinSizeRel>:${ONNX_RUNTIME_ROOTDIR}/lib/onnxruntime.lib>
    )
    # Ensure to use pre-built onnxruntime.dll rather than that in system folder
    file(COPY ${ONNX_RUNTIME_ROOTDIR}/lib/onnxruntime.dll DESTINATION ${CMAKE_BINARY_DIR})
    file(COPY ${ONNX_RUNTIME_ROOTDIR}/lib/onnxruntime.dll DESTINATION ${CMAKE_BINARY_DIR}/Debug)
    file(COPY ${ONNX_RUNTIME_ROOTDIR}/lib/onnxruntime.dll DESTINATION ${CMAKE_BINARY_DIR}/RelWithDebInfo)
    file(COPY ${ONNX_RUNTIME_ROOTDIR}/lib/onnxruntime.dll DESTINATION ${CMAKE_BINARY_DIR}/Release)
else()

endif()

set(ONNX_RUNTIME_INC "${ONNX_RUNTIME_ROOTDIR}/include" "${ONNX_RUNTIME_ROOTDIR}/include/onnxruntime/core/session")
