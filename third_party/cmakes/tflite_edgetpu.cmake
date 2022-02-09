set(TFLITE_EDGETPU_INC
    ${CMAKE_CURRENT_LIST_DIR}/../edgetpu_prebuilt/include/
)

if(MSVC_VERSION)
    set(TFLITE_EDGETPU_LIB
        $<$<CONFIG:Debug>:${CMAKE_CURRENT_LIST_DIR}/../edgetpu_prebuilt/direct/windows-vs2019/debug/edgetpu_direct_all.dll.if.lib>
        $<$<CONFIG:RelWithDebInfo>:${CMAKE_CURRENT_LIST_DIR}/../edgetpu_prebuilt/direct/windows-vs2019/edgetpu_direct_all.dll.if.lib>
        $<$<CONFIG:Release>:${CMAKE_CURRENT_LIST_DIR}/../edgetpu_prebuilt/direct/windows-vs2019/edgetpu_direct_all.dll.if.lib>
        $<$<CONFIG:MinSizeRel>:${CMAKE_CURRENT_LIST_DIR}/../edgetpu_prebuilt/direct/windows-vs2019/edgetpu_direct_all.dll.if.lib>
    )
    file(COPY ${CMAKE_CURRENT_LIST_DIR}/../edgetpu_prebuilt/direct/windows-vs2019/edgetpu_direct_all.dll DESTINATION ${CMAKE_BINARY_DIR})
    file(COPY ${CMAKE_CURRENT_LIST_DIR}/../edgetpu_prebuilt/direct/windows-vs2019/debug/edgetpu_direct_all.dll DESTINATION ${CMAKE_BINARY_DIR}/Debug)
    file(COPY ${CMAKE_CURRENT_LIST_DIR}/../edgetpu_prebuilt/direct/windows-vs2019/libusb-1.0.dll DESTINATION ${CMAKE_BINARY_DIR})
else()
    if(${BUILD_SYSTEM} STREQUAL "x64_linux")
        set(TFLITE_EDGETPU_LIB ${CMAKE_CURRENT_LIST_DIR}/../edgetpu_prebuilt/direct/k8/libedgetpu.so.1.0)
    elseif(${BUILD_SYSTEM} STREQUAL "armv7")
        set(TFLITE_EDGETPU_LIB ${CMAKE_CURRENT_LIST_DIR}/../edgetpu_prebuilt/direct/armv7a/libedgetpu.so.1.0)
    elseif(${BUILD_SYSTEM} STREQUAL "aarch64")
        set(TFLITE_EDGETPU_LIB ${CMAKE_CURRENT_LIST_DIR}/../edgetpu_prebuilt/direct/aarch64/libedgetpu.so.1.0)
    else()	
        message(FATAL_ERROR "[tflite_edgetpu] unsupported platform")
    endif()
    file(COPY ${TFLITE_EDGETPU_LIB} DESTINATION ${CMAKE_BINARY_DIR})
endif()
