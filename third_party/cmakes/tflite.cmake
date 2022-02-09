set(TFLITE_INC
    ${CMAKE_CURRENT_LIST_DIR}/../tensorflow
    ${CMAKE_CURRENT_LIST_DIR}/../tensorflow_deps/abseil-cpp
    ${CMAKE_CURRENT_LIST_DIR}/../tensorflow_deps/flatbuffers/include
)

if(DEFINED  ANDROID_ABI)
    # set(TFLITE_LIB ${CMAKE_CURRENT_LIST_DIR}/../tensorflow_prebuilt/android/${ANDROID_ABI}/libtensorflowlite.so)
    add_library(TFLITE SHARED IMPORTED GLOBAL)
    set_target_properties(
        TFLITE
        PROPERTIES IMPORTED_LOCATION
        ${CMAKE_CURRENT_LIST_DIR}/../tflite_prebuilt/android/${ANDROID_ABI}/libtensorflowlite.so
    )
    set(TFLITE_LIB TFLITE)
elseif(MSVC_VERSION)
    set(TFLITE_LIB
        $<$<CONFIG:Debug>:${CMAKE_CURRENT_LIST_DIR}/../tflite_prebuilt/windows-vs2019/debug/libtensorflowlite.so.if.lib>
        $<$<CONFIG:RelWithDebInfo>:${CMAKE_CURRENT_LIST_DIR}/../tflite_prebuilt/windows-vs2019/libtensorflowlite.so.if.lib>
        $<$<CONFIG:Release>:${CMAKE_CURRENT_LIST_DIR}/../tflite_prebuilt/windows-vs2019/libtensorflowlite.so.if.lib>
        $<$<CONFIG:MinSizeRel>:${CMAKE_CURRENT_LIST_DIR}/../tflite_prebuilt/windows-vs2019/libtensorflowlite.so.if.lib>
    )
    file(COPY ${CMAKE_CURRENT_LIST_DIR}/../tflite_prebuilt/windows-vs2019/libtensorflowlite.so DESTINATION ${CMAKE_BINARY_DIR})
    file(COPY ${CMAKE_CURRENT_LIST_DIR}/../tflite_prebuilt/windows-vs2019/debug/libtensorflowlite.so DESTINATION ${CMAKE_BINARY_DIR}/Debug)
else()
    if(${BUILD_SYSTEM} STREQUAL "x64_linux")
        set(TFLITE_LIB ${CMAKE_CURRENT_LIST_DIR}/../tflite_prebuilt/ubuntu/libtensorflowlite.so)
    elseif(${BUILD_SYSTEM} STREQUAL "armv7")
        set(TFLITE_LIB ${CMAKE_CURRENT_LIST_DIR}/../tflite_prebuilt/armv7/libtensorflowlite.so)
    elseif(${BUILD_SYSTEM} STREQUAL "aarch64")
        set(TFLITE_LIB ${CMAKE_CURRENT_LIST_DIR}/../tflite_prebuilt/aarch64/libtensorflowlite.so)
    else()	
        message(FATAL_ERROR "[tflite] unsupported platform")
    endif()
    file(COPY ${TFLITE_LIB} DESTINATION ${CMAKE_BINARY_DIR})
endif()
