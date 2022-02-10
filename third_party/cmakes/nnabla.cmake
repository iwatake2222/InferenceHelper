if(DEFINED  ANDROID_ABI)
    # set(NNABLA_LIB ${CMAKE_CURRENT_LIST_DIR}/../nnabla_prebuilt/android/${ANDROID_ABI}/libtensorflowlite.so)
    # add_library(NNABLA SHARED IMPORTED GLOBAL)
    # set_target_properties(
    #     NNABLA
    #     PROPERTIES IMPORTED_LOCATION
    #     ${CMAKE_CURRENT_LIST_DIR}/../nnabla_prebuilt/android/${ANDROID_ABI}/libtensorflowlite.so
    # )
    # set(NNABLA_LIB NNABLA)
elseif(MSVC_VERSION)
    set(NNABLA_INC ${CMAKE_CURRENT_LIST_DIR}/../nnabla_prebuilt/windows-vs2019/include)
    set(NNABLA_LIB 
        ${CMAKE_CURRENT_LIST_DIR}/../nnabla_prebuilt/windows-vs2019/lib/nnabla.lib
        ${CMAKE_CURRENT_LIST_DIR}/../nnabla_prebuilt/windows-vs2019/lib/nnabla_utils.lib
    )
    file(COPY ${CMAKE_CURRENT_LIST_DIR}/../nnabla_prebuilt/windows-vs2019/bin/nnabla.dll DESTINATION ${CMAKE_BINARY_DIR})
    file(COPY ${CMAKE_CURRENT_LIST_DIR}/../nnabla_prebuilt/windows-vs2019/bin/nnabla_utils.dll DESTINATION ${CMAKE_BINARY_DIR})
    file(COPY ${CMAKE_CURRENT_LIST_DIR}/../nnabla_prebuilt/windows-vs2019/bin/archive.dll DESTINATION ${CMAKE_BINARY_DIR})
    if(INFERENCE_HELPER_ENABLE_NNABLA_CUDA)
        set(NNABLA_LIB ${NNABLA_LIB} ${CMAKE_CURRENT_LIST_DIR}/../nnabla_prebuilt/windows-vs2019/lib/nnabla_cuda_110_8.lib)
        file(COPY ${CMAKE_CURRENT_LIST_DIR}/../nnabla_prebuilt/windows-vs2019/bin/nnabla_cuda_110_8.dll DESTINATION ${CMAKE_BINARY_DIR})
    endif()
else()
    if(${BUILD_SYSTEM} STREQUAL "x64_linux")
    set(NNABLA_INC ${CMAKE_CURRENT_LIST_DIR}/../nnabla_prebuilt/ubuntu/include)
        set(NNABLA_LIB
            ${CMAKE_CURRENT_LIST_DIR}/../nnabla_prebuilt/ubuntu/lib/libnnabla.so
            ${CMAKE_CURRENT_LIST_DIR}/../nnabla_prebuilt/ubuntu/lib/libnnabla_utils.so
        )
        if(INFERENCE_HELPER_ENABLE_NNABLA_CUDA)
            set(NNABLA_LIB ${NNABLA_LIB} ${CMAKE_CURRENT_LIST_DIR}/../nnabla_prebuilt/ubuntu/lib/libnnabla_cuda.so)
        endif()
    elseif(${BUILD_SYSTEM} STREQUAL "armv7")
        message(FATAL_ERROR "[nnabla] unsupported platform")
    elseif(${BUILD_SYSTEM} STREQUAL "aarch64")
        set(NNABLA_INC ${CMAKE_CURRENT_LIST_DIR}/../nnabla_prebuilt/aarch64/include)
        set(NNABLA_LIB
            ${CMAKE_CURRENT_LIST_DIR}/../nnabla_prebuilt/aarch64/lib/libnnabla.so
            ${CMAKE_CURRENT_LIST_DIR}/../nnabla_prebuilt/aarch64/lib/libnnabla_utils.so
        )
        if(INFERENCE_HELPER_ENABLE_NNABLA_CUDA)
            set(NNABLA_LIB ${NNABLA_LIB} ${CMAKE_CURRENT_LIST_DIR}/../nnabla_prebuilt/aarch64/lib/libnnabla_cuda.so)
        endif()
    else()
        message(FATAL_ERROR "[nnabla] unsupported platform")
    endif()
    file(COPY ${NNABLA_LIB} DESTINATION ${CMAKE_BINARY_DIR})
endif()
