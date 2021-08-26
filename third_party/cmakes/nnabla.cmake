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
    set(NNABLA_INC ${CMAKE_CURRENT_LIST_DIR}/../nnabla_prebuilt/x64_windows/include)
    set(NNABLA_LIB 
        ${CMAKE_CURRENT_LIST_DIR}/../nnabla_prebuilt/x64_windows/lib/nnabla.lib
        ${CMAKE_CURRENT_LIST_DIR}/../nnabla_prebuilt/x64_windows/lib/nnabla_utils.lib
        ${CMAKE_CURRENT_LIST_DIR}/../nnabla_prebuilt/x64_windows/lib/nnabla_cli.lib
    )
    file(COPY ${CMAKE_CURRENT_LIST_DIR}/../nnabla_prebuilt/x64_windows/bin/nnabla.dll DESTINATION ${CMAKE_BINARY_DIR})
    file(COPY ${CMAKE_CURRENT_LIST_DIR}/../nnabla_prebuilt/x64_windows/bin/nnabla_utils.dll DESTINATION ${CMAKE_BINARY_DIR})
    file(COPY ${CMAKE_CURRENT_LIST_DIR}/../nnabla_prebuilt/x64_windows/bin/nnabla_cli.dll DESTINATION ${CMAKE_BINARY_DIR})
else()
    if(${BUILD_SYSTEM} STREQUAL "x64_linux")
    set(NNABLA_INC ${CMAKE_CURRENT_LIST_DIR}/../nnabla_prebuilt/x64_linux/include)
        set(NNABLA_LIB
            ${CMAKE_CURRENT_LIST_DIR}/../nnabla_prebuilt/x64_linux/lib/libnnabla.so
            ${CMAKE_CURRENT_LIST_DIR}/../nnabla_prebuilt/x64_linux/lib/nnabla_utils.so
            ${CMAKE_CURRENT_LIST_DIR}/../nnabla_prebuilt/x64_linux/lib/nnabla_cli.so
        )
    elseif(${BUILD_SYSTEM} STREQUAL "armv7")
        message(FATAL_ERROR "[nnabla] unsupported platform")
    elseif(${BUILD_SYSTEM} STREQUAL "aarch64")
        set(NNABLA_INC ${CMAKE_CURRENT_LIST_DIR}/../nnabla_prebuilt/aarch64/include)
        set(NNABLA_LIB
            ${CMAKE_CURRENT_LIST_DIR}/../nnabla_prebuilt/aarch64/lib/libnnabla.so
            ${CMAKE_CURRENT_LIST_DIR}/../nnabla_prebuilt/aarch64/lib/nnabla_utils.so
            ${CMAKE_CURRENT_LIST_DIR}/../nnabla_prebuilt/aarch64/lib/nnabla_cli.so
        )
    else()	
        message(FATAL_ERROR "[nnabla] unsupported platform")
    endif()
    file(COPY ${NNABLA_LIB} DESTINATION ${CMAKE_BINARY_DIR})
endif()
