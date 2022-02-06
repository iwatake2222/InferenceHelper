set(mnn_use_vulkan OFF)
if(mnn_use_vulkan)
    set(mnn_suffix_vulkan "-vulkan")
else()
    set(mnn_suffix_vulkan "")
endif()


if(DEFINED ANDROID_ABI)
    set(mnn_DIR ${CMAKE_CURRENT_LIST_DIR}/../mnn_prebuilt/android${mnn_suffix_vulkan}/${ANDROID_ABI})
elseif(MSVC_VERSION)
    set(mnn_DIR ${CMAKE_CURRENT_LIST_DIR}/../mnn_prebuilt/windows-vs2019${mnn_suffix_vulkan})
    file(COPY ${mnn_DIR}/lib/MNN.dll DESTINATION ${CMAKE_BINARY_DIR})
    file(COPY ${mnn_DIR}/lib/debug/MNN.dll DESTINATION ${CMAKE_BINARY_DIR}/Debug)
else()
    if(${BUILD_SYSTEM} STREQUAL "armv7")
        set(mnn_DIR ${CMAKE_CURRENT_LIST_DIR}/../mnn_prebuilt/armv7${mnn_suffix_vulkan})
    elseif(${BUILD_SYSTEM} STREQUAL "aarch64")
        set(mnn_DIR ${CMAKE_CURRENT_LIST_DIR}/../mnn_prebuilt/aarch64${mnn_suffix_vulkan})
    else()
        set(mnn_DIR ${CMAKE_CURRENT_LIST_DIR}/../mnn_prebuilt/ubuntu${mnn_suffix_vulkan})
    endif()
endif()

if(MSVC_VERSION)
    set(MNN_LIB
        $<$<CONFIG:Debug>:${mnn_DIR}/lib/debug/MNN.lib>
        $<$<CONFIG:RelWithDebInfo>:${mnn_DIR}/lib/MNN.lib>
        $<$<CONFIG:Release>:${mnn_DIR}/lib/MNN.lib>
        $<$<CONFIG:MinSizeRel>:${mnn_DIR}/lib/MNN.lib>
    )
    # set_target_properties(mnn PROPERTIES IMPORTED_LOCATION ${mnn_DIR}/lib/MNN.lib)
else()
    add_library(mnn SHARED IMPORTED GLOBAL)
    set_target_properties(mnn PROPERTIES IMPORTED_LOCATION ${mnn_DIR}/lib/libMNN.so)
    set(MNN_LIB mnn)
endif()
set(MNN_INC ${mnn_DIR}/include)


# if(DEFINED  ANDROID_ABI)
#     # set(MNN_LIB ${CMAKE_CURRENT_LIST_DIR}/../mnn_prebuilt/android/${ANDROID_ABI}/libMNN.so)
#     add_library(MNN SHARED IMPORTED GLOBAL)
#     set_target_properties(
#         MNN
#         PROPERTIES IMPORTED_LOCATION
#         ${CMAKE_CURRENT_LIST_DIR}/../mnn_prebuilt/android/${ANDROID_ABI}/lib/libMNN.so
#     )
#     set(MNN_LIB MNN)
#     set(MNN_INC ${CMAKE_CURRENT_LIST_DIR}/../mnn_prebuilt/android/${ANDROID_ABI}/include)
# elseif(MSVC_VERSION)
#     file(COPY ${CMAKE_CURRENT_LIST_DIR}/../mnn_prebuilt/x64_windows/VS2017/lib/Debug/MNN.dll DESTINATION ${CMAKE_BINARY_DIR}/Debug)
#     file(COPY ${CMAKE_CURRENT_LIST_DIR}/../mnn_prebuilt/x64_windows/VS2017/lib/RelWithDebInfo/MNN.dll DESTINATION ${CMAKE_BINARY_DIR}/RelWithDebInfo)
#     file(COPY ${CMAKE_CURRENT_LIST_DIR}/../mnn_prebuilt/x64_windows/VS2017/lib/Release/MNN.dll DESTINATION ${CMAKE_BINARY_DIR}/Release)
#     file(COPY ${CMAKE_CURRENT_LIST_DIR}/../mnn_prebuilt/x64_windows/VS2017/lib/MinSizeRel/MNN.dll DESTINATION ${CMAKE_BINARY_DIR}/MinSizeRel)
#     # file(COPY ${CMAKE_CURRENT_LIST_DIR}/../mnn_prebuilt/x64_windows/MNNd.dll DESTINATION ${CMAKE_BINARY_DIR})
#     set(MNN_LIB
#         $<$<CONFIG:Debug>:${CMAKE_CURRENT_LIST_DIR}/../mnn_prebuilt/x64_windows/VS2017/lib/Debug/MNN.lib>
#         $<$<CONFIG:RelWithDebInfo>:${CMAKE_CURRENT_LIST_DIR}/../mnn_prebuilt/x64_windows/VS2017/lib/RelWithDebInfo/MNN.lib>
#         $<$<CONFIG:Release>:${CMAKE_CURRENT_LIST_DIR}/../mnn_prebuilt/x64_windows/VS2017/lib/Release/MNN.lib>
#         $<$<CONFIG:MinSizeRel>:${CMAKE_CURRENT_LIST_DIR}/../mnn_prebuilt/x64_windows/VS2017/lib/MinSizeRel/MNN.lib>
#     )
#     set(MNN_INC ${CMAKE_CURRENT_LIST_DIR}/../mnn_prebuilt/x64_windows/VS2017/include)
# else()
#     set(MNN_LIB
#         $<$<STREQUAL:${BUILD_SYSTEM},x64_linux>:${CMAKE_CURRENT_LIST_DIR}/../mnn_prebuilt/ubuntu-2004/lib/libMNN.so>
#         $<$<STREQUAL:${BUILD_SYSTEM},armv7>:${CMAKE_CURRENT_LIST_DIR}/../mnn_prebuilt/armv7/lib/libMNN.so>
#         $<$<STREQUAL:${BUILD_SYSTEM},aarch64>:${CMAKE_CURRENT_LIST_DIR}/../mnn_prebuilt/aarch64/lib/libMNN.so>
#     )
#     set(MNN_INC
#         $<$<STREQUAL:${BUILD_SYSTEM},x64_linux>:${CMAKE_CURRENT_LIST_DIR}/../mnn_prebuilt/ubuntu-2004/include>
#         $<$<STREQUAL:${BUILD_SYSTEM},armv7>:${CMAKE_CURRENT_LIST_DIR}/../mnn_prebuilt/armv7/include>
#         $<$<STREQUAL:${BUILD_SYSTEM},aarch64>:${CMAKE_CURRENT_LIST_DIR}/../mnn_prebuilt/aarch64/include>
#     )
# endif()
