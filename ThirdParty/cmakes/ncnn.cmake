if(DEFINED  ANDROID_ABI)
	set(NCNN_LIB ${CMAKE_CURRENT_LIST_DIR}/../ncnn_prebuilt/android/${ANDROID_ABI}/lib/libncnn.a)
	set(NCNN_INC ${CMAKE_CURRENT_LIST_DIR}/../ncnn_prebuilt/android/${ANDROID_ABI}/include/ncnn)
elseif(MSVC_VERSION)
	set(NCNN_LIB
		$<$<CONFIG:Debug>:${CMAKE_CURRENT_LIST_DIR}/../ncnn_prebuilt/x64_windows/lib/ncnnd.lib>
		$<$<CONFIG:RelWithDebInfo>:${CMAKE_CURRENT_LIST_DIR}/../ncnn_prebuilt/x64_windows/lib/ncnnRelWithDebInfo.lib>
		$<$<CONFIG:Release>:${CMAKE_CURRENT_LIST_DIR}/../ncnn_prebuilt/x64_windows/lib/ncnn.lib>
		$<$<CONFIG:MinSizeRel>:${CMAKE_CURRENT_LIST_DIR}/../ncnn_prebuilt/x64_windows/lib/ncnnMinSizeRel.lib>
	)
	set(NCNN_INC ${CMAKE_CURRENT_LIST_DIR}/../ncnn_prebuilt/x64_windows/include/ncnn)
else()
	set(NCNN_LIB
		$<$<STREQUAL:${BUILD_SYSTEM},x64_linux>:${CMAKE_CURRENT_LIST_DIR}/../ncnn_prebuilt/x64_linux/lib/libncnn.a>
		$<$<STREQUAL:${BUILD_SYSTEM},armv7>:${CMAKE_CURRENT_LIST_DIR}/../ncnn_prebuilt/armv7/lib/libncnn.a>
		$<$<STREQUAL:${BUILD_SYSTEM},aarch64>:${CMAKE_CURRENT_LIST_DIR}/../ncnn_prebuilt/aarch64/lib/libncnn.a>
	)
	set(NCNN_INC
		$<$<STREQUAL:${BUILD_SYSTEM},x64_linux>:${CMAKE_CURRENT_LIST_DIR}/../ncnn_prebuilt/x64_linux/include/ncnn>
		$<$<STREQUAL:${BUILD_SYSTEM},armv7>:${CMAKE_CURRENT_LIST_DIR}/../ncnn_prebuilt/armv7/include/ncnn>
		$<$<STREQUAL:${BUILD_SYSTEM},aarch64>:${CMAKE_CURRENT_LIST_DIR}/../ncnn_prebuilt/aarch64/include/ncnn>
	)
endif()
