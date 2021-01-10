if(DEFINED  ANDROID_ABI)
	# set(MNN_LIB ${CMAKE_CURRENT_LIST_DIR}/../MNN_prebuilt/android/${ANDROID_ABI}/libMNN.so)
	add_library(MNN SHARED IMPORTED GLOBAL)
	set_target_properties(
		MNN
		PROPERTIES IMPORTED_LOCATION
		${CMAKE_CURRENT_LIST_DIR}/../MNN_prebuilt/android/${ANDROID_ABI}/lib/libMNN.so
	)
	set(MNN_LIB MNN)
	set(MNN_INC ${CMAKE_CURRENT_LIST_DIR}/../MNN_prebuilt/android/${ANDROID_ABI}/include)
elseif(MSVC_VERSION)
	file(COPY ${CMAKE_CURRENT_LIST_DIR}/../MNN_prebuilt/x64_windows/VS2017/lib/Debug/MNN.dll DESTINATION ${CMAKE_BINARY_DIR}/Debug)
	file(COPY ${CMAKE_CURRENT_LIST_DIR}/../MNN_prebuilt/x64_windows/VS2017/lib/RelWithDebInfo/MNN.dll DESTINATION ${CMAKE_BINARY_DIR}/RelWithDebInfo)
	file(COPY ${CMAKE_CURRENT_LIST_DIR}/../MNN_prebuilt/x64_windows/VS2017/lib/Release/MNN.dll DESTINATION ${CMAKE_BINARY_DIR}/Release)
	file(COPY ${CMAKE_CURRENT_LIST_DIR}/../MNN_prebuilt/x64_windows/VS2017/lib/MinSizeRel/MNN.dll DESTINATION ${CMAKE_BINARY_DIR}/MinSizeRel)
	# file(COPY ${CMAKE_CURRENT_LIST_DIR}/../MNN_prebuilt/x64_windows/MNNd.dll DESTINATION ${CMAKE_BINARY_DIR})
	set(MNN_LIB
		$<$<CONFIG:Debug>:${CMAKE_CURRENT_LIST_DIR}/../MNN_prebuilt/x64_windows/VS2017/lib/Debug/MNN.lib>
		$<$<CONFIG:RelWithDebInfo>:${CMAKE_CURRENT_LIST_DIR}/../MNN_prebuilt/x64_windows/VS2017/lib/RelWithDebInfo/MNN.lib>
		$<$<CONFIG:Release>:${CMAKE_CURRENT_LIST_DIR}/../MNN_prebuilt/x64_windows/VS2017/lib/Release/MNN.lib>
		$<$<CONFIG:MinSizeRel>:${CMAKE_CURRENT_LIST_DIR}/../MNN_prebuilt/x64_windows/VS2017/lib/MinSizeRel/MNN.lib>
	)
	set(MNN_INC ${CMAKE_CURRENT_LIST_DIR}/../MNN_prebuilt/x64_windows/VS2017/include)
else()
	set(MNN_LIB
		$<$<STREQUAL:${BUILD_SYSTEM},x64_linux>:${CMAKE_CURRENT_LIST_DIR}/../MNN_prebuilt/x64_linux/lib/libMNN.so>
		$<$<STREQUAL:${BUILD_SYSTEM},armv7>:${CMAKE_CURRENT_LIST_DIR}/../MNN_prebuilt/armv7/lib/libMNN.so>
		$<$<STREQUAL:${BUILD_SYSTEM},aarch64>:${CMAKE_CURRENT_LIST_DIR}/../MNN_prebuilt/aarch64/lib/libMNN.so>
	)
	set(MNN_INC
		$<$<STREQUAL:${BUILD_SYSTEM},x64_linux>:${CMAKE_CURRENT_LIST_DIR}/../MNN_prebuilt/x64_linux/include>
		$<$<STREQUAL:${BUILD_SYSTEM},armv7>:${CMAKE_CURRENT_LIST_DIR}/../MNN_prebuilt/armv7/include>
		$<$<STREQUAL:${BUILD_SYSTEM},aarch64>:${CMAKE_CURRENT_LIST_DIR}/../MNN_prebuilt/aarch64/include>
	)
endif()
