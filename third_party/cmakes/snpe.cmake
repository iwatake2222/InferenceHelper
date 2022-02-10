set(SNPE_INC 
	${CMAKE_CURRENT_LIST_DIR}/../snpe_prebuilt/include/zdl
)

if(DEFINED  ANDROID_ABI)
	# Copy libSNPE.so to app/jniExternalLibs/arm64-v8a
	# if(${ANDROID_ABI} STREQUAL "arm64-v8a")
	# 	set(SNPE_LIB ${CMAKE_CURRENT_LIST_DIR}/../snpe_prebuilt/lib/aarch64-android-clang6.0/libSNPE.so)
	# elseif(${ANDROID_ABI} STREQUAL "armeabi-v7a")
	# 	set(SNPE_LIB ${CMAKE_CURRENT_LIST_DIR}/../snpe_prebuilt/lib/arm-android-clang6.0/libSNPE.so)
	# else()
	# 	message(FATAL_ERROR "other")
	# endif()

	add_library(SNPE SHARED IMPORTED GLOBAL)
	if(${ANDROID_ABI} STREQUAL "arm64-v8a")
		set_target_properties(
			SNPE
			PROPERTIES IMPORTED_LOCATION
			${CMAKE_CURRENT_LIST_DIR}/../snpe_prebuilt/lib/aarch64-android-clang6.0/libSNPE.so
		)
	elseif(${ANDROID_ABI} STREQUAL "armeabi-v7a")
		set_target_properties(
			SNPE
			PROPERTIES IMPORTED_LOCATION
			${CMAKE_CURRENT_LIST_DIR}/../snpe_prebuilt/lib/arm-android-clang6.0/libSNPE.so
		)
	else()
		message(FATAL_ERROR "other")
	endif()
	set(SNPE_LIB SNPE)
elseif(MSVC_VERSION)
	message(FATAL_ERROR "[SNPE] unsupported platform")
else()
	if(${BUILD_SYSTEM} STREQUAL "x64_linux")
		set(SNPE_PATH ${CMAKE_CURRENT_LIST_DIR}/../snpe_prebuilt/lib/x86_64-linux-clang/)
	elseif(${BUILD_SYSTEM} STREQUAL "armv7")
		message(FATAL_ERROR "[SNPE] unsupported platform")
	elseif(${BUILD_SYSTEM} STREQUAL "aarch64")
		set(SNPE_PATH ${CMAKE_CURRENT_LIST_DIR}/../snpe_prebuilt/lib/aarch64-linux-gcc4.9/)
	else()	
		message(FATAL_ERROR "[SNPE] unsupported platform")
	endif()
	file(GLOB LIB_FILES ${SNPE_PATH}/*)
	file(COPY ${LIB_FILES} DESTINATION ${CMAKE_BINARY_DIR})
	file(GLOB LIB_FILES ${SNPE_PATH}/../dsp/*)
	file(COPY ${LIB_FILES} DESTINATION ${CMAKE_BINARY_DIR})
	set(SNPE_LIB ${CMAKE_BINARY_DIR}/libSNPE.so)
endif()
