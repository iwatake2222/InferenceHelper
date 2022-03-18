if(DEFINED ANDROID_ABI)
elseif(MSVC_VERSION)
else()
    if(${BUILD_SYSTEM} STREQUAL "armv7")
    elseif(${BUILD_SYSTEM} STREQUAL "aarch64")
    else()
    endif()
endif()

if(DEFINED ANDROID_ABI)
else()
    if(MSVC_VERSION)
    else()
    endif()
endif()

# set the following variables
set(sample_LIB libsample.so)
set(sample_INC "./")
