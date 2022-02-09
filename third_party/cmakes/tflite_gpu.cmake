set(TFLITE_GPU_INC
    ${CMAKE_CURRENT_LIST_DIR}/../tensorflow
    ${CMAKE_CURRENT_LIST_DIR}/../tensorflow_deps/abseil-cpp
    ${CMAKE_CURRENT_LIST_DIR}/../tensorflow_deps/flatbuffers/include
)

if(DEFINED  ANDROID_ABI)
    # set(TFLITE_GPU_LIB ${CMAKE_CURRENT_LIST_DIR}/../tensorflow_prebuilt/android/${ANDROID_ABI}/libtensorflowlite_gpu_delegate.so)
    add_library(TFLITE_GPU SHARED IMPORTED GLOBAL)
    set_target_properties(
        TFLITE_GPU
        PROPERTIES IMPORTED_LOCATION
        ${CMAKE_CURRENT_LIST_DIR}/../tflite_prebuilt/android/${ANDROID_ABI}/libtensorflowlite_gpu_delegate.so
    )
    set(TFLITE_GPU_LIB TFLITE_GPU)
else()
    message(FATAL_ERROR "[tflite_gpu] unsupported platform")
endif()
