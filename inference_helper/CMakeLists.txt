cmake_minimum_required(VERSION 3.0)

set(LibraryName "InferenceHelper")
set(THIRD_PARTY_DIR ${CMAKE_CURRENT_LIST_DIR}/../third_party/)

set(INFERENCE_HELPER_ENABLE_PRE_PROCESS_BY_OPENCV off CACHE BOOL "Enable PreProcess by OpenCV? [on/off]")
set(INFERENCE_HELPER_ENABLE_OPENCV off CACHE BOOL "With OpenCV? [on/off]")
set(INFERENCE_HELPER_ENABLE_TFLITE off CACHE BOOL "With Tflite? [on/off]")
set(INFERENCE_HELPER_ENABLE_TFLITE_DELEGATE_XNNPACK off CACHE BOOL "With Tflite Delegate XNNPACK? [on/off]")
set(INFERENCE_HELPER_ENABLE_TFLITE_DELEGATE_GPU off CACHE BOOL "With Tflite Delegate GPU? [on/off]")
set(INFERENCE_HELPER_ENABLE_TFLITE_DELEGATE_EDGETPU off CACHE BOOL "With Tflite Delegate EdgeTPU? [on/off]")
set(INFERENCE_HELPER_ENABLE_TFLITE_DELEGATE_NNAPI off CACHE BOOL "With Tflite Delegate NNAPI? [on/off]")
set(INFERENCE_HELPER_ENABLE_TENSORRT off CACHE BOOL "With TensorRT? [on/off]")
set(INFERENCE_HELPER_ENABLE_NCNN off CACHE BOOL "With Ncnn? [on/off]")
set(INFERENCE_HELPER_ENABLE_MNN off CACHE BOOL "With Mnn? [on/off]")
set(INFERENCE_HELPER_ENABLE_SNPE off CACHE BOOL "With SNPE? [on/off]")

# Create library
set(SRC inference_helper.h inference_helper.cpp)

if(INFERENCE_HELPER_ENABLE_OPENCV)
    set(SRC ${SRC} inference_helper_opencv.h inference_helper_opencv.cpp)
endif()

if(INFERENCE_HELPER_ENABLE_TFLITE OR INFERENCE_HELPER_ENABLE_TFLITE_DELEGATE_XNNPACK OR INFERENCE_HELPER_ENABLE_TFLITE_DELEGATE_GPU OR INFERENCE_HELPER_ENABLE_TFLITE_DELEGATE_EDGETPU OR INFERENCE_HELPER_ENABLE_TFLITE_DELEGATE_NNAPI)
    set(SRC ${SRC} inference_helper_tensorflow_lite.h inference_helper_tensorflow_lite.cpp)
endif()

if(INFERENCE_HELPER_ENABLE_TENSORRT)
    set(SRC ${SRC} inference_helper_tensorrt.h inference_helper_tensorrt.cpp)
    set(SRC ${SRC} tensorrt/logger.cpp tensorrt/BatchStream.h tensorrt/common.h tensorrt/EntropyCalibrator.h tensorrt/logger.h tensorrt/logging.h)
endif()

if(INFERENCE_HELPER_ENABLE_NCNN)
    set(SRC ${SRC} inference_helper_ncnn.h inference_helper_ncnn.cpp)
endif()

if(INFERENCE_HELPER_ENABLE_MNN)
    set(SRC ${SRC} inference_helper_mnn.h inference_helper_mnn.cpp)
endif()

if(INFERENCE_HELPER_ENABLE_SNPE)
	set(SRC ${SRC} inference_helper_snpe.h inference_helper_snpe.cpp)
	set(SRC ${SRC} snpe/CreateUserBuffer.cpp snpe/CreateUserBuffer.hpp snpe/Util.cpp snpe/Util.hpp snpe/udlExample.cpp snpe/udlExample.hpp)
endif()

add_library(${LibraryName} ${SRC})

add_subdirectory(../common_helper common_helper)
target_include_directories(${LibraryName} PUBLIC ../common_helper)
target_link_libraries(${LibraryName} CommonHelper)

# For TensorInfo (Pre process calculation)
if(INFERENCE_HELPER_ENABLE_PRE_PROCESS_BY_OPENCV)
    find_package(OpenCV REQUIRED)
    target_include_directories(${LibraryName} PUBLIC ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(${LibraryName} ${OpenCV_LIBS})
    add_definitions(-DINFERENCE_HELPER_ENABLE_PRE_PROCESS_BY_OPENCV)
endif()

# For OpenCV
if(INFERENCE_HELPER_ENABLE_OPENCV)
    find_package(OpenCV REQUIRED)
    target_include_directories(${LibraryName} PUBLIC ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(${LibraryName} ${OpenCV_LIBS})
    add_definitions(-DINFERENCE_HELPER_ENABLE_OPENCV)
endif()

# For Tensorflow Lite
if(INFERENCE_HELPER_ENABLE_TFLITE OR INFERENCE_HELPER_ENABLE_TFLITE_DELEGATE_XNNPACK OR INFERENCE_HELPER_ENABLE_TFLITE_DELEGATE_GPU OR INFERENCE_HELPER_ENABLE_TFLITE_DELEGATE_EDGETPU OR INFERENCE_HELPER_ENABLE_TFLITE_DELEGATE_NNAPI)
    include(${THIRD_PARTY_DIR}/cmakes/tflite.cmake)
    target_include_directories(${LibraryName} PUBLIC ${TFLITE_INC})
    target_link_libraries(${LibraryName} ${TFLITE_LIB})
    add_definitions(-DINFERENCE_HELPER_ENABLE_TFLITE)
endif()

# For Tensorflow Lite Delegate(XNNPACK)
if(INFERENCE_HELPER_ENABLE_TFLITE_DELEGATE_XNNPACK)
    add_definitions(-DINFERENCE_HELPER_ENABLE_TFLITE_DELEGATE_XNNPACK)
endif()

# For Tensorflow Lite Delegate(GPU)
if(INFERENCE_HELPER_ENABLE_TFLITE_DELEGATE_GPU)
    find_package(OpenCL)
    if(OpenCL_Found)
        target_include_directories(${LibraryName} PUBLIC ${OpenCL_INCLUDE_DIRS})
        target_link_libraries(${LibraryName} ${OpenCL_LIBRARIES})
    endif()
    include(${THIRD_PARTY_DIR}/cmakes/tflite_gpu.cmake)
    target_include_directories(${LibraryName} PUBLIC ${TFLITE_GPU_INC})
    target_link_libraries(${LibraryName} ${TFLITE_GPU_LIB} EGL GLESv2)
    add_definitions(-DINFERENCE_HELPER_ENABLE_TFLITE_DELEGATE_GPU)
endif()

# For Tensorflow Lite Delegate(Edge TPU)
if(INFERENCE_HELPER_ENABLE_TFLITE_DELEGATE_EDGETPU)
    include(${THIRD_PARTY_DIR}/cmakes/tflite_edgetpu.cmake)
    target_include_directories(${LibraryName} PUBLIC ${TFLITE_EDGETPU_INC})
    target_link_libraries(${LibraryName} ${TFLITE_EDGETPU_LIB})
    add_definitions(-DINFERENCE_HELPER_ENABLE_TFLITE_DELEGATE_EDGETPU)
endif()

# For Tensorflow Lite Delegate(NNAPI)
if(INFERENCE_HELPER_ENABLE_TFLITE_DELEGATE_NNAPI)
    add_definitions(-DINFERENCE_HELPER_ENABLE_TFLITE_DELEGATE_NNAPI)
endif()


# For TensorRT
if(INFERENCE_HELPER_ENABLE_TENSORRT)
    find_package(CUDA)
    if(CUDA_FOUND)
        target_link_libraries(${LibraryName}
            ${CUDA_LIBRARIES}
            nvinfer
            nvonnxparser
            nvinfer_plugin
            cudnn
        )
        target_include_directories(${LibraryName} PUBLIC
            ${CUDA_INCLUDE_DIRS}
            tensorrt
        )
        add_definitions(-DINFERENCE_HELPER_ENABLE_TENSORRT)
        message("CUDA_INCLUDE_DIRS: ${CUDA_INCLUDE_DIRS}")
    else()
        message(WARNING, "Cannot find CUDA")
    endif()
endif()

# For NCNN
if(INFERENCE_HELPER_ENABLE_NCNN)
    include(${THIRD_PARTY_DIR}/cmakes/ncnn.cmake)
    target_include_directories(${LibraryName} PUBLIC ${NCNN_INC})
    target_link_libraries(${LibraryName} ${NCNN_LIB})
    add_definitions(-DINFERENCE_HELPER_ENABLE_NCNN)
endif()

# For MNN
if(INFERENCE_HELPER_ENABLE_MNN)
    include(${THIRD_PARTY_DIR}/cmakes/mnn.cmake)
    target_include_directories(${LibraryName} PUBLIC ${MNN_INC})
    target_link_libraries(${LibraryName} ${MNN_LIB})
    add_definitions(-DINFERENCE_HELPER_ENABLE_MNN)
endif()

# For SNPE
if(INFERENCE_HELPER_ENABLE_SNPE)
	include(${THIRD_PARTY_DIR}/cmakes/snpe.cmake)
	target_include_directories(${LibraryName} PUBLIC ${SNPE_INC} ./snpe)
	target_link_libraries(${LibraryName} ${SNPE_LIB})
	add_definitions(-DINFERENCE_HELPER_ENABLE_SNPE)
endif()

