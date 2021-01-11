# InferenceHelper
- This is a helper class for deep learning frameworks especially for inference
- This class provides an interface to use various deep learnig frameworks, so that you can use the same application code

![Class Diagram](00_doc/class_diagram.png) 

## Supported frameworks
- TensorFlow Lite
- TensorFlow Lite with delegate (XNNPACK, GPU, EdgeTPU)
- TensorRT
- OpenCV(dnn)
- ncnn
- MNN

## Supported targets
- Windows 10 (Visual Studio 2017 x64, Visual Studio 2019 x64)
- Linux (x64, armv7, aarch64)
- Android (armv7, aarch64)

## Tested Environment
| Framework                 | Windows (x64)            | Linux (x64)   | Linux (armv7) | Linux (aarch64)  | Android (aarch64) |
|---------------------------|--------------------------|---------------|---------------|------------------|-------------------|
| OpenCV(dnn)               | OK                       | OK            | OK            | OK               | not tested        |
| TensorFlow Lite           | OK                       | OK            | OK            | OK               | OK                |
| TensorFlow Lite + XNNPACK | OK                       | OK            | OK            | OK               | OK                |
| TensorFlow Lite + GPU     | not supported            | OK            | OK            | OK               | OK                |
| TensorFlow Lite + EdgeTPU | OK                       | not tested    | OK            | OK               | not supported     |
| TensorRT                  | not tested               | not tested    | not tested    | OK               | not supported     |
| ncnn                      | OK                       | OK            | OK            | OK               | OK                |
| MNN                       | OK                       | OK            | OK            | OK               | OK                |
| Note                      | Visual Studio 2017, 2019 | Xubuntu 18.04 | Raspberry Pi  | Jetson Xavier NX | Pixel 4a          |


## Sample project
https://github.com/iwatake2222/InferenceHelper_Sample

## Related projects
- https://github.com/iwatake2222/play_with_tflite
- https://github.com/iwatake2222/play_with_tensorrt
- https://github.com/iwatake2222/play_with_ncnn
- https://github.com/iwatake2222/play_with_mnn

# Usage
## Installation
- Add this repository into your project (Using `git submodule` is recommended)
- Download prebuilt libraries
	- Download prebuilt libraries (ThirdParty.zip) from https://github.com/iwatake2222/InferenceHelper/releases/ 
	- Extract it to `ThirdParty`

### (For Tensorflow Lite)
- After adding or cloning this repository, you need to download header files
	```
	git submodule init
	git submodule update
	cd ThirdParty/tensorflow
	chmod +x tensorflow/lite/tools/make/download_dependencies.sh
	tensorflow/lite/tools/make/download_dependencies.sh
	```

## Project settings in CMake
- Add InferenceHelper and CommonHelper to your project
	```cmake
	set(INFERENCE_HELPER_DIR ${CMAKE_CURRENT_LIST_DIR}/../../InferenceHelper/)
	add_subdirectory(${INFERENCE_HELPER_DIR}/CommonHelper CommonHelper)
	target_include_directories(${LibraryName} PUBLIC ${INFERENCE_HELPER_DIR}/CommonHelper)
	target_link_libraries(${LibraryName} CommonHelper)

	add_subdirectory(${INFERENCE_HELPER_DIR}/InferenceHelper InferenceHelper)
	target_include_directories(${LibraryName} PUBLIC ${INFERENCE_HELPER_DIR}/InferenceHelper)
	target_link_libraries(${LibraryName} InferenceHelper)
	```

## CMake options
- Deep learning framework:
	- You can enable multiple options althoguh the following example enables just one option

	```sh
	# OpenCV (dnn)
	cmake .. -DINFERENCE_HELPER_ENABLE_OPENCV=on
	# Tensorflow Lite
	cmake .. -DINFERENCE_HELPER_ENABLE_TFLITE=on
	# Tensorflow Lite (XNNPACK)
	cmake .. -DINFERENCE_HELPER_ENABLE_TFLITE_DELEGATE_XNNPACK=on
	# Tensorflow Lite (GPU)
	cmake .. -DINFERENCE_HELPER_ENABLE_TFLITE_DELEGATE_GPU=on
	# Tensorflow Lite (EdgeTPU)
	cmake .. -DINFERENCE_HELPER_ENABLE_TFLITE_DELEGATE_EDGETPU=on
	# TensorRT
	cmake .. -DINFERENCE_HELPER_ENABLE_TENSORRT=on
	# ncnn
	cmake .. -DINFERENCE_HELPER_ENABLE_NCNN=on
	# MNN
	cmake .. -DINFERENCE_HELPER_ENABLE_MNN=on
	```

- Enable/Disable preprocess using OpenCV:
	- By disabling this option, InferenceHelper is not dependent on OpenCV
	```sh
	cmake .. -INFERENCE_HELPER_ENABLE_PRE_PROCESS_BY_OPENCV=off
	```

# APIs
## InferenceHelper
### Enumeration
```c++
typedef enum {
	OPEN_CV,
	OPEN_CV_GPU,
	TENSORFLOW_LITE,
	TENSORFLOW_LITE_XNNPACK,
	TENSORFLOW_LITE_GPU,
	TENSORFLOW_LITE_EDGETPU,
	TENSOR_RT,
	NCNN,
	MNN,
} HELPER_TYPE;
```

### static InferenceHelper* create(const HELPER_TYPE typeFw)
- Create InferenceHelper instance for the selected framework

```c++
std::unique_ptr<InferenceHelper> inferenceHelper(InferenceHelper::create(InferenceHelper::TENSORFLOW_LITE));
```

### static void preProcessByOpenCV(const InputTensorInfo& inputTensorInfo, bool isNCHW, cv::Mat& imgBlob)
- Run preprocess (convert image to blob(NCHW or NHWC))
- This is just a helper function. You may not use this function.
	- Available when `INFERENCE_HELPER_ENABLE_PRE_PROCESS_BY_OPENCV=on`

```c++
InferenceHelper::preProcessByOpenCV(inputTensorInfo, false, imgSrc);
```

### int32_t setNumThread(const int32_t numThread)
- Set the number of threads to be used
- This function needs to be called before initialize

```c++
inferenceHelper->setNumThread(4);
```

### int32_t setCustomOps(const std::vector<std::pair<const char*, const void*>>& customOps)
- Set custom ops
- This function needs to be called before initialize

```c++
std::vector<std::pair<const char*, const void*>> customOps;
customOps.push_back(std::pair<const char*, const void*>("Convolution2DTransposeBias", (const void*)mediapipe::tflite_operations::RegisterConvolution2DTransposeBias()));
inferenceHelper->setCustomOps(customOps);
```

### int32_t initialize(const std::string& modelFilename, std::vector<InputTensorInfo>& inputTensorInfoList, std::vector<OutputTensorInfo>& outputTensorInfoList)
- Initialize inference helper
	- Load model
	- Set tensor information

```c++
std::vector<InputTensorInfo> inputTensorList;
InputTensorInfo inputTensorInfo;
inputTensorInfo.name = "input";
inputTensorInfo.tensorType = TensorInfo::TENSOR_TYPE_FP32;
inputTensorInfo.tensorDims.batch = 1;
inputTensorInfo.tensorDims.width = 224;
inputTensorInfo.tensorDims.height = 224;
inputTensorInfo.tensorDims.channel = 3;
inputTensorInfo.data = imgSrc.data;
inputTensorInfo.dataType = InputTensorInfo::DATA_TYPE_IMAGE;
inputTensorInfo.imageInfo.width = imgSrc.cols;
inputTensorInfo.imageInfo.height = imgSrc.rows;
inputTensorInfo.imageInfo.channel = imgSrc.channels();
inputTensorInfo.imageInfo.cropX = 0;
inputTensorInfo.imageInfo.cropY = 0;
inputTensorInfo.imageInfo.cropWidth = imgSrc.cols;
inputTensorInfo.imageInfo.cropHeight = imgSrc.rows;
inputTensorInfo.imageInfo.isBGR = false;
inputTensorInfo.imageInfo.swapColor = false;
inputTensorInfo.normalize.mean[0] = 0.485f;
inputTensorInfo.normalize.mean[1] = 0.456f;
inputTensorInfo.normalize.mean[2] = 0.406f;
inputTensorInfo.normalize.norm[0] = 0.229f;
inputTensorInfo.normalize.norm[1] = 0.224f;
inputTensorInfo.normalize.norm[2] = 0.225f;
inputTensorList.push_back(inputTensorInfo);

std::vector<OutputTensorInfo> outputTensorList;
OutputTensorInfo outputTensorInfo;
outputTensorInfo.name = "MobilenetV2/Predictions/Reshape_1";
outputTensorInfo.tensorType = TensorInfo::TENSOR_TYPE_FP32;
outputTensorList.push_back(outputTensorInfo);

inferenceHelper->initialize("mobilenet_v2_1.0_224.tflite", inputTensorList, outputTensorList);
```

### int32_t finalize(void)
- Finalize inference helper

```c++
inferenceHelper->finalize();
```

### int32_t preProcess(const std::vector<InputTensorInfo>& inputTensorInfoList)
- Run preprocess
- Call this function before invoke
- Call this function even if the input data is already pre-processed in order to copy data to memory
- **Note** : Some frameworks don't support crop, resize. So, it's better to resize image before calling preProcess.

```c++
inferenceHelper->preProcess(inputTensorList);
```

### int32_t invoke(std::vector<OutputTensorInfo>& outputTensorInfoList)
- Run inference

```c++
inferenceHelper->invoke(outputTensorList)
```

## TensorInfo (InputTensorInfo, OutputTensorInfo)
### Enumeration
```c++
enum {
	TENSOR_TYPE_NONE,
	TENSOR_TYPE_UINT8,
	TENSOR_TYPE_FP32,
	TENSOR_TYPE_INT32,
	TENSOR_TYPE_INT64,
};
```

### Properties
```c++
std::string name;			// [In] Set the name of tensor
int32_t     id;				// [Out] Do not modify (Used in InferenceHelper)
int32_t     tensorType;		// [In] The type of tensor (e.g. TENSOR_TYPE_FP32)
struct {
	int32_t batch;   // 0
	int32_t width;   // 1
	int32_t height;  // 2
	int32_t channel; // 3
} tensorDims;				// InputTensorInfo: [In] The dimentions of tensor. (If -1 is set at initialize, the size is updated from model info.)
								// OutputTensorInfo: [Out] The dimentions of tensor is set from model information
```

## InputTensorInfo
### Enumeration
```c++
enum {
	DATA_TYPE_IMAGE,
	DATA_TYPE_BLOB_NHWC,	// data which already finished preprocess(color conversion, resize, normalize, etc.)
	DATA_TYPE_BLOB_NCHW,
};
```

### Properties
```c++
void*   data;		// [In] Set the pointer to image/blob
int32_t dataType;	// [In] Set the type of data (e.g. DATA_TYPE_IMAGE)
struct {
	int32_t width;
	int32_t height;
	int32_t channel;
	int32_t cropX;
	int32_t cropY;
	int32_t cropWidth;
	int32_t cropHeight;
	bool    isBGR;        // used when channel == 3 (true: BGR, false: RGB)
	bool    swapColor;
} imageInfo;              // [In] used when dataType == DATA_TYPE_IMAGE
struct {
	float mean[3];
	float norm[3];
} normalize;              // [In] used when dataType == DATA_TYPE_IMAGE
```


## OutputTensorInfo
### Properties
```c++
void* data;				// [Out] Pointer to the output data
struct {
	float   scale;
	uint8_t zeroPoint;
} quant;				// [Out] Parameters for dequantization (convert uint8 to float)
```

### float* getDataAsFloat()
- Get output data in the form of FP32
- When tensor type is INT8 (quantized), the data is converted to FP32 (dequantized)

```c++
const float* valFloat = outputTensorList[0].getDataAsFloat();
```

# License
- InferenceHelper
- https://github.com/iwatake2222/InferenceHelper
- Copyright 2020 iwatake2222
- Licensed under the Apache License, Version 2.0

# Acknowledgements
- This project utilizes OSS (Open Source Software)
	- [NOTICE.md](NOTICE.md)
