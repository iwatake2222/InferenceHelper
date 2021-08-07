# InferenceHelper
<p align="center">
  <img src="00_doc/logo.png" />
</p>

- This is a helper class for deep learning frameworks especially for inference
- This class provides an interface to use various deep learnig frameworks, so that you can use the same application code

![GitHub](https://img.shields.io/badge/c++-%2300599C.svg?style=for-the-badge&logo=c%2B%2B&logoColor=white)
![GitHub](https://img.shields.io/github/license/iwatake2222/InferenceHelper)

![Class Diagram](00_doc/class_diagram.png) 

## Supported frameworks
- TensorFlow Lite
- TensorFlow Lite with delegate (XNNPACK, GPU, EdgeTPU, NNAPI)
- TensorRT (GPU, DLA)
- OpenCV(dnn)
- ncnn
- MNN
- SNPE (Snapdragon Neural Processing Engine SDK (Qualcomm Neural Processing SDK for AI v1.51.0))
- Arm NN

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
| TensorFlow Lite + NNAPI   | not supported            | not supported | not supported | not supported    | OK                |
| TensorRT                  | not tested               | not tested    | not tested    | OK               | not supported     |
| ncnn                      | OK                       | OK            | OK            | OK               | OK                |
| MNN                       | OK                       | OK            | OK            | OK               | OK                |
| SNPE                      | not supported            | not supported | not tested    | OK               | OK                |
| Arm NN                    | not supported            | OK            | not supported | OK               | not supported     |
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
    - Download prebuilt libraries (third_party.zip) from https://github.com/iwatake2222/InferenceHelper/releases/ 
    - Extract it to `third_party`

### For Tensorflow Lite
- After adding or cloning this repository, you need to download header files
    ```
    git submodule init
    git submodule update
    cd third_party/tensorflow
    chmod +x tensorflow/lite/tools/make/download_dependencies.sh
    tensorflow/lite/tools/make/download_dependencies.sh
    ```

### For SNPE
- After adding or cloning this repository, you need to download library from https://developer.qualcomm.com/software/qualcomm-neural-processing-sdk/tools
- Extract `snpe-1.51.0.zip` , then place `lib` and `include` folders to `third_party/snpe`

## Project settings in CMake
- Add InferenceHelper and CommonHelper to your project
    ```cmake
    set(INFERENCE_HELPER_DIR ${CMAKE_CURRENT_LIST_DIR}/../../InferenceHelper/)
    # add_subdirectory(${INFERENCE_HELPER_DIR}/common_helper common_helper)
    target_include_directories(${LibraryName} PUBLIC ${INFERENCE_HELPER_DIR}/common_helper)
    target_link_libraries(${LibraryName} CommonHelper)

    add_subdirectory(${INFERENCE_HELPER_DIR}/inference_helper inference_helper)
    target_include_directories(${LibraryName} PUBLIC ${INFERENCE_HELPER_DIR}/inference_helper)
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
    # Tensorflow Lite (NNAPI)
    cmake .. -DINFERENCE_HELPER_ENABLE_TFLITE_DELEGATE_NNAPI=on
    # TensorRT
    cmake .. -DINFERENCE_HELPER_ENABLE_TENSORRT=on
    # ncnn
    cmake .. -DINFERENCE_HELPER_ENABLE_NCNN=on
    # MNN
    cmake .. -DINFERENCE_HELPER_ENABLE_MNN=on
    # SNPE
    cmake .. -DINFERENCE_HELPER_ENABLE_SNPE=on
    # Arm NN
    cmake .. -DINFERENCE_HELPER_ENABLE_ARMNN=on
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
    kOpencv,
    kOpencvGpu,
    kTensorflowLite,
    kTensorflowLiteXnnpack,
    kTensorflowLiteGpu,
    kTensorflowLiteEdgetpu,
    kTensorflowLiteNnapi,
    kTensorrt,
    kNcnn,
    kMnn,
    kSnpe,
    kArmnn,
} HelperType;
```

### static InferenceHelper* Create(const HelperType helper_type)
- Create InferenceHelper instance for the selected framework

```c++
std::unique_ptr<InferenceHelper> inference_helper(InferenceHelper::Create(InferenceHelper::kTensorflowLite));
```

### static void PreProcessByOpenCV(const InputTensorInfo& input_tensor_info, bool is_nchw, cv::Mat& img_blob)
- Run preprocess (convert image to blob(NCHW or NHWC))
- This is just a helper function. You may not use this function.
    - Available when `INFERENCE_HELPER_ENABLE_PRE_PROCESS_BY_OPENCV=on`

```c++
InferenceHelper::PreProcessByOpenCV(input_tensor_info, false, img_blob);
```

### int32_t SetNumThreads(const int32_t num_threads)
- Set the number of threads to be used
- This function needs to be called before initialize

```c++
inference_helper->SetNumThreads(4);
```

### int32_t SetCustomOps(const std::vector<std::pair<const char*, const void*>>& custom_ops)
- Set custom ops
- This function needs to be called before initialize

```c++
std::vector<std::pair<const char*, const void*>> custom_ops;
custom_ops.push_back(std::pair<const char*, const void*>("Convolution2DTransposeBias", (const void*)mediapipe::tflite_operations::RegisterConvolution2DTransposeBias()));
inference_helper->SetCustomOps(custom_ops);
```

### int32_t Initialize(const std::string& model_filename, std::vector<InputTensorInfo>& input_tensor_info_list, std::vector<OutputTensorInfo>& output_tensor_info_list)
- Initialize inference helper
    - Load model
    - Set tensor information

```c++
std::vector<InputTensorInfo> input_tensor_list;
InputTensorInfo input_tensor_info("input", TensorInfo::TENSOR_TYPE_FP32, false);    /* name, data_type, NCHW or NHWC */
input_tensor_info.tensor_dims = { 1, 224, 224, 3 };
input_tensor_info.data_type = InputTensorInfo::kDataTypeImage;
input_tensor_info.data = img_src.data;
input_tensor_info.image_info.width = img_src.cols;
input_tensor_info.image_info.height = img_src.rows;
input_tensor_info.image_info.channel = img_src.channels();
input_tensor_info.image_info.crop_x = 0;
input_tensor_info.image_info.crop_y = 0;
input_tensor_info.image_info.crop_width = img_src.cols;
input_tensor_info.image_info.crop_height = img_src.rows;
input_tensor_info.image_info.is_bgr = false;
input_tensor_info.image_info.swap_color = false;
input_tensor_info.normalize.mean[0] = 0.485f;   /* https://github.com/onnx/models/tree/master/vision/classification/mobilenet#preprocessing */
input_tensor_info.normalize.mean[1] = 0.456f;
input_tensor_info.normalize.mean[2] = 0.406f;
input_tensor_info.normalize.norm[0] = 0.229f;
input_tensor_info.normalize.norm[1] = 0.224f;
input_tensor_info.normalize.norm[2] = 0.225f;
input_tensor_list.push_back(input_tensor_info);

std::vector<OutputTensorInfo> output_tensor_list;
output_tensor_list.push_back(OutputTensorInfo("MobilenetV2/Predictions/Reshape_1", TensorInfo::TENSOR_TYPE_FP32));

inference_helper->initialize("mobilenet_v2_1.0_224.tflite", input_tensor_list, output_tensor_list);
```

### int32_t Finalize(void)
- Finalize inference helper

```c++
inference_helper->Finalize();
```

### int32_t PreProcess(const std::vector<InputTensorInfo>& input_tensor_info_list)
- Run preprocess
- Call this function before invoke
- Call this function even if the input data is already pre-processed in order to copy data to memory
- **Note** : Some frameworks don't support crop, resize. So, it's better to resize image before calling preProcess.

```c++
inference_helper->PreProcess(input_tensor_list);
```

### int32_t Process(std::vector<OutputTensorInfo>& output_tensor_info_list)
- Run inference

```c++
inference_helper->Process(output_tensor_info_list)
```

## TensorInfo (InputTensorInfo, OutputTensorInfo)
### Enumeration
```c++
enum {
    kTensorTypeNone,
    kTensorTypeUint8,
    kTensorTypeInt8,
    kTensorTypeFp32,
    kTensorTypeInt32,
    kTensorTypeInt64,
};
```

### Properties
```c++
std::string name;           // [In] Set the name_ of tensor
int32_t     id;             // [Out] Do not modify (Used in InferenceHelper)
int32_t     tensor_type;    // [In] The type of tensor (e.g. kTensorTypeFp32)
std::vector<int32_t> tensor_dims;    // InputTensorInfo:   [In] The dimentions of tensor. (If empty at initialize, the size is updated from model info.)
                                     // OutputTensorInfo: [Out] The dimentions of tensor is set from model information
bool        is_nchw;        // [IN] NCHW or NHWC

```

## InputTensorInfo
### Enumeration
```c++
enum {
    kDataTypeImage,
    kDataTypeBlobNhwc,  // data_ which already finished preprocess(color conversion, resize, normalize_, etc.)
    kDataTypeBlobNchw,
};
```

### Properties
```c++
void*   data;      // [In] Set the pointer to image/blob
int32_t data_type; // [In] Set the type of data_ (e.g. kDataTypeImage)

struct {
    int32_t width;
    int32_t height;
    int32_t channel;
    int32_t crop_x;
    int32_t crop_y;
    int32_t crop_width;
    int32_t crop_height;
    bool    is_bgr;        // used when channel == 3 (true: BGR, false: RGB)
    bool    swap_color;
} image_info;              // [In] used when data_type_ == kDataTypeImage

struct {
    float mean[3];
    float norm[3];
} normalize;              // [In] used when data_type_ == kDataTypeImage
```


## OutputTensorInfo
### Properties
```c++
void* data;     // [Out] Pointer to the output data_
struct {
    float   scale;
    uint8_t zero_point;
} quant;        // [Out] Parameters for dequantization (convert uint8 to float)
```

### float* GetDataAsFloat()
- Get output data in the form of FP32
- When tensor type is INT8 (quantized), the data is converted to FP32 (dequantized)

```c++
const float* val_float = output_tensor_list[0].GetDataAsFloat();
```

# License
- InferenceHelper
- https://github.com/iwatake2222/InferenceHelper
- Copyright 2020 iwatake2222
- Licensed under the Apache License, Version 2.0

# Acknowledgements
- This project utilizes OSS (Open Source Software)
    - [NOTICE.md](NOTICE.md)
