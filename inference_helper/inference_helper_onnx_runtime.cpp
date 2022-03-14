/* Copyright 2022 iwatake2222

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
/*** Include ***/
/* for general */
#include <cstdint>
#include <cstdlib>
#include <cinttypes>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>
#include <array>
#include <algorithm>
#include <chrono>

#ifdef _WIN32
#include <atlstr.h>
#endif

/* for ONNX Runtime */
#include <onnxruntime_cxx_api.h>
//#ifdef INFERENCE_HELPER_ENABLE_ONNX_RUNTIME_CUDA
//#include <cuda_provider_factory.h>
//#endif

/* for My modules */
#include "inference_helper_log.h"
#include "inference_helper_onnx_runtime.h"

/*** Macro ***/
#define TAG "InferenceHelperOnnxRuntime"
#define PRINT(...)   INFERENCE_HELPER_LOG_PRINT(TAG, __VA_ARGS__)
#define PRINT_E(...) INFERENCE_HELPER_LOG_PRINT_E(TAG, __VA_ARGS__)

static constexpr const char* DATA_TYPE_ID_TO_NAME_MAP[] = {
    nullptr,   // ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED
    "float32", // ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
    "uint8",   // ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8
    "int8",    // ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8
    "uint16",  // ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16
    "int16",   // ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16
    "int32",   // ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32
    "int64",   // ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64
    "string",  // ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING
    "bool",    // ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL
    "float16", // ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16
    "float64", // ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE
    "uint32",  // ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32
    "uint64",  // ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64
    nullptr,   // ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64
    nullptr,   // ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128
    nullptr    // ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16
};


/*** Function ***/
static void PrintTensorInfo(const Ort::TypeInfo& info, size_t index, const char* name)
{
    auto shape_info = info.GetTensorTypeAndShapeInfo();
    ONNXTensorElementDataType element_type = shape_info.GetElementType();
    const char* element_type_str = DATA_TYPE_ID_TO_NAME_MAP[element_type];
    size_t element_count = shape_info.GetElementCount();
    std::vector<int64_t> shape = shape_info.GetShape();
    PRINT("    info[%zu]->name: %s\n", index, name);
    PRINT("    info[%zu]->element_type: %s\n", index, element_type_str);
    PRINT("    info[%zu]->element_count: %zu\n", index, element_count);
    for (size_t j = 0; j < shape.size(); j++) {
        PRINT("    info[%zu]->shape[%zu]: %" PRId64 "\n", index, j, shape[j]);
    }
}

static void DisplayModelInfo(const Ort::Session& session)
{
    Ort::AllocatorWithDefaultOptions ort_alloc;
    size_t input_num = session.GetInputCount();
    PRINT("Input num = %zu\n", input_num);
    for (size_t i = 0; i < input_num; i++) {
        char* name = session.GetInputName(i, ort_alloc);
        Ort::TypeInfo info = session.GetInputTypeInfo(i);
        PrintTensorInfo(info, i, name);
        ort_alloc.Free(name);
    }

    size_t output_num = session.GetOutputCount();
    PRINT("Output num = %zu\n", output_num);
    for (size_t i = 0; i < output_num; i++) {
        char* name = session.GetOutputName(i, ort_alloc);
        Ort::TypeInfo info = session.GetOutputTypeInfo(i);
        PrintTensorInfo(info, i, name);
        ort_alloc.Free(name);
    }
}

InferenceHelperOnnxRuntime::InferenceHelperOnnxRuntime()
{
    num_threads_ = 1;
}

InferenceHelperOnnxRuntime::~InferenceHelperOnnxRuntime()
{
}

int32_t InferenceHelperOnnxRuntime::SetNumThreads(const int32_t num_threads)
{
    num_threads_ = num_threads;
    return kRetOk;
}

int32_t InferenceHelperOnnxRuntime::SetCustomOps(const std::vector<std::pair<const char*, const void*>>& custom_ops)
{
    PRINT("[WARNING] This method is not supported\n");
    return kRetOk;
}

int32_t InferenceHelperOnnxRuntime::Initialize(const std::string& model_filename, std::vector<InputTensorInfo>& input_tensor_info_list, std::vector<OutputTensorInfo>& output_tensor_info_list)
{
    /*** Create session ***/
    Ort::SessionOptions session_options;
    session_options.SetInterOpNumThreads(1);            /* todo: need to check documents */
    session_options.SetIntraOpNumThreads(num_threads_);
    
#ifdef INFERENCE_HELPER_ENABLE_ONNX_RUNTIME_CUDA
    if (helper_type_ == kOnnxRuntimeCuda) {
        try {
            Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0));
        } catch (std::exception& e) {
            PRINT_E("[ERROR] OrtSessionOptionsAppendExecutionProvider_CUDA: %s\n", e.what());
            return kRetErr;
        }
    }
#endif

#ifdef _WIN32
    CStringW model_filename_cstringw(model_filename.c_str());
    auto onnx_model_filename_pcxstr = model_filename_cstringw.GetString();
#else
    auto onnx_model_filename_pcxstr = model_filename.c_str();
#endif
    try {
        session_ = Ort::Session(env_, onnx_model_filename_pcxstr, session_options);
    } catch (std::exception& e) {
        PRINT_E("[ERROR] Unable to create session for %s: %s\n", model_filename.c_str(), e.what());
        return kRetErr;
    }

    DisplayModelInfo(session_);

    /*** Allocate Tensors ***/
    size_t input_num = session_.GetInputCount();
    for (size_t i = 0; i < input_num; i++) {
        if (AllocateTensor(true, i, input_tensor_info_list, output_tensor_info_list)) {
            PRINT_E("Input tensor %zu is not allocated\n", i);
            return kRetErr;
        }
    }
    size_t output_num = session_.GetOutputCount();
    for (size_t i = 0; i < output_num; i++) {
        if (AllocateTensor(false, i, input_tensor_info_list, output_tensor_info_list)) {
            PRINT_E("Output tensor %zu is not allocated\n", i);
            return kRetErr;
        }
    }

    /*** Convert normalize parameter to speed up ***/
    for (auto& input_tensor_info : input_tensor_info_list) {
        ConvertNormalizeParameters(input_tensor_info);
    }

    return kRetOk;
};

int32_t InferenceHelperOnnxRuntime::Finalize(void)
{
    for (auto& tensor : input_tensor_list_) {
        Ort::OrtRelease(tensor.release());
    }
    for (auto& tensor : output_tensor_list_) {
        Ort::OrtRelease(tensor.release());
    }
    Ort::OrtRelease(session_.release());

    return kRetOk;
}

int32_t InferenceHelperOnnxRuntime::PreProcess(const std::vector<InputTensorInfo>& input_tensor_info_list)
{
    for (const auto& input_tensor_info : input_tensor_info_list) {
        const int32_t img_width = input_tensor_info.GetWidth();
        const int32_t img_height = input_tensor_info.GetHeight();
        const int32_t img_channel = input_tensor_info.GetChannel();
        if (input_tensor_info.data_type == InputTensorInfo::kDataTypeImage) {
            if ((input_tensor_info.image_info.width != input_tensor_info.image_info.crop_width) || (input_tensor_info.image_info.height != input_tensor_info.image_info.crop_height)) {
                PRINT_E("Crop is not supported\n");
                return  kRetErr;
            }
            if ((input_tensor_info.image_info.crop_width != img_width) || (input_tensor_info.image_info.crop_height != img_height)) {
                PRINT_E("Resize is not supported\n");
                return  kRetErr;
            }
            if (input_tensor_info.image_info.channel != img_channel) {
                PRINT_E("Color conversion is not supported\n");
                return  kRetErr;
            }

            /* Normalize image */
            if (input_tensor_info.tensor_type == TensorInfo::kTensorTypeFp32) {
                float* dst = (float*)(input_buffer_list_[input_tensor_info.id].get());
                PreProcessImage(num_threads_, input_tensor_info, dst);
            } else if (input_tensor_info.tensor_type == TensorInfo::kTensorTypeUint8) {
                uint8_t* dst = (uint8_t*)(input_buffer_list_[input_tensor_info.id].get());
                PreProcessImage(num_threads_, input_tensor_info, dst);
            } else if (input_tensor_info.tensor_type == TensorInfo::kTensorTypeInt8) {
                int8_t* dst = (int8_t*)(input_buffer_list_[input_tensor_info.id].get());
                PreProcessImage(num_threads_, input_tensor_info, dst);
            } else {
                PRINT_E("Unsupported tensor_type (%d)\n", input_tensor_info.tensor_type);
                return kRetErr;
            }
        } else if ((input_tensor_info.data_type == InputTensorInfo::kDataTypeBlobNhwc) || (input_tensor_info.data_type == InputTensorInfo::kDataTypeBlobNchw)) {
            if (input_tensor_info.tensor_type == TensorInfo::kTensorTypeFp32) {
                float* dst = (float*)(input_buffer_list_[input_tensor_info.id].get());
                PreProcessBlob<float>(num_threads_, input_tensor_info, dst);
            } else if (input_tensor_info.tensor_type == TensorInfo::kTensorTypeUint8 || input_tensor_info.tensor_type == TensorInfo::kTensorTypeInt8) {
                uint8_t* dst = (uint8_t*)(input_buffer_list_[input_tensor_info.id].get());
                PreProcessBlob<uint8_t>(num_threads_, input_tensor_info, dst);
            } else if (input_tensor_info.tensor_type == TensorInfo::kTensorTypeInt32) {
                int32_t* dst = (int32_t*)(input_buffer_list_[input_tensor_info.id].get());
                PreProcessBlob<int32_t>(num_threads_, input_tensor_info, dst);
            } else {
                PRINT_E("Unsupported tensor_type (%d)\n", input_tensor_info.tensor_type);
                return kRetErr;
            }
        } else {
            PRINT_E("Unsupported data_type (%d)\n", input_tensor_info.data_type);
            return kRetErr;
        }
    }
    return kRetOk;
}

int32_t InferenceHelperOnnxRuntime::Process(std::vector<OutputTensorInfo>& output_tensor_info_list)
{
    std::vector<const char*> input_name_char_list;
    std::vector<const char*> output_name_char_list;
    for (const auto& str : input_name_list_) {
        input_name_char_list.emplace_back(str.c_str());
    }
    for (const auto& str : output_name_list_) {
        output_name_char_list.emplace_back(str.c_str());
    }

    try {
        session_.Run(Ort::RunOptions{ nullptr }, input_name_char_list.data(), input_tensor_list_.data(), input_tensor_list_.size(), output_name_char_list.data(), output_tensor_list_.data(), output_tensor_list_.size());
    } catch (std::exception& e) {
        PRINT_E("[ERROR] Unable to run session: %s\n", e.what());
        return kRetErr;
    }

    return kRetOk;
}

int32_t InferenceHelperOnnxRuntime::AllocateTensor(bool is_input, size_t index, std::vector<InputTensorInfo>& input_tensor_info_list, std::vector<OutputTensorInfo>& output_tensor_info_list)
{
    /* Get tensor name from model */
    Ort::AllocatorWithDefaultOptions ort_alloc;
    char* name_from_model = is_input ? session_.GetInputName(index, ort_alloc) : session_.GetOutputName(index, ort_alloc);
    std::string name_from_model_str = name_from_model;
    ort_alloc.Free(name_from_model);

    /* Find corresponding configure */
    static constexpr size_t kInvalid = 99999;
    size_t matched_index = kInvalid;
    if (is_input) {
        for (size_t i = 0; i < input_tensor_info_list.size(); i++) {
            auto& tensor_info = input_tensor_info_list[i];
            if (name_from_model_str == tensor_info.name) {
                tensor_info.id = static_cast<int32_t>(index);
                matched_index = i;
            }
        }
    } else {
        for (size_t i = 0; i < output_tensor_info_list.size(); i++) {
            auto& tensor_info = output_tensor_info_list[i];
            if (name_from_model_str == tensor_info.name) {
                tensor_info.id = static_cast<int32_t>(index);
                matched_index = i;
            }
        }
    }

    if (matched_index == kInvalid) {
        PRINT_E("tensor[%s] is not configured\n", name_from_model_str.c_str());
        return kRetErr;
    }
    TensorInfo* tensor_info = nullptr;
    if (is_input) {
        tensor_info = &input_tensor_info_list[matched_index];
    } else {
        tensor_info = &output_tensor_info_list[matched_index];
    }

    /* Check tensor shape (input only) */
    Ort::TypeInfo info = is_input ? session_.GetInputTypeInfo(index) : session_.GetOutputTypeInfo(index);
    auto shape_info = info.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> shape = shape_info.GetShape();
    if (is_input) {
        if (shape.size() != tensor_info->tensor_dims.size()) {
            PRINT_E("%s: element_dim doesn't match. %zu != %zu\n", name_from_model_str.c_str(), shape.size(), tensor_info->tensor_dims.size());
            return kRetErr;
        }
        for (size_t i = 0; i < shape.size(); i++) {
            if (shape[i] != tensor_info->tensor_dims[i]) {
                PRINT_E("%s: dim[%zu] doesn't match. %" PRId64 " != %d\n", name_from_model_str.c_str(), i, shape[i], tensor_info->tensor_dims[i]);
                return kRetErr;
            }
        }
    }

    /* Create tensor mem */
    std::unique_ptr<uint8_t[]> buffer;
    Ort::Value tensor(nullptr);
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    //auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    ONNXTensorElementDataType element_type = shape_info.GetElementType();
    size_t element_count = shape_info.GetElementCount();
    size_t byte_count = element_count;
    switch (element_type) {
    default:
        PRINT_E("%s: Unsupported element_type: %s\n", name_from_model_str.c_str(), DATA_TYPE_ID_TO_NAME_MAP[element_type]);
        return kRetErr;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
        if (tensor_info->tensor_type != TensorInfo::kTensorTypeFp32) {
            PRINT_E("%s: tensor_type doesn't match. %s != %d\n", name_from_model_str.c_str(), DATA_TYPE_ID_TO_NAME_MAP[element_type], tensor_info->tensor_type);
            return kRetErr;
        }
        byte_count *= sizeof(float);
        break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
        if (tensor_info->tensor_type != TensorInfo::kTensorTypeUint8) {
            PRINT_E("%s: tensor_type doesn't match. %s != %d\n", name_from_model_str.c_str(), DATA_TYPE_ID_TO_NAME_MAP[element_type], tensor_info->tensor_type);
            return kRetErr;
        }
        byte_count *= sizeof(uint8_t);
        break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
        if (tensor_info->tensor_type != TensorInfo::kTensorTypeInt8) {
            PRINT_E("%s: tensor_type doesn't match. %s != %d\n", name_from_model_str.c_str(), DATA_TYPE_ID_TO_NAME_MAP[element_type], tensor_info->tensor_type);
            return kRetErr;
        }
        byte_count *= sizeof(int8_t);
        break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
        if (tensor_info->tensor_type != TensorInfo::kTensorTypeInt32) {
            PRINT_E("%s: tensor_type doesn't match. %s != %d\n", name_from_model_str.c_str(), DATA_TYPE_ID_TO_NAME_MAP[element_type], tensor_info->tensor_type);
            return kRetErr;
        }
        byte_count *= sizeof(int32_t);
        break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
        if (tensor_info->tensor_type != TensorInfo::kTensorTypeInt64) {
            PRINT_E("%s: tensor_type doesn't match. %s != %d\n", name_from_model_str.c_str(), DATA_TYPE_ID_TO_NAME_MAP[element_type], tensor_info->tensor_type);
            return kRetErr;
        }
        byte_count *= sizeof(int64_t);
        break;
    }
    buffer = std::make_unique<uint8_t[]>(byte_count);
    tensor = Ort::Value::CreateTensor(memory_info, buffer.get(), byte_count, shape.data(), shape.size(), element_type);

    /* Store tensor info */
    if (is_input) {
        input_name_list_.emplace_back(name_from_model_str);
        input_tensor_list_.emplace_back(std::move(tensor));
        input_buffer_list_.emplace_back(std::move(buffer));
    } else {
        output_name_list_.emplace_back(name_from_model_str);
        output_tensor_list_.emplace_back(std::move(tensor));
        output_buffer_list_.emplace_back(std::move(buffer));
    }

    /* Set buffer index and shape (output only) */
    if (is_input) {
        //input_tensor_info_list[matched_index].data = input_buffer_list_.back().get();
    } else {
        output_tensor_info_list[matched_index].data = output_buffer_list_.back().get();
        output_tensor_info_list[matched_index].tensor_dims.clear();
        for (auto shape_val : shape) {
            output_tensor_info_list[matched_index].tensor_dims.push_back(static_cast<int32_t>(shape_val));
        }
    }

    return kRetOk;
}
