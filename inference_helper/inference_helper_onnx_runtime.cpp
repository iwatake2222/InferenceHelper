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



static void PrintTensorInfo(const Ort::TypeInfo& info, size_t index, const char* name)
{
    auto tensor_info = info.GetTensorTypeAndShapeInfo();
    ONNXTensorElementDataType element_type = tensor_info.GetElementType();
    const char* element_type_str = DATA_TYPE_ID_TO_NAME_MAP[element_type];
    size_t element_count = tensor_info.GetElementCount();
    std::vector<int64_t> shape = tensor_info.GetShape();
    PRINT("    info[%zu]->name: %s\n", index, name);
    PRINT("    info[%zu]->element_type: %s\n", index, element_type_str);
    PRINT("    info[%zu]->element_count: %zu\n", index, element_count);
    for (size_t j = 0; j < shape.size(); j++) {
        PRINT("    info[%zu]->shape[%zu]: %lld\n", index, j, shape[j]);
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

/*** Function ***/
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
    Ort::Env env_(ORT_LOGGING_LEVEL_WARNING, "Default");
    Ort::SessionOptions session_options;
#ifdef USE_CUDA
    Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0));
#endif

#ifdef _WIN32
    CStringW model_filename_cstringw(model_filename.c_str());
    auto onnx_model_filename_pcxstr = model_filename_cstringw.GetString();
#else
    auto onnx_model_filename = model_filename.c_str();
#endif
    try {
        session_ = Ort::Session(env_, onnx_model_filename_pcxstr, session_options);
    } catch (std::exception& e) {
        PRINT_E("[ERROR] Unable to create session: %s\n", e.what());
    }

    DisplayModelInfo(session_);


    return kRetErr;
};


int32_t InferenceHelperOnnxRuntime::Finalize(void)
{
    return kRetOk;
}

int32_t InferenceHelperOnnxRuntime::PreProcess(const std::vector<InputTensorInfo>& input_tensor_info_list)
{
    return kRetOk;
}

int32_t InferenceHelperOnnxRuntime::Process(std::vector<OutputTensorInfo>& output_tensor_info_list)
{
    return kRetOk;
}
