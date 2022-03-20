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

/* for TensorFlow */
#include <tensorflow/c/c_api.h>

/* for My modules */
#include "inference_helper_log.h"
#include "inference_helper_tensorflow.h"

/*** Macro ***/
#define TAG "InferenceHelperTensorflow"
#define PRINT(...)   INFERENCE_HELPER_LOG_PRINT(TAG, __VA_ARGS__)
#define PRINT_E(...) INFERENCE_HELPER_LOG_PRINT_E(TAG, __VA_ARGS__)


/*** Function ***/
InferenceHelperTensorflow::InferenceHelperTensorflow()
{
    num_threads_ = 1;

    session_ = nullptr;
    graph_ = TF_NewGraph();
}

InferenceHelperTensorflow::~InferenceHelperTensorflow()
{
}

int32_t InferenceHelperTensorflow::SetNumThreads(const int32_t num_threads)
{
    num_threads_ = num_threads;
    return kRetOk;
}

int32_t InferenceHelperTensorflow::SetCustomOps(const std::vector<std::pair<const char*, const void*>>& custom_ops)
{
    PRINT("[WARNING] This method is not supported\n");
    return kRetOk;
}

static std::string GetOpName(const std::string& model_filename)
{
    std::string name;
    for (const auto c : model_filename) {
        if (c == ':') break;
        name.push_back(c);
    }
    return name;
}

static int32_t GetOpIndex(const std::string& model_filename)
{
    bool is_index_start = false;
    std::string index_str;
    for (const auto c : model_filename) {
        if (c == ':') {
            is_index_start = true;
            continue;
        }
        if (is_index_start) {
            index_str.push_back(c);
        }
    }
    return std::stoi(index_str);
}

int32_t InferenceHelperTensorflow::Initialize(const std::string& model_filename, std::vector<InputTensorInfo>& input_tensor_info_list, std::vector<OutputTensorInfo>& output_tensor_info_list)
{
    /*** Load model ***/
    TF_Status* status = TF_NewStatus();
    TF_SessionOptions* session_options = TF_NewSessionOptions();
    const char* tags = "serve";
    int32_t ntags = 1;
    session_ = TF_LoadSessionFromSavedModel(session_options, nullptr, model_filename.c_str(), &tags, ntags, graph_, nullptr, status);
    TF_DeleteSessionOptions(session_options);
    TF_Code status_code = TF_GetCode(status);
    TF_DeleteStatus(status);
    if (status_code != TF_OK) {
        PRINT_E("Unable to load model: %d, %s, %s\n", status_code, model_filename.c_str(), tags);
        return kRetErr;
    }

    /*** Display graph ***/
    //size_t pos = 0;
    //TF_Operation* oper;
    //printf("--- graph info ---\n");
    //while ((oper = TF_GraphNextOperation(graph, &pos)) != nullptr) {
    //    printf("%s\n", TF_OperationName(oper));
    //}
    //printf("--- graph info ---\n");

    /*** Allocate tensors ***/
    int32_t id_input = 0;
    for (auto& input_tensor_info : input_tensor_info_list) {
        input_tensor_info.id = id_input++;
        TF_Output op = { TF_GraphOperationByName(graph_, GetOpName(input_tensor_info.name).c_str()), GetOpIndex(input_tensor_info.name) };
        if (op.oper == nullptr) {
            PRINT_E("Can't find input tensor name: %s\n", input_tensor_info.name.c_str());
            return kRetErr;
        }
        input_op_list_.emplace_back(op);

        std::vector<int64_t> dims;
        for (const auto& dim : input_tensor_info.tensor_dims) {
            dims.push_back(dim);
        }
        TF_Tensor* input_tensor = TF_AllocateTensor(TF_FLOAT, dims.data(), static_cast<int32_t>(dims.size()), input_tensor_info.GetElementNum() * sizeof(float));
        input_tensor_list_.emplace_back(input_tensor);
    }

    for (auto& output_tensor_info : output_tensor_info_list) {
        TF_Output op = { TF_GraphOperationByName(graph_, GetOpName(output_tensor_info.name).c_str()), GetOpIndex(output_tensor_info.name) };
        if (op.oper == nullptr) {
            PRINT_E("Can't find output tensor name: %s\n", output_tensor_info.name.c_str());
            return kRetErr;
        }
        output_op_list_.emplace_back(op);
        output_tensor_list_.emplace_back(nullptr);
    }

    /*** Convert normalize parameter to speed up ***/
    for (auto& input_tensor_info : input_tensor_info_list) {
        ConvertNormalizeParameters(input_tensor_info);
    }

    return kRetOk;
};

int32_t InferenceHelperTensorflow::Finalize(void)
{
    for (auto& tensor : input_tensor_list_) {
        TF_DeleteTensor(tensor);
    }
    for (auto& tensor : output_tensor_list_) {
        TF_DeleteTensor(tensor);
    }
    TF_DeleteGraph(graph_);
    TF_Status* status = TF_NewStatus();
    TF_CloseSession(session_, status);
    TF_DeleteSession(session_, status);
    TF_DeleteStatus(status);

    input_op_list_.clear();
    output_op_list_.clear();
    input_tensor_list_.clear();
    output_tensor_list_.clear();
    return kRetOk;
}

int32_t InferenceHelperTensorflow::PreProcess(const std::vector<InputTensorInfo>& input_tensor_info_list)
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
                float* dst = static_cast<float*>(TF_TensorData(input_tensor_list_[input_tensor_info.id]));
                PreProcessImage(num_threads_, input_tensor_info, dst);
            } else if (input_tensor_info.tensor_type == TensorInfo::kTensorTypeUint8) {
                uint8_t* dst = static_cast<uint8_t*>(TF_TensorData(input_tensor_list_[input_tensor_info.id]));
                PreProcessImage(num_threads_, input_tensor_info, dst);
            } else if (input_tensor_info.tensor_type == TensorInfo::kTensorTypeInt8) {
                int8_t* dst = static_cast<int8_t*>(TF_TensorData(input_tensor_list_[input_tensor_info.id]));
                PreProcessImage(num_threads_, input_tensor_info, dst);
            } else {
                PRINT_E("Unsupported tensor_type (%d)\n", input_tensor_info.tensor_type);
                return kRetErr;
            }
        } else if ((input_tensor_info.data_type == InputTensorInfo::kDataTypeBlobNhwc) || (input_tensor_info.data_type == InputTensorInfo::kDataTypeBlobNchw)) {
            if (input_tensor_info.tensor_type == TensorInfo::kTensorTypeFp32) {
                float* dst = static_cast<float*>(TF_TensorData(input_tensor_list_[input_tensor_info.id]));
                PreProcessBlob<float>(num_threads_, input_tensor_info, dst);
            } else if (input_tensor_info.tensor_type == TensorInfo::kTensorTypeUint8 || input_tensor_info.tensor_type == TensorInfo::kTensorTypeInt8) {
                uint8_t* dst = static_cast<uint8_t*>(TF_TensorData(input_tensor_list_[input_tensor_info.id]));
                PreProcessBlob<uint8_t>(num_threads_, input_tensor_info, dst);
            } else if (input_tensor_info.tensor_type == TensorInfo::kTensorTypeInt32) {
                int32_t* dst = static_cast<int32_t*>(TF_TensorData(input_tensor_list_[input_tensor_info.id]));
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

int32_t InferenceHelperTensorflow::Process(std::vector<OutputTensorInfo>& output_tensor_info_list)
{
    /*** Delete previous result ***/
    for (auto& output_tensor : output_tensor_list_) {
        TF_DeleteTensor(output_tensor);
    }

    /*** Run session ***/
    TF_Code status_code = TF_ABORTED;
    TF_Status* status = TF_NewStatus();
    try {
        TF_SessionRun(session_, nullptr,
            &input_op_list_[0], &input_tensor_list_[0], static_cast<int32_t>(input_op_list_.size()),
            &output_op_list_[0], &output_tensor_list_[0], static_cast<int32_t>(output_op_list_.size()),
            nullptr, 0, nullptr, status
        );
        status_code = TF_GetCode(status);
    } catch (std::exception& e) {
        PRINT_E("Exception during run session: %s\n", e.what());
        TF_DeleteStatus(status);
        return kRetErr;
    }
    TF_DeleteStatus(status);
    if (status_code != TF_OK) {
        PRINT_E("Error run session: %d\n", status_code);
        return kRetErr;
    }

    /*** Get result ***/
    for (size_t i = 0; i < output_tensor_info_list.size(); i++) {
        auto& output_tensor_info = output_tensor_info_list[i];
        auto& output_tensor = output_tensor_list_[i];
        
        /* Get output tensor type */
        TF_DataType data_type = TF_TensorType(output_tensor);
        switch (data_type) {
        case TF_FLOAT:
            output_tensor_info.tensor_type = TensorInfo::kTensorTypeFp32;
            break;
        case TF_INT32:
            output_tensor_info.tensor_type = TensorInfo::kTensorTypeInt32;
            break;
        case TF_INT8:
            output_tensor_info.tensor_type = TensorInfo::kTensorTypeInt8;
            break;
        case TF_UINT8:
            output_tensor_info.tensor_type = TensorInfo::kTensorTypeUint8;
            break;
        default:
            PRINT_E("Unsupported data type: %d\n", data_type);
            return kRetErr;
        }

        /* Get output tensor dims */
        output_tensor_info.tensor_dims.clear();
        int32_t ndims = TF_NumDims(output_tensor);
        for (int32_t index_dim = 0; index_dim < ndims; index_dim++) {
            int64_t dim = TF_Dim(output_tensor, index_dim);
            output_tensor_info.tensor_dims.push_back(static_cast<int32_t>(dim));
        }

        /* Get output tensor data */
        output_tensor_info.data = TF_TensorData(output_tensor);
    }

    return kRetOk;
}
