/* Copyright 2021 iwatake2222

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

/* for nnabla */
#include <nbla/logger.hpp>
#include <nbla_utils/nnp.hpp>

#ifdef INFERENCE_HELPER_ENABLE_NNABLA_CUDA
#include <nbla/cuda/cudnn/init.hpp>
#include <nbla/cuda/init.hpp>
#include <cuda_runtime_api.h>
#endif

/* for My modules */
#include "inference_helper_log.h"
#include "inference_helper_nnabla.h"

/*** Macro ***/
#define TAG "InferenceHelperNnabla"
#define PRINT(...)   INFERENCE_HELPER_LOG_PRINT(TAG, __VA_ARGS__)
#define PRINT_E(...) INFERENCE_HELPER_LOG_PRINT_E(TAG, __VA_ARGS__)

/*** Function ***/
InferenceHelperNnabla::InferenceHelperNnabla()
{
    num_threads_ = 1;
}

InferenceHelperNnabla::~InferenceHelperNnabla()
{
}

int32_t InferenceHelperNnabla::SetNumThreads(const int32_t num_threads)
{
    num_threads_ = num_threads;
    return kRetOk;
}

int32_t InferenceHelperNnabla::SetCustomOps(const std::vector<std::pair<const char*, const void*>>& custom_ops)
{
    PRINT("[WARNING] This method is not supported\n");
    return kRetOk;
}


int32_t InferenceHelperNnabla::Initialize(const std::string& model_filename, std::vector<InputTensorInfo>& input_tensor_info_list, std::vector<OutputTensorInfo>& output_tensor_info_list)
{
    try {
        ctx_cpu_.reset(new nbla::Context{ {"cpu:float"}, "CpuCachedArray", "0" });
        if (helper_type_ == kNnabla) {
            nnp_.reset(new nbla::utils::nnp::Nnp(*ctx_cpu_));
        }
#ifdef INFERENCE_HELPER_ENABLE_NNABLA_CUDA
        else if (helper_type_ == kNnablaCuda) {
            nbla::init_cudnn();
            ctx_gpu_.reset(new nbla::Context{ {"cudnn:float", "cuda:float", "cpu:float" }, "CudaCachedArray", "0"});
            nnp_.reset(new nbla::utils::nnp::Nnp(*ctx_gpu_));
        }
#endif

        nnp_->add(model_filename);
        const auto executor_name = nnp_->get_executor_names()[0];
        PRINT("Model filename = %s, executor name = %s\n", model_filename.c_str(), executor_name.c_str());
        executor_ = nnp_->get_executor(executor_name);
        executor_->set_batch_size(1);

        DisplayModelInfo();

        if (AllocateBuffers(input_tensor_info_list, output_tensor_info_list) != kRetOk) {
            PRINT_E("Error at AllocateBuffers\n");
            return kRetErr;
        }

    } catch (std::exception& e) {
        PRINT_E("Exception: %s\n", e.what());
        return kRetErr;
    }

    /* Convert normalize parameter to speed up */
    for (auto& input_tensor_info : input_tensor_info_list) {
        ConvertNormalizeParameters(input_tensor_info);
    }

    return kRetOk;
};


int32_t InferenceHelperNnabla::Finalize(void)
{
    return kRetErr;
}


int32_t InferenceHelperNnabla::PreProcess(const std::vector<InputTensorInfo>& input_tensor_info_list)
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
                float* dst = GetInputVariable(input_tensor_info.id)->cast_data_and_get_pointer<float>(*ctx_cpu_);
                PreProcessImage(num_threads_, input_tensor_info, dst);
            } else if (input_tensor_info.tensor_type == TensorInfo::kTensorTypeUint8) {
                uint8_t* dst = GetInputVariable(input_tensor_info.id)->cast_data_and_get_pointer<uint8_t>(*ctx_cpu_);
                PreProcessImage(num_threads_, input_tensor_info, dst);
            } else if (input_tensor_info.tensor_type == TensorInfo::kTensorTypeInt8) {
                int8_t* dst = GetInputVariable(input_tensor_info.id)->cast_data_and_get_pointer<int8_t>(*ctx_cpu_);
                PreProcessImage(num_threads_, input_tensor_info, dst);
            } else {
                PRINT_E("Unsupported tensor_type (%d)\n", input_tensor_info.tensor_type);
                return kRetErr;
            }
        } else if ((input_tensor_info.data_type == InputTensorInfo::kDataTypeBlobNhwc) || (input_tensor_info.data_type == InputTensorInfo::kDataTypeBlobNchw)) {
            if (input_tensor_info.tensor_type == TensorInfo::kTensorTypeFp32) {
                float* dst = GetInputVariable(input_tensor_info.id)->cast_data_and_get_pointer<float>(*ctx_cpu_);
                PreProcessBlob<float>(num_threads_, input_tensor_info, dst);
            } else if (input_tensor_info.tensor_type == TensorInfo::kTensorTypeUint8 || input_tensor_info.tensor_type == TensorInfo::kTensorTypeInt8) {
                uint8_t* dst = GetInputVariable(input_tensor_info.id)->cast_data_and_get_pointer<uint8_t>(*ctx_cpu_);
                PreProcessBlob<uint8_t>(num_threads_, input_tensor_info, dst);
            } else if (input_tensor_info.tensor_type == TensorInfo::kTensorTypeInt32) {
                int32_t* dst = GetInputVariable(input_tensor_info.id)->cast_data_and_get_pointer<int32_t>(*ctx_cpu_);
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

int32_t InferenceHelperNnabla::Process(std::vector<OutputTensorInfo>& output_tensor_info_list)
{
    try {
#ifdef INFERENCE_HELPER_ENABLE_NNABLA_CUDA
        if (helper_type_ == kNnablaCuda) {
            nbla::cuda_device_synchronize("0");
        }
#endif
        executor_->execute();

#ifdef INFERENCE_HELPER_ENABLE_NNABLA_CUDA
        if (helper_type_ == kNnablaCuda) {
            nbla::cuda_device_synchronize("0");
            

            /* todo: Do I really need this?  they don't use cudaMemcpy in sample code (mnist_runtime.cpp) */
            for (auto& tensor_info : output_tensor_info_list) {
                const auto& variable = GetOutputVariable(tensor_info.id);
                switch (tensor_info.tensor_type) {
                default:
                case TensorInfo::kTensorTypeFp32:
                    cudaMemcpy(tensor_info.data, variable->cast_data_and_get_pointer<float>(*ctx_gpu_), tensor_info.GetElementNum() * 4, cudaMemcpyDeviceToHost);
                    break;
                case TensorInfo::kTensorTypeInt32:
                    cudaMemcpy(tensor_info.data, variable->cast_data_and_get_pointer<int32_t>(*ctx_gpu_), tensor_info.GetElementNum() * 4, cudaMemcpyDeviceToHost);
                    break;
                case TensorInfo::kTensorTypeInt8:
                    cudaMemcpy(tensor_info.data, variable->cast_data_and_get_pointer<int8_t>(*ctx_gpu_), tensor_info.GetElementNum() * 1, cudaMemcpyDeviceToHost);
                    break;
                case TensorInfo::kTensorTypeUint8:
                    cudaMemcpy(tensor_info.data, variable->cast_data_and_get_pointer<uint8_t>(*ctx_gpu_), tensor_info.GetElementNum() * 1, cudaMemcpyDeviceToHost);
                    break;
                case TensorInfo::kTensorTypeInt64:
                    cudaMemcpy(tensor_info.data, variable->cast_data_and_get_pointer<int64_t>(*ctx_gpu_), tensor_info.GetElementNum() * 8, cudaMemcpyDeviceToHost);
                    break;
                }
            }
        }
#endif
    } catch (std::exception& e) {
        PRINT_E("Exception: %s\n", e.what());
        return kRetErr;
    }
    return kRetOk;
}



void InferenceHelperNnabla::DisplayModelInfo()
{
    PRINT("Input Tensors:\n");
    for (const auto& variable : executor_->get_data_variables()) {
        const auto& cg_variable = variable.variable->variable();
        PRINT("  %s:\n", variable.variable_name.c_str());
        PRINT("      size: %d\n", static_cast<int32_t>(cg_variable->size()));
        for (int32_t i = 0; i < static_cast<int32_t>(cg_variable->shape().size()); i++) {
            PRINT("      shape[%d]: %d\n", i, static_cast<int32_t>(cg_variable->shape()[i]));
        }
    }

    PRINT("Output Tensors:\n");
    for (const auto& variable : executor_->get_output_variables()) {
        const auto& cg_variable = variable.variable->variable();
        PRINT("  %s:\n", variable.variable_name.c_str());
        PRINT("      size: %d\n", static_cast<int32_t>(cg_variable->size()));
        for (int32_t i = 0; i < static_cast<int32_t>(cg_variable->shape().size()); i++) {
            PRINT("      shape[%d]: %d\n", i, static_cast<int32_t>(cg_variable->shape()[i]));
        }
    }
}

std::shared_ptr<nbla::Variable> InferenceHelperNnabla::GetInputVariable(int32_t index)
{
    return executor_->get_data_variables().at(index).variable->variable();
}

std::shared_ptr<nbla::Variable> InferenceHelperNnabla::GetOutputVariable(int32_t index)
{
    return executor_->get_output_variables().at(index).variable->variable();
}

int32_t InferenceHelperNnabla::AllocateBuffers(std::vector<InputTensorInfo>& input_tensor_info_list, std::vector<OutputTensorInfo>& output_tensor_info_list)
{
    /* Find index in variables of executor */
    for (auto& tensor_info : input_tensor_info_list) {
        const auto& variable_list = executor_->get_data_variables();
        int32_t index = 0;
        for (const auto& variable : variable_list) {
            if (tensor_info.name == variable.variable_name) break;
            index++;
        }
        if (index < static_cast<int32_t>(variable_list.size())) {
            tensor_info.id = index;
        } else {
            PRINT_E("Input tensor name doesn't exist in the model (%s)\n", tensor_info.name.c_str());
            return kRetErr;
        }
    }
    for (auto& tensor_info : output_tensor_info_list) {
        const auto& variable_list = executor_->get_output_variables();
        int32_t index = 0;
        for (const auto& variable : variable_list) {
            if (tensor_info.name == variable.variable_name) break;
            index++;
        }
        if (index < static_cast<int32_t>(variable_list.size())) {
            tensor_info.id = index;
        } else {
            PRINT_E("Output tensor name doesn't exist in the model (%s)\n", tensor_info.name.c_str());
            return kRetErr;
        }
    }

    /* Check tensor size and get buffer address */
    for (auto& tensor_info : input_tensor_info_list) {
        const auto& variable = GetInputVariable(tensor_info.id);
        if (CheckTensorInfo(tensor_info, variable) != kRetOk) {
            PRINT_E("Input tensor info is not correct (%s)\n", tensor_info.name.c_str());
            return kRetErr;
        }
        tensor_info.data = variable->cast_data_and_get_pointer<float>(*ctx_cpu_);
    }
    for (auto& tensor_info : output_tensor_info_list) {
        const auto& variable = GetOutputVariable(tensor_info.id);
        if (CheckTensorInfo(tensor_info, variable) != kRetOk) {
            PRINT_E("Output tensor info is not correct (%s)\n", tensor_info.name.c_str());
            return kRetErr;
        }
        tensor_info.data = variable->cast_data_and_get_pointer<float>(*ctx_cpu_);
    }

    return kRetOk;
}

int32_t InferenceHelperNnabla::CheckTensorInfo(TensorInfo& tensor_info, const shared_ptr<nbla::Variable> variable)
{
    const auto ndim = variable->ndim();
    const auto shape = variable->shape();
    if (tensor_info.tensor_dims.empty()) {
        /* tensor size is not assigned. so get from the model */
        for (int32_t i = 0; i < ndim; i++) {
            tensor_info.tensor_dims.push_back(static_cast<int32_t>(shape[i]));
        }
    } else {
        if (tensor_info.tensor_dims.size() != static_cast<size_t>(ndim)) {
            PRINT_E("Input Tensor dims doesn't match\n");
            return kRetErr;
        }
        for (auto i = 0; i < ndim; i++) {
            if (tensor_info.tensor_dims[i] != shape[i]) {
                PRINT_E("Input Tensor size doesn't match\n");
                return kRetErr;
            }
        }
    }

    /* todo: check data type (fp32, uint8, etc.) */

    return kRetOk;
}
