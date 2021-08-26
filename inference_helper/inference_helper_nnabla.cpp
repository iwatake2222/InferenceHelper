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

//#include <nbla/cuda/cudnn/init.hpp>
//#include <nbla/cuda/init.hpp>

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
        if (helper_type_ == kNnabla) {
            ctx_.reset(new nbla::Context{ {"cpu:float"}, "CpuCachedArray", "0" });
        } else if (helper_type_ == kNnablaCuda) {
            //nbla::init_cudnn();
            //ctx_.reset(new nbla::Context{ {"cudnn:float", "cuda:float", "cpu:float" }, "CudaCachedArray", "0"});
        }
        nnp_.reset(new nbla::utils::nnp::Nnp(*ctx_));
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
                float* dst = (float*)GetInputVariable(input_tensor_info.id)->cast_data_and_get_pointer<float>(*ctx_);
                uint8_t* src = (uint8_t*)(input_tensor_info.data);
                if (input_tensor_info.is_nchw == true) {
                    /* convert NHWC to NCHW */
#pragma omp parallel for num_threads(num_threads_)
                    for (int32_t c = 0; c < img_channel; c++) {
                        for (int32_t i = 0; i < img_width * img_height; i++) {
                            dst[c * img_width * img_height + i] = (src[i * img_channel + c] - input_tensor_info.normalize.mean[c]) * input_tensor_info.normalize.norm[c];
                        }
                    }
                } else {
                    /* convert NHWC to NHWC */
#pragma omp parallel for num_threads(num_threads_)
                    for (int32_t i = 0; i < img_width * img_height; i++) {
                        for (int32_t c = 0; c < img_channel; c++) {
                            dst[i * img_channel + c] = (src[i * img_channel + c] - input_tensor_info.normalize.mean[c]) * input_tensor_info.normalize.norm[c];
                        }
                    }
                }
            } else {
                PRINT_E("Unsupported tensor_type (%d)\n", input_tensor_info.tensor_type);
                return kRetErr;
            }
        } else if (input_tensor_info.data_type == InputTensorInfo::kDataTypeBlobNhwc) {
            PRINT_E("Unsupported tensor_type (%d)\n", input_tensor_info.tensor_type);
            return kRetErr;
        } else if (input_tensor_info.data_type == InputTensorInfo::kDataTypeBlobNchw) {
            PRINT_E("Unsupported tensor_type (%d)\n", input_tensor_info.tensor_type);
            return kRetErr;
        } else {
            PRINT_E("Unsupported tensor_type (%d)\n", input_tensor_info.tensor_type);
            return kRetErr;
        }

    }
    return kRetOk;
}

int32_t InferenceHelperNnabla::Process(std::vector<OutputTensorInfo>& output_tensor_info_list)
{
    try {
        executor_->execute();
    } catch (std::exception& e) {
        PRINT_E("Exception: %s\n", e.what());
        return kRetErr;
    }
    return kRetOk;
}


void InferenceHelperNnabla::ConvertNormalizeParameters(InputTensorInfo& tensor_info)
{
    if (tensor_info.data_type != InputTensorInfo::kDataTypeImage) return;

#if 0
    /* Convert to speeden up normalization:  ((src / 255) - mean) / norm  = src * 1 / (255 * norm) - (mean / norm) */
    for (int32_t i = 0; i < 3; i++) {
        tensor_info.normalize.mean[i] /= tensor_info.normalize.norm[i];
        tensor_info.normalize.norm[i] *= 255.0f;
        tensor_info.normalize.norm[i] = 1.0f / tensor_info.normalize.norm[i];
    }
#endif
#if 1
    /* Convert to speeden up normalization:  ((src / 255) - mean) / norm = (src  - (mean * 255))  * (1 / (255 * norm)) */
    for (int32_t i = 0; i < 3; i++) {
        tensor_info.normalize.mean[i] *= 255.0f;
        tensor_info.normalize.norm[i] *= 255.0f;
        tensor_info.normalize.norm[i] = 1.0f / tensor_info.normalize.norm[i];
    }
#endif
}


void InferenceHelperNnabla::DisplayModelInfo()
{
    PRINT("Input Tensors:\n");
    for (const auto& variable : executor_->get_data_variables()) {
        const auto& cg_variable = variable.variable->variable();
        PRINT("  %s:\n", variable.variable_name.c_str());
        PRINT("      size: %d\n", static_cast<int32_t>(cg_variable->size()));
        for (auto i = 0; i < cg_variable->shape().size(); i++) {
            PRINT("      shape[%d]: %d\n", static_cast<int32_t>(i), static_cast<int32_t>(cg_variable->shape()[i]));
        }
    }

    PRINT("Output Tensors:\n");
    for (const auto& variable : executor_->get_output_variables()) {
        const auto& cg_variable = variable.variable->variable();
        PRINT("  %s:\n", variable.variable_name.c_str());
        PRINT("      size: %d\n", static_cast<int32_t>(cg_variable->size()));
        for (auto i = 0; i < cg_variable->shape().size(); i++) {
            PRINT("      shape[%d]: %d\n", static_cast<int32_t>(i), static_cast<int32_t>(cg_variable->shape()[i]));
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
    /* Find index in variable of executor */
    for (auto& tensor_info : input_tensor_info_list) {
        const auto& variable_list = executor_->get_data_variables();
        int32_t index = 0;
        for (const auto& variable : variable_list) {
            if (tensor_info.name == variable.variable_name) break;
            index++;
        }
        if (index < variable_list.size()) {
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
        if (index < variable_list.size()) {
            tensor_info.id = index;
        } else {
            PRINT_E("Output tensor name doesn't exist in the model (%s)\n", tensor_info.name.c_str());
            return kRetErr;
        }
    }

    /* Check tensor size and get buffer address */
    for (auto& tensor_info : input_tensor_info_list) {
        const auto& variable = executor_->get_data_variables().at(tensor_info.id).variable->variable();
        if (CheckTensorInfo(tensor_info, variable) != kRetOk) {
            PRINT_E("Input tensor info is not correct (%s)\n", tensor_info.name.c_str());
            return kRetErr;
        }
        tensor_info.data = variable->cast_data_and_get_pointer<float>(*ctx_);
    }
    for (auto& tensor_info : output_tensor_info_list) {
        const auto& variable = executor_->get_output_variables().at(tensor_info.id).variable->variable();
        if (CheckTensorInfo(tensor_info, variable) != kRetOk) {
            PRINT_E("Output tensor info is not correct (%s)\n", tensor_info.name.c_str());
            return kRetErr;
        }
        tensor_info.data = variable->cast_data_and_get_pointer<float>(*ctx_);
    }

    return kRetOk;
}

int32_t InferenceHelperNnabla::CheckTensorInfo(TensorInfo& tensor_info, const shared_ptr <nbla::Variable> variable)
{
    const auto ndim = variable->ndim();
    const auto shape = variable->shape();
    if (tensor_info.tensor_dims.empty()) {
        /* tensor size is not assigned. so get from the model */
        for (int32_t i = 0; i < ndim; i++) {
            tensor_info.tensor_dims.push_back(static_cast<int32_t>(shape[i]));
        }
    } else {
        if (tensor_info.tensor_dims.size() != ndim) {
            PRINT_E("Input Tensor dims doesn't match\n");
            return kRetErr;
        }
        for (int32_t i = 0; i < ndim; i++) {
            if (tensor_info.tensor_dims[i] != shape[i]) {
                PRINT_E("Input Tensor size doesn't match\n");
                return kRetErr;
            }
        }
    }

    /* todo: check data type (fp32, uint8, etc.) */

    return kRetOk;
}
