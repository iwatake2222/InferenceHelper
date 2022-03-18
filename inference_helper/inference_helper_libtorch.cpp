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

/* for LibTorch */
//#include <torch/torch.h>
#include <torch/script.h> // One-stop header.
#include <torch/cuda.h>
#include <ATen/Parallel.h>
#include <ATen/core/ivalue.h>

/* for My modules */
#include "inference_helper_log.h"
#include "inference_helper_libtorch.h"

/*** Macro ***/
#define TAG "InferenceHelperLibtorch"
#define PRINT(...)   INFERENCE_HELPER_LOG_PRINT(TAG, __VA_ARGS__)
#define PRINT_E(...) INFERENCE_HELPER_LOG_PRINT_E(TAG, __VA_ARGS__)


/*** Function ***/
InferenceHelperLibtorch::InferenceHelperLibtorch()
{
    num_threads_ = 1;
}

InferenceHelperLibtorch::~InferenceHelperLibtorch()
{
}

int32_t InferenceHelperLibtorch::SetNumThreads(const int32_t num_threads)
{
    num_threads_ = num_threads;
    torch::set_num_threads(num_threads_);
    return kRetOk;
}

int32_t InferenceHelperLibtorch::SetCustomOps(const std::vector<std::pair<const char*, const void*>>& custom_ops)
{
    PRINT("[WARNING] This method is not supported\n");
    return kRetOk;
}

int32_t InferenceHelperLibtorch::Initialize(const std::string& model_filename, std::vector<InputTensorInfo>& input_tensor_info_list, std::vector<OutputTensorInfo>& output_tensor_info_list)
{
    /*** Note
    * Do not analyze model information.
    * The order of model inputs/ontputs must be the same as that of input_tensor_info_list/output_tensor_info_list
    */

    /*** Check CUDA ***/
    if (torch::cuda::is_available()) {
        PRINT("CUDA is available\n");
        if (helper_type_ == InferenceHelper::kLibtorchCuda) {
            device_type_ = torch::kCUDA;
        } else {
            device_type_ = torch::kCPU;
        }
    } else {
        PRINT("CUDA is not available\n");
        if (helper_type_ == InferenceHelper::kLibtorchCuda) {
            PRINT("[WARNING] kLibtorchCuda is selected, but CUDA is not available\n");
            device_type_ = torch::kCPU;
        } else {
            device_type_ = torch::kCPU;
        }
    }

    /*** Load model ***/
    try {
        module_ = torch::jit::load(model_filename);
    }
    catch (const c10::Error& e) {
        PRINT_E("[ERROR] Unable to load model %s: %s\n", model_filename.c_str(), e.what());
        return kRetErr;
    }
    module_.to(device_type_);
    module_.eval();

    /*** Convert normalize parameter to speed up ***/
    for (auto& input_tensor_info : input_tensor_info_list) {
        ConvertNormalizeParameters(input_tensor_info);
    }

    return kRetOk;
};

int32_t InferenceHelperLibtorch::Finalize(void)
{
    return kRetOk;
}


int32_t InferenceHelperLibtorch::PreProcess(const std::vector<InputTensorInfo>& input_tensor_info_list)
{
    /*** Allocate input tensor every frame ***/
    /* We need this only for the first time for kCPU, but tensor device changes for kCUDA. So, We need to reallocate it */
    /* Todo: there may be a way to reuse allocated GPU memory */
    input_tensor_list_.clear();

    /*** Normalize input data and store the converted data into the input tensor buffer ***/
    for (size_t input_tensor_index = 0; input_tensor_index < input_tensor_info_list.size(); input_tensor_index++) {
        const auto& input_tensor_info = input_tensor_info_list[input_tensor_index];
        const int32_t img_width = input_tensor_info.GetWidth();
        const int32_t img_height = input_tensor_info.GetHeight();
        const int32_t img_channel = input_tensor_info.GetChannel();

        torch::TensorOptions tensor_options;
        if (input_tensor_info.tensor_type == TensorInfo::kTensorTypeFp32) {
            tensor_options = torch::TensorOptions().dtype(torch::kFloat32).requires_grad(false);
        }
        std::vector<int64_t> sizes;
        for (auto v : input_tensor_info.tensor_dims) {
            sizes.push_back(v);
        }
        torch::Tensor input_tensor = torch::zeros(sizes, tensor_options);


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
                float* dst = (float*)(input_tensor.data_ptr());
                PreProcessImage(num_threads_, input_tensor_info, dst);
            } else if (input_tensor_info.tensor_type == TensorInfo::kTensorTypeUint8) {
                uint8_t* dst = (uint8_t*)(input_tensor.data_ptr());
                PreProcessImage(num_threads_, input_tensor_info, dst);
            } else if (input_tensor_info.tensor_type == TensorInfo::kTensorTypeInt8) {
                int8_t* dst = (int8_t*)(input_tensor.data_ptr());
                PreProcessImage(num_threads_, input_tensor_info, dst);
            } else {
                PRINT_E("Unsupported tensor_type (%d)\n", input_tensor_info.tensor_type);
                return kRetErr;
            }
        } else if ((input_tensor_info.data_type == InputTensorInfo::kDataTypeBlobNhwc) || (input_tensor_info.data_type == InputTensorInfo::kDataTypeBlobNchw)) {
            if (input_tensor_info.tensor_type == TensorInfo::kTensorTypeFp32) {
                float* dst = (float*)(input_tensor.data_ptr());
                PreProcessBlob<float>(num_threads_, input_tensor_info, dst);
            } else if (input_tensor_info.tensor_type == TensorInfo::kTensorTypeUint8 || input_tensor_info.tensor_type == TensorInfo::kTensorTypeInt8) {
                uint8_t* dst = (uint8_t*)(input_tensor.data_ptr());
                PreProcessBlob<uint8_t>(num_threads_, input_tensor_info, dst);
            } else if (input_tensor_info.tensor_type == TensorInfo::kTensorTypeInt32) {
                int32_t* dst = (int32_t*)(input_tensor.data_ptr());
                PreProcessBlob<int32_t>(num_threads_, input_tensor_info, dst);
            } else {
                PRINT_E("Unsupported tensor_type (%d)\n", input_tensor_info.tensor_type);
                return kRetErr;
            }
        } else {
            PRINT_E("Unsupported data_type (%d)\n", input_tensor_info.data_type);
            return kRetErr;
        }

        input_tensor_list_.push_back(input_tensor.to(device_type_));
    }
    
    return kRetOk;
}

int32_t InferenceHelperLibtorch::Process(std::vector<OutputTensorInfo>& output_tensor_info_list)
{
    /*** Inference ***/
    torch::jit::IValue outputs;
    try {
        outputs = module_.forward(input_tensor_list_);
    } catch (std::exception& e) {
        PRINT("Error at forward: %s\n", e.what());
    }

    /*** Extract output tensor data and save them to output_tensor_list_ ***/
    output_tensor_list_.clear();
    if (outputs.isTensor()) {
        torch::Tensor output_tensor = outputs.toTensor().to(torch::kCPU);
        output_tensor_list_.emplace_back(output_tensor);
        //std::cout << output_tensor << std::endl;
    } else if (outputs.isTuple()) {
        PRINT("Multiple output is not tested\n");
        const auto& output_tuple = outputs.toTuple()->elements();
        for (const auto& o : output_tuple) {
            torch::Tensor output_tensor = o.toTensor().to(torch::kCPU);
            output_tensor_list_.emplace_back(output_tensor);
        }
    // } else if (outputs.isTensorList()) {
    //     PRINT("Multiple output is not tested\n");
    //     const auto& output_list = outputs.toTensorList();
    //     for (const auto& o : output_list) {
    //         torch::Tensor output_tensor = o;
    //         output_tensor = output_tensor.to(torch::kCPU);
    //         output_tensor_list_.emplace_back(output_tensor);
    //     }
    } else {
        PRINT_E("Invalid output format\n");
        return kRetErr;
    }

    /*** Set output data for caller ***/
    if (output_tensor_list_.size() != output_tensor_info_list.size()) {
        PRINT_E("The num of output tensors doesn't match. Model has %zu output, but code expects %zu\n", output_tensor_list_.size(), output_tensor_info_list.size());
    }

    for (size_t i = 0; i < output_tensor_list_.size(); i++) {
        const auto& output_tensor = output_tensor_list_[i];
        auto& tensor_info = output_tensor_info_list[i];
        int32_t ndim = output_tensor.dim();
        tensor_info.tensor_dims.clear();
        for (int idim = 0; idim < ndim; idim++) {
            tensor_info.tensor_dims.push_back(output_tensor.size(idim));
        }
        tensor_info.data = output_tensor.data_ptr();
    }

    return kRetOk;
}
