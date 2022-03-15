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
#include <torch/torch.h>

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
    return kRetOk;
}

int32_t InferenceHelperLibtorch::SetCustomOps(const std::vector<std::pair<const char*, const void*>>& custom_ops)
{
    PRINT("[WARNING] This method is not supported\n");
    return kRetOk;
}

int32_t InferenceHelperLibtorch::Initialize(const std::string& model_filename, std::vector<InputTensorInfo>& input_tensor_info_list, std::vector<OutputTensorInfo>& output_tensor_info_list)
{
    torch::Tensor tensor = torch::rand({ 2, 3 });
    std::cout << tensor << std::endl;
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
    return kRetOk;
}

int32_t InferenceHelperLibtorch::Process(std::vector<OutputTensorInfo>& output_tensor_info_list)
{
    return kRetOk;
}
