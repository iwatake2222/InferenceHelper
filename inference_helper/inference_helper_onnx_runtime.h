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
#ifndef INFERENCE_HELPER_ONNX_RUNTIME_
#define INFERENCE_HELPER_ONNX_RUNTIME_

/* for general */
#include <cstdint>
#include <cmath>
#include <string>
#include <vector>
#include <array>
#include <memory>

/* for ONNX Runtime */
#include <onnxruntime_cxx_api.h>

/* for My modules */
#include "inference_helper.h"

class InferenceHelperOnnxRuntime : public InferenceHelper {
public:
    InferenceHelperOnnxRuntime();
    ~InferenceHelperOnnxRuntime() override;
    int32_t SetNumThreads(const int32_t num_threads) override;
    int32_t SetCustomOps(const std::vector<std::pair<const char*, const void*>>& custom_ops) override;
    int32_t Initialize(const std::string& model_filename, std::vector<InputTensorInfo>& input_tensor_info_list, std::vector<OutputTensorInfo>& output_tensor_info_list) override;
    int32_t Finalize(void) override;
    int32_t PreProcess(const std::vector<InputTensorInfo>& input_tensor_info_list) override;
    int32_t Process(std::vector<OutputTensorInfo>& output_tensor_info_list) override;

private:
    int32_t AllocateTensor(bool is_input, size_t index, std::vector<InputTensorInfo>& input_tensor_info_list, std::vector<OutputTensorInfo>& output_tensor_info_list);

private:
    int32_t num_threads_;

    Ort::Session session_{ nullptr };
    Ort::Env env_{ ORT_LOGGING_LEVEL_WARNING, "Default" };
    std::vector<std::string> input_name_list_;
    std::vector<std::string> output_name_list_;
    std::vector<Ort::Value> input_tensor_list_;
    std::vector<Ort::Value> output_tensor_list_;
    std::vector<std::unique_ptr<uint8_t[]>> input_buffer_list_;
    std::vector<std::unique_ptr<uint8_t[]>> output_buffer_list_;
};

#endif
