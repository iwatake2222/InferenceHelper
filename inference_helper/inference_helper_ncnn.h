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
#ifndef INFERENCE_HELPER_NCNN_
#define INFERENCE_HELPER_NCNN_

/* for general */
#include <cstdint>
#include <cmath>
#include <string>
#include <vector>
#include <array>
#include <memory>

/* for ncnn */
#include "net.h"

/* for My modules */
#include "inference_helper.h"

class InferenceHelperNcnn : public InferenceHelper {
public:
    InferenceHelperNcnn();
    ~InferenceHelperNcnn() override;
    int32_t SetNumThreads(const int32_t num_threads) override;
    int32_t SetCustomOps(const std::vector<std::pair<const char*, const void*>>& custom_ops) override;
    int32_t Initialize(const std::string& model_filename, std::vector<InputTensorInfo>& input_tensor_info_list, std::vector<OutputTensorInfo>& output_tensor_info_list) override;
    int32_t Finalize(void) override;
    int32_t PreProcess(const std::vector<InputTensorInfo>& input_tensor_info_list) override;
    int32_t Process(std::vector<OutputTensorInfo>& output_tensor_info_list) override;

private:
    std::unique_ptr<ncnn::Net> net_;
    std::vector<std::pair<std::string, ncnn::Mat>> in_mat_list_;	// <name, mat>
    std::vector<ncnn::Mat> out_mat_list_;
    int32_t num_threads_;
    std::vector<std::pair<const char*, const void*>> custom_ops_;
};

#endif
