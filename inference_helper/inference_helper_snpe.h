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
#ifndef INFERENCE_HELPER_SNPE_
#define INFERENCE_HELPER_SNPE_

/* for general */
#include <cstdint>
#include <cmath>
#include <string>
#include <vector>
#include <array>
#include <memory>
#include <unordered_map>

/* for SNPE */

/* for My modules */
#include "inference_helper.h"

namespace zdl { namespace SNPE { class SNPE; } }
namespace zdl { namespace DlSystem { class IUserBuffer; } }
namespace zdl { namespace DlSystem { class UserBufferMap; } }

class InferenceHelperSnpe : public InferenceHelper {
private:
    enum { UNKNOWN, USERBUFFER_FLOAT, USERBUFFER_TF8, ITENSOR, USERBUFFER_TF16 };
    enum { CPUBUFFER, GLBUFFER };

public:
    InferenceHelperSnpe();
    ~InferenceHelperSnpe() override;
    int32_t SetNumThreads(const int32_t num_threads) override;
    int32_t SetCustomOps(const std::vector<std::pair<const char*, const void*>>& custom_ops) override;
    int32_t Initialize(const std::string& model_filename, std::vector<InputTensorInfo>& input_tensor_info_list, std::vector<OutputTensorInfo>& output_tensor_info_list) override;
    int32_t Finalize(void) override;
    int32_t PreProcess(const std::vector<InputTensorInfo>& input_tensor_info_list) override;
    int32_t Process(std::vector<OutputTensorInfo>& output_tensor_info_list) override;

private:
    std::unique_ptr<zdl::SNPE::SNPE> CreateSnpe(const std::string& model_filename, bool use_user_supplied_buffers);
    int32_t GetTensorInfo(std::unique_ptr<zdl::SNPE::SNPE> const& snpe, const std::string& name, std::vector<int32_t>& dims);
    int32_t GetAllTensorInfo(std::unique_ptr<zdl::SNPE::SNPE> const& snpe, std::vector<InputTensorInfo>& input_tensor_info_list, std::vector<OutputTensorInfo>& output_tensor_info_list);

private:
    int32_t num_threads_;
    std::unique_ptr<zdl::SNPE::SNPE> snpe_;
    std::unique_ptr<zdl::DlSystem::UserBufferMap> input_map_;
    std::unique_ptr<zdl::DlSystem::UserBufferMap> output_map_;
    std::vector <std::unique_ptr<zdl::DlSystem::IUserBuffer>> snpe_user_input_buffers_;
    std::vector <std::unique_ptr<zdl::DlSystem::IUserBuffer>> snpe_user_output_buffers_;
    std::unordered_map <std::string, std::vector<uint8_t>> application_input_buffers_;
    std::unordered_map <std::string, std::vector<uint8_t>> application_output_buffers_;
};

#endif
