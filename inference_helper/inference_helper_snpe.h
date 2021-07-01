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
	enum {UNKNOWN, USERBUFFER_FLOAT, USERBUFFER_TF8, ITENSOR, USERBUFFER_TF16};
	enum {CPUBUFFER, GLBUFFER};
public:
	InferenceHelperSnpe();
	~InferenceHelperSnpe() override;
	int32_t setNumThread(const int32_t numThread) override;
	int32_t setCustomOps(const std::vector<std::pair<const char*, const void*>>& customOps) override;
	int32_t initialize(const std::string& modelFilename, std::vector<InputTensorInfo>& inputTensorInfoList, std::vector<OutputTensorInfo>& outputTensorInfoList) override;
	int32_t finalize(void) override;
	int32_t preProcess(const std::vector<InputTensorInfo>& inputTensorInfoList) override;
	int32_t invoke(std::vector<OutputTensorInfo>& outputTensorInfoList) override;

private:
	std::unique_ptr<zdl::SNPE::SNPE> createSnpe(const std::string& modelFilename, bool useUserSuppliedBuffers);
	int32_t getTensorInfo(std::unique_ptr<zdl::SNPE::SNPE> const& snpe, const std::string& name, int32_t& batch, int32_t& height, int32_t& width, int32_t& channel);
	int32_t getAllTensorInfo(std::unique_ptr<zdl::SNPE::SNPE> const& snpe, std::vector<InputTensorInfo>& inputTensorInfoList, std::vector<OutputTensorInfo>& outputTensorInfoList);
	void convertNormalizeParameters(InputTensorInfo& tensorInfo);

private:
	int32_t m_numThread;
	std::unique_ptr<zdl::SNPE::SNPE> m_snpe;
	std::unique_ptr<zdl::DlSystem::UserBufferMap> m_inputMap;
	std::unique_ptr<zdl::DlSystem::UserBufferMap> m_outputMap;
	std::vector <std::unique_ptr<zdl::DlSystem::IUserBuffer>> m_snpeUserBackedInputBuffers;
	std::vector <std::unique_ptr<zdl::DlSystem::IUserBuffer>> m_snpeUserBackedOutputBuffers;
	std::unordered_map <std::string, std::vector<uint8_t>> m_applicationOutputBuffers;
	std::unordered_map <std::string, std::vector<uint8_t>> m_applicationInputBuffers;
};

#endif
