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
    int32_t setNumThread(const int32_t numThread) override;
    int32_t setCustomOps(const std::vector<std::pair<const char*, const void*>>& customOps) override;
    int32_t initialize(const std::string& modelFilename, std::vector<InputTensorInfo>& inputTensorInfoList, std::vector<OutputTensorInfo>& outputTensorInfoList) override;
    int32_t finalize(void) override;
    int32_t preProcess(const std::vector<InputTensorInfo>& inputTensorInfoList) override;
    int32_t invoke(std::vector<OutputTensorInfo>& outputTensorInfoList) override;

private:
    void convertNormalizeParameters(InputTensorInfo& tensorInfo);

private:
    std::unique_ptr<ncnn::Net> m_net;
    std::vector<std::pair<std::string, ncnn::Mat>> m_inMatList;	// <name, mat>
    std::vector<ncnn::Mat> m_outMatList;
    int32_t m_numThread;
};

#endif
