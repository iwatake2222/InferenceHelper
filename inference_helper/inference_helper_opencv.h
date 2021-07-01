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
#ifndef INFERENCE_HELPER_OPENCV_
#define INFERENCE_HELPER_OPENCV_

/* for general */
#include <cstdint>
#include <cmath>
#include <string>
#include <vector>
#include <array>

/* for OpenCV */
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

/* for My modules */
#include "inference_helper.h"

class InferenceHelperOpenCV : public InferenceHelper {
public:
    InferenceHelperOpenCV();
    ~InferenceHelperOpenCV() override;
    int32_t setNumThread(const int32_t numThread) override;
    int32_t setCustomOps(const std::vector<std::pair<const char*, const void*>>& customOps) override;
    int32_t initialize(const std::string& modelFilename, std::vector<InputTensorInfo>& inputTensorInfoList, std::vector<OutputTensorInfo>& outputTensorInfoList) override;
    int32_t finalize(void) override;
    int32_t preProcess(const std::vector<InputTensorInfo>& inputTensorInfoList) override;
    int32_t invoke(std::vector<OutputTensorInfo>& outputTensorInfoList) override;

private:
    void convertNormalizeParameters(InputTensorInfo& inputTensorInfo);

private:
    cv::dnn::Net m_net;
    std::vector<cv::Mat> m_inMatList;
    std::vector<cv::Mat> m_outMatList;	// store data as member variable so that an user can refer the results
};

#endif
