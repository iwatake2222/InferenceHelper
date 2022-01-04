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

/* for OpenCV */
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

/* for My modules */
#include "inference_helper_log.h"
#include "inference_helper_opencv.h"

/*** Macro ***/
#define TAG "InferenceHelperOpenCV"
#define PRINT(...)   INFERENCE_HELPER_LOG_PRINT(TAG, __VA_ARGS__)
#define PRINT_E(...) INFERENCE_HELPER_LOG_PRINT_E(TAG, __VA_ARGS__)

/*** Function ***/
InferenceHelperOpenCV::InferenceHelperOpenCV()
{
}

InferenceHelperOpenCV::~InferenceHelperOpenCV()
{
}

int32_t InferenceHelperOpenCV::SetNumThreads(const int32_t num_threads)
{
    cv::setNumThreads(num_threads);
    return kRetOk;
}

int32_t InferenceHelperOpenCV::SetCustomOps(const std::vector<std::pair<const char*, const void*>>& custom_ops)
{
    PRINT("[WARNING] This method is not supported\n");
    return kRetOk;
}

int32_t InferenceHelperOpenCV::Initialize(const std::string& model_filename, std::vector<InputTensorInfo>& input_tensor_info_list, std::vector<OutputTensorInfo>& output_tensor_info_list)
{
    /*** check model format ***/
    bool is_onnx_model = false;
    bool is_darknet_model = false;
    bool is_openvino_model = false;
    std::string model_filename_darknet_weight = model_filename;
    std::string model_filename_openvino_weight = model_filename;
    if (model_filename.find(".onnx") != std::string::npos) {
        is_onnx_model = true;
    } else if (model_filename.find(".cfg") != std::string::npos) {
        is_darknet_model = true;
        model_filename_darknet_weight = model_filename_darknet_weight.replace(model_filename_darknet_weight.find(".cfg"), std::string(".weights").length(), ".weights");
    } else if (model_filename.find(".xml") != std::string::npos) {
        is_openvino_model = true;
        model_filename_openvino_weight = model_filename_openvino_weight.replace(model_filename_openvino_weight.find(".xml"), std::string(".bin").length(), ".bin");
    } else {
        PRINT_E("unsupoprted file format (%s)\n", model_filename.c_str());
        return kRetErr;
    }

    /*** Create network ***/
    try {
        if (is_onnx_model) {
            net_ = cv::dnn::readNetFromONNX(model_filename);
        } else if (is_darknet_model) {
            net_ = cv::dnn::readNetFromDarknet(model_filename, model_filename_darknet_weight);
        } else if (is_openvino_model) {
            net_ = cv::dnn::readNetFromModelOptimizer(model_filename, model_filename_openvino_weight);
        }
    } catch (std::exception& e) {
        PRINT_E("%s\n", e.what());
    }
    if (net_.empty() == true) {
        PRINT_E("Failed to create inference engine (%s)\n", model_filename.c_str());
        return kRetErr;
    }
    
    if (helper_type_ == kOpencv) {
        if (is_openvino_model) {
            net_.setPreferableBackend(cv::dnn::DNN_BACKEND_INFERENCE_ENGINE);
        } else {
            net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
            net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        }
    } else if (helper_type_ == kOpencvGpu) {
        net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net_.setPreferableTarget(cv::dnn::DNN_TARGET_OPENCL);
        // net_.setPreferableTarget(cv::dnn::DNN_TARGET_VULKAN);
    } else {
        PRINT_E("Invalid helper type (%d)\n", helper_type_);
        return kRetErr;
    }

    /*** Check tensor information  ***/
    /* Only one input tensor is supported. The input tensor name changes for some reasons. */
    if (input_tensor_info_list.size() == 1) {
        const auto& input_tensor = net_.getLayer(0);
        input_tensor_info_list[0].name = input_tensor->name;
        input_tensor_info_list[0].id = net_.getLayerId(input_tensor->name);
    } else {
        PRINT_E("Invalid input tensor num (%zu)\n", input_tensor_info_list.size());
    }

    /* Check output tensor name */
    for (auto& output_tensor_info : output_tensor_info_list) {
        bool is_found = false;
        for (const auto& layerName : net_.getLayerNames()) {
            if (output_tensor_info.name == layerName) {
                is_found = true;
                output_tensor_info.id = net_.getLayerId(layerName);
                break;
            }
        }
        if (is_found == false) {
            PRINT_E("Output name (%s) not found\n", output_tensor_info.name.c_str());
            return kRetErr;
        }
    }

    /* Convert normalize parameter to speed up */
    for (auto& input_tensor_info : input_tensor_info_list) {
        ConvertNormalizeParameters(input_tensor_info);
    }

    /* Check if tensor info is set */
    for (const auto& input_tensor_info : input_tensor_info_list) {
        for (const auto& dim : input_tensor_info.tensor_dims) {
            if (dim <= 0) {
                PRINT_E("Invalid tensor size\n");
                return kRetErr;
            }
        }
    }
    for (const auto& output_tensor_info : output_tensor_info_list) {
        for (const auto& dim : output_tensor_info.tensor_dims) {
            if (dim <= 0) {
                PRINT_E("Invalid tensor size\n");
                return kRetErr;
            }
        }
    }

    return kRetOk;
};


int32_t InferenceHelperOpenCV::Finalize(void)
{
    return kRetErr;
}

int32_t InferenceHelperOpenCV::PreProcess(const std::vector<InputTensorInfo>& input_tensor_info_list)
{
    in_mat_list_.clear();
    for (const auto& input_tensor_info : input_tensor_info_list) {
        cv::Mat img_blob;
        if (input_tensor_info.data_type == InputTensorInfo::kDataTypeImage) {
            /* Generate mat from original data */
            cv::Mat img_src = cv::Mat(cv::Size(input_tensor_info.image_info.width, input_tensor_info.image_info.height), (input_tensor_info.image_info.channel == 3) ? CV_8UC3 : CV_8UC1, input_tensor_info.data);

            /* Crop image */
            if (input_tensor_info.image_info.width == input_tensor_info.image_info.crop_width && input_tensor_info.image_info.height == input_tensor_info.image_info.crop_height) {
                /* do nothing */
            } else {
                img_src = img_src(cv::Rect(input_tensor_info.image_info.crop_x, input_tensor_info.image_info.crop_y, input_tensor_info.image_info.crop_width, input_tensor_info.image_info.crop_height));
            }

            /* Resize image */
            if (input_tensor_info.image_info.crop_width == input_tensor_info.GetWidth() && input_tensor_info.image_info.crop_height == input_tensor_info.GetHeight()) {
                /* do nothing */
            } else {
                cv::resize(img_src, img_src, cv::Size(input_tensor_info.GetWidth(), input_tensor_info.GetHeight()));
            }

            /* Convert color type */
            if (input_tensor_info.image_info.channel == input_tensor_info.GetChannel()) {
                if (input_tensor_info.image_info.channel == 3 && input_tensor_info.image_info.swap_color) {
                    cv::cvtColor(img_src, img_src, cv::COLOR_BGR2RGB);
                }
            } else if (input_tensor_info.image_info.channel == 3 && input_tensor_info.GetChannel() == 1) {
                cv::cvtColor(img_src, img_src, (input_tensor_info.image_info.is_bgr) ? cv::COLOR_BGR2GRAY : cv::COLOR_RGB2GRAY);
            } else if (input_tensor_info.image_info.channel == 1 && input_tensor_info.GetChannel() == 3) {
                cv::cvtColor(img_src, img_src, cv::COLOR_GRAY2BGR);
            } else {
                PRINT_E("Unsupported color conversion (%d, %d)\n", input_tensor_info.image_info.channel, input_tensor_info.GetChannel());
                return kRetErr;
            }

            if (input_tensor_info.tensor_type == TensorInfo::kTensorTypeFp32) {
                /* Normalize image */
                if (input_tensor_info.GetChannel() == 3) {
#if 1
                    img_src.convertTo(img_src, CV_32FC3);
                    cv::subtract(img_src, cv::Scalar(cv::Vec<float, 3>(input_tensor_info.normalize.mean)), img_src);
                    cv::multiply(img_src, cv::Scalar(cv::Vec<float, 3>(input_tensor_info.normalize.norm)), img_src);
                    
#else
                    img_src.convertTo(img_src, CV_32FC3, 1.0 / 255);
                    cv::subtract(img_src, cv::Scalar(cv::Vec<float, 3>(input_tensor_info.normalize.mean)), img_src);
                    cv::divide(img_src, cv::Scalar(cv::Vec<float, 3>(input_tensor_info.normalize.norm)), img_src);
#endif
                } else if (input_tensor_info.GetChannel() == 1) {
#if 1
                    img_src.convertTo(img_src, CV_32FC1);
                    cv::subtract(img_src, cv::Scalar(cv::Vec<float, 1>(input_tensor_info.normalize.mean)), img_src);
                    cv::multiply(img_src, cv::Scalar(cv::Vec<float, 1>(input_tensor_info.normalize.norm)), img_src);
#else
                    img_src.convertTo(img_src, CV_32FC1, 1.0 / 255);
                    cv::subtract(img_src, cv::Scalar(cv::Vec<float, 1>(input_tensor_info.normalize.mean)), img_src);
                    cv::divide(img_src, cv::Scalar(cv::Vec<float, 1>(input_tensor_info.normalize.norm)), img_src);
#endif
                } else {
                    PRINT_E("Unsupported channel num (%d)\n", input_tensor_info.GetChannel());
                    return kRetErr;
                }
                if (input_tensor_info.is_nchw) {
                    /* Convert to 4-dimensional Mat in NCHW */
                    img_blob = cv::dnn::blobFromImage(img_src);
                } else {
                    const std::vector<int> sizes = { 1, img_src.rows, img_src.cols, img_src.channels() };
                    cv::Mat mat(sizes, CV_32F, img_src.data);
                    img_blob = mat.clone();
                }
            } else if (input_tensor_info.tensor_type == TensorInfo::kTensorTypeUint8) {
                /* Convert to 4-dimensional Mat in NCHW */
                img_blob = cv::dnn::blobFromImage(img_src);
            } else {
                PRINT_E("Unsupported tensor_type (%d)\n", input_tensor_info.tensor_type);
                return kRetErr;
            }

        } else if (input_tensor_info.data_type == InputTensorInfo::kDataTypeBlobNhwc) {
            cv::Mat img_src;
            if (input_tensor_info.tensor_type == TensorInfo::kTensorTypeFp32) {
                img_src = cv::Mat(cv::Size(input_tensor_info.GetWidth(), input_tensor_info.GetHeight()), (input_tensor_info.GetChannel() == 3) ? CV_32FC3 : CV_32FC1, input_tensor_info.data);
            } else if (input_tensor_info.tensor_type == TensorInfo::kTensorTypeUint8) {
                img_src = cv::Mat(cv::Size(input_tensor_info.GetWidth(), input_tensor_info.GetHeight()), (input_tensor_info.GetChannel() == 3) ? CV_8UC3 : CV_8UC1, input_tensor_info.data);
            } else {
                PRINT_E("Unsupported tensor_type (%d)\n", input_tensor_info.tensor_type);
                return kRetErr;
            }
            img_blob = cv::dnn::blobFromImage(img_src);
        } else if (input_tensor_info.data_type == InputTensorInfo::kDataTypeBlobNchw) {
            PRINT_E("Unsupported data_type (%d)\n", input_tensor_info.data_type);
            return kRetErr;
        } else {
            PRINT_E("Unsupported data type (%d)\n", input_tensor_info.data_type);
            return kRetErr;
        }
        in_mat_list_.push_back(img_blob);
    }
    return kRetOk;
}

int32_t InferenceHelperOpenCV::Process(std::vector<OutputTensorInfo>& output_tensor_info_list)
{
    if (in_mat_list_.size() != 1) {
        PRINT_E("Input tensor is not set\n");
        return kRetErr;
    }
    net_.setInput(in_mat_list_[0]);

    /*** Run inference ***/
    std::vector<cv::String> outNameList;
    for (const auto& output_tensor_info : output_tensor_info_list) {
        outNameList.push_back(output_tensor_info.name);
    }
    out_mat_list_.clear();
    try {
        net_.forward(out_mat_list_, outNameList);
    } catch (std::exception& e) {
        PRINT_E("Error at forward: %s\n", e.what());
    }

    /*** Retrieve the results ***/
    if (out_mat_list_.size() != output_tensor_info_list.size()) {
        PRINT_E("Unexpected output tensor num (%zu)\n", out_mat_list_.size());
        return kRetErr;
    }
    for (int32_t i = 0; i < static_cast<int32_t>(out_mat_list_.size()); i++) {
        output_tensor_info_list[i].data = out_mat_list_[i].data;
        output_tensor_info_list[i].tensor_dims.clear();
        output_tensor_info_list[i].tensor_dims.push_back(1);
        output_tensor_info_list[i].tensor_dims.push_back(out_mat_list_[i].channels());
        output_tensor_info_list[i].tensor_dims.push_back(out_mat_list_[i].rows);
        output_tensor_info_list[i].tensor_dims.push_back(out_mat_list_[i].cols);
    }

    return kRetOk;
}

