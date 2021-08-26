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

/* for My modules */
#include "inference_helper_log.h"
#include "inference_helper.h"

#ifdef INFERENCE_HELPER_ENABLE_OPENCV
#include "inference_helper_opencv.h"
#endif
#if defined(INFERENCE_HELPER_ENABLE_TFLITE) || defined(INFERENCE_HELPER_ENABLE_TFLITE_DELEGATE_XNNPACK) || defined(INFERENCE_HELPER_ENABLE_TFLITE_DELEGATE_GPU) || defined(INFERENCE_HELPER_ENABLE_TFLITE_DELEGATE_EDGETPU)
#include "inference_helper_tensorflow_lite.h"
#endif
#ifdef INFERENCE_HELPER_ENABLE_TENSORRT
#include "inference_helper_tensorrt.h"
#endif
#ifdef INFERENCE_HELPER_ENABLE_NCNN
#include "inference_helper_ncnn.h"
#endif
#ifdef INFERENCE_HELPER_ENABLE_MNN
#include "inference_helper_mnn.h"
#endif
#ifdef INFERENCE_HELPER_ENABLE_SNPE
#include "inference_helper_snpe.h"
#endif
#ifdef INFERENCE_HELPER_ENABLE_ARMNN
#include "inference_helper_armnn.h"
#endif
#ifdef INFERENCE_HELPER_ENABLE_NNABLA
#include "inference_helper_nnabla.h"
#endif

/*** Macro ***/
#define TAG "InferenceHelper"
#define PRINT(...)   INFERENCE_HELPER_LOG_PRINT(TAG, __VA_ARGS__)
#define PRINT_E(...) INFERENCE_HELPER_LOG_PRINT_E(TAG, __VA_ARGS__)


InferenceHelper* InferenceHelper::Create(const InferenceHelper::HelperType helper_type)
{
    InferenceHelper* p = nullptr;
    switch (helper_type) {
#ifdef INFERENCE_HELPER_ENABLE_OPENCV
    case kOpencv:
    case kOpencvGpu:
        PRINT("Use OpenCV \n");
        p = new InferenceHelperOpenCV();
        break;
#endif
#ifdef INFERENCE_HELPER_ENABLE_TFLITE
    case kTensorflowLite:
        PRINT("Use TensorflowLite\n");
        p = new InferenceHelperTensorflowLite();
        break;
#endif
#ifdef INFERENCE_HELPER_ENABLE_TFLITE_DELEGATE_XNNPACK
    case kTensorflowLiteXnnpack:
        PRINT("Use TensorflowLite XNNPACK Delegate\n");
        p = new InferenceHelperTensorflowLite();
        break;
#endif
#ifdef INFERENCE_HELPER_ENABLE_TFLITE_DELEGATE_GPU
    case kTensorflowLiteGpu:
        PRINT("Use TensorflowLite GPU Delegate\n");
        p = new InferenceHelperTensorflowLite();
        break;
#endif
#ifdef INFERENCE_HELPER_ENABLE_TFLITE_DELEGATE_EDGETPU
    case kTensorflowLiteEdgetpu:
        PRINT("Use TensorflowLite EdgeTPU Delegate\n");
        p = new InferenceHelperTensorflowLite();
        break;
#endif
#ifdef INFERENCE_HELPER_ENABLE_TFLITE_DELEGATE_NNAPI
    case kTensorflowLiteNnapi:
        PRINT("Use TensorflowLite NNAPI Delegate\n");
        p = new InferenceHelperTensorflowLite();
        break;
#endif
#ifdef INFERENCE_HELPER_ENABLE_TENSORRT
    case kTensorrt:
        PRINT("Use TensorRT \n");
        p = new InferenceHelperTensorRt();
        break;
#endif
#ifdef INFERENCE_HELPER_ENABLE_NCNN
    case kNcnn:
        PRINT("Use NCNN\n");
        p = new InferenceHelperNcnn();
        break;
#endif
#ifdef INFERENCE_HELPER_ENABLE_MNN
    case kMnn:
        PRINT("Use MNN\n");
        p = new InferenceHelperMnn();
        break;
#endif
#ifdef INFERENCE_HELPER_ENABLE_SNPE
    case kSnpe:
        PRINT("Use SNPE\n");
        p = new InferenceHelperSnpe();
        break;
#endif
#ifdef INFERENCE_HELPER_ENABLE_ARMNN
    case kArmnn:
        PRINT("Use ARMNN\n");
        p = new InferenceHelperArmnn();
        break;
#endif
#ifdef INFERENCE_HELPER_ENABLE_NNABLA
    case kNnabla:
        PRINT("Use NNabla\n");
        p = new InferenceHelperNnabla();
        break;
#endif
    default:
        PRINT_E("Unsupported inference helper type (%d)\n", helper_type);
        break;
    }
    if (p == nullptr) {
        PRINT_E("Failed to create inference helper\n");
    } else {
        p->helper_type_ = helper_type;
    }
    return p;
}

#ifdef INFERENCE_HELPER_ENABLE_PRE_PROCESS_BY_OPENCV
#include <opencv2/opencv.hpp>
void InferenceHelper::PreProcessByOpenCV(const InputTensorInfo& input_tensor_info, bool is_nchw, cv::Mat& img_blob)
{
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
        } else {
#if 1
            img_src.convertTo(img_src, CV_32FC1);
            cv::subtract(img_src, cv::Scalar(cv::Vec<float, 1>(input_tensor_info.normalize.mean)), img_src);
            cv::multiply(img_src, cv::Scalar(cv::Vec<float, 1>(input_tensor_info.normalize.norm)), img_src);
#else
            img_src.convertTo(img_src, CV_32FC1, 1.0 / 255);
            cv::subtract(img_src, cv::Scalar(cv::Vec<float, 1>(input_tensor_info.normalize.mean)), img_src);
            cv::divide(img_src, cv::Scalar(cv::Vec<float, 1>(input_tensor_info.normalize.norm)), img_src);
#endif
        }
    } else {
        /* do nothing */
    }

    if (is_nchw) {
        /* Convert to 4-dimensional Mat in NCHW */
        img_src = cv::dnn::blobFromImage(img_src);
    }

    img_blob = img_src;
    //memcpy(blobData, img_src.data, img_src.cols * img_src.rows * img_src.channels());

}

#else 
/* For the environment where OpenCV is not supported */
void InferenceHelper::PreProcessByOpenCV(const InputTensorInfo& input_tensor_info, bool is_nchw, cv::Mat& img_blob)
{
    PRINT_E("[PreProcessByOpenCV] Unsupported function called\n");
    exit(-1);
}
#endif

