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
#include <unordered_map>

/* for SNPE lib */
#include "SNPE/SNPE.hpp"
#include "SNPE/SNPEFactory.hpp"
#include "SNPE/SNPEBuilder.hpp"
#include "DlSystem/DlError.hpp"
#include "DlSystem/RuntimeList.hpp"
#include "DlSystem/UserBufferMap.hpp"
#include "DlSystem/UDLFunc.hpp"
#include "DlSystem/IUserBuffer.hpp"
#include "DlContainer/IDlContainer.hpp"
#include "DiagLog/IDiagLog.hpp"
#include "DlSystem/ITensor.hpp"
#include "DlSystem/StringList.hpp"
#include "DlSystem/TensorMap.hpp"
#include "DlSystem/TensorShape.hpp"
#include "DlSystem/IUserBufferFactory.hpp"

/* for SNPE helper func */
#include "udlExample.hpp"
#include "CreateUserBuffer.hpp"

/* for My modules */
#include "inference_helper_log.h"
#include "inference_helper_snpe.h"

/*** Macro ***/
#define TAG "InferenceHelperSnpe"
#define PRINT(...)   INFERENCE_HELPER_LOG_PRINT(TAG, __VA_ARGS__)
#define PRINT_E(...) INFERENCE_HELPER_LOG_PRINT_E(TAG, __VA_ARGS__)


/*** Function ***/
InferenceHelperSnpe::InferenceHelperSnpe()
{
    num_threads_ = 1;
    input_map_.reset(new zdl::DlSystem::UserBufferMap());
    output_map_.reset(new zdl::DlSystem::UserBufferMap());
}

InferenceHelperSnpe::~InferenceHelperSnpe()
{
}

int32_t InferenceHelperSnpe::SetNumThreads(const int32_t num_threads)
{
    num_threads_ = num_threads;
    return kRetOk;
}

int32_t InferenceHelperSnpe::SetCustomOps(const std::vector<std::pair<const char*, const void*>>& custom_ops)
{
    PRINT("[WARNING] This method is not supported\n");
    return kRetOk;
}

int32_t InferenceHelperSnpe::Initialize(const std::string& model_filename, std::vector<InputTensorInfo>& input_tensor_info_list, std::vector<OutputTensorInfo>& output_tensor_info_list)
{
    /* Settings for SNPE */
    int32_t user_buffer_source_type = CPUBUFFER;
    int32_t buffer_type = USERBUFFER_FLOAT;
    int32_t bit_width = 0;
    switch (input_tensor_info_list[0].tensor_type) {
    case InputTensorInfo::kTensorTypeFp32:
        buffer_type = USERBUFFER_FLOAT;
        break;
    case InputTensorInfo::kTensorTypeUint8:
        buffer_type = USERBUFFER_TF8;
        bit_width = 8;
        break;
    default:
        PRINT_E("Unsupported tensor type\n");
        return kRetErr;
    }
    bool use_user_supplied_buffers = (buffer_type == USERBUFFER_FLOAT || buffer_type == USERBUFFER_TF8 || buffer_type == USERBUFFER_TF16);

    /* Create network */
    snpe_ = CreateSnpe(model_filename, use_user_supplied_buffers);
    if (!snpe_) {
        PRINT_E("Failed to create SNPE\n");
        return kRetErr;
    }

    GetAllTensorInfo(snpe_, input_tensor_info_list, output_tensor_info_list);


    /* Allocate buffer memory for input/output */
    if (use_user_supplied_buffers) {
        if (buffer_type == USERBUFFER_TF8 || buffer_type == USERBUFFER_TF16) {
            PRINT_E("Not tested\n");
            createOutputBufferMap(*output_map_, application_output_buffers_, snpe_user_output_buffers_, snpe_, true, bit_width);
            createInputBufferMap(*input_map_, application_input_buffers_, snpe_user_input_buffers_, snpe_, true, bit_width);
        } else if (buffer_type == USERBUFFER_FLOAT) {
            createOutputBufferMap(*output_map_, application_output_buffers_, snpe_user_output_buffers_, snpe_, false, bit_width);
            if (user_buffer_source_type == CPUBUFFER) {
                createInputBufferMap(*input_map_, application_input_buffers_, snpe_user_input_buffers_, snpe_, false, bit_width);
            } else {
                PRINT_E("Not supported\n");
                return kRetErr;
            }
        }
    } else {
        PRINT_E("Not supported\n");
        return kRetErr;
    }

    /* Convert normalize parameter to speed up */
    for (auto& input_tensor_info : input_tensor_info_list) {
        ConvertNormalizeParameters(input_tensor_info);
    }

    return kRetOk;
};


int32_t InferenceHelperSnpe::Finalize(void)
{
    return kRetErr;
}

int32_t InferenceHelperSnpe::PreProcess(const std::vector<InputTensorInfo>& input_tensor_info_list)
{
    if (!snpe_ || !input_map_ || !output_map_) {
        PRINT_E("Interpreter is not built yet\n");
        return kRetErr;
    }

    for (const auto& input_tensor_info : input_tensor_info_list) {
        const int32_t img_width = input_tensor_info.GetWidth();
        const int32_t img_height = input_tensor_info.GetHeight();
        const int32_t img_channel = input_tensor_info.GetChannel();
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

            /* Normalize image (NHWC to NHWC)*/
            uint8_t* src = static_cast<uint8_t*>(input_tensor_info.data);
            if (input_tensor_info.tensor_type == TensorInfo::kTensorTypeUint8) {
                PRINT_E("kTensorTypeUint8 is not supported\n");
            } else if (input_tensor_info.tensor_type == TensorInfo::kTensorTypeFp32) {
                float* dst = reinterpret_cast<float*> (&application_input_buffers_.at(input_tensor_info.name)[0]);
#pragma omp parallel for num_threads(num_threads_)
                for (int32_t i = 0; i < img_width * img_height; i++) {
                    for (int32_t c = 0; c < img_channel; c++) {
#if 1
                        dst[i * img_channel + c] = (src[i * img_channel + c] - input_tensor_info.normalize.mean[c]) * input_tensor_info.normalize.norm[c];
#else
                        dst[i * img_channel + c] = (src[i * img_channel + c] / 255.0f - input_tensor_info.normalize.mean[c]) / input_tensor_info.normalize.norm[c];
#endif
                    }
                }
            } else {
                PRINT_E("Unsupported tensor_type (%d)\n", input_tensor_info.tensor_type);
                return kRetErr;
            }

        } else if ( (input_tensor_info.data_type == InputTensorInfo::kDataTypeBlobNhwc) || (input_tensor_info.data_type == InputTensorInfo::kDataTypeBlobNchw) ){
            PRINT_E("kDataTypeBlobNhwc (kDataTypeBlobNchw) is not supported\n");
            return kRetErr;
        } else {
            PRINT_E("Unsupported data type (%d)\n", input_tensor_info.data_type);
            return kRetErr;
        }
    }
    return kRetOk;
}

int32_t InferenceHelperSnpe::Process(std::vector<OutputTensorInfo>& output_tensor_info_list)
{
    if (!snpe_ || !input_map_ || !output_map_) {
        PRINT_E("Interpreter is not built yet\n");
        return kRetErr;
    }

    bool exec_status = snpe_->execute(*input_map_, *output_map_);
    if (exec_status == false) {
        PRINT_E("Error while executing the network.\n");
        return kRetErr;
    }

    for (auto& output_tensor_info : output_tensor_info_list) {
        output_tensor_info.data = application_output_buffers_.at(output_tensor_info.name).data();
    }

    return kRetOk;
}


static zdl::DlSystem::RuntimeList GetSystemAvailability(void)
{
    zdl::DlSystem::Version_t version = zdl::SNPE::SNPEFactory::getLibraryVersion();
    PRINT("SNPE Version: %s\n", version.asString().c_str());

    zdl::DlSystem::RuntimeList runtime_list;
    zdl::DlSystem::Runtime_t runtime = zdl::DlSystem::Runtime_t::DSP_FIXED8_TF;
    if (zdl::SNPE::SNPEFactory::isRuntimeAvailable(runtime) == false) {
        PRINT_E("DSP is not available. Falling back to GPU.\n");
        runtime = zdl::DlSystem::Runtime_t::GPU_FLOAT32_16_HYBRID;
        if (zdl::SNPE::SNPEFactory::isRuntimeAvailable(runtime) == false) {
            PRINT_E("GPU is not available. Falling back to CPU.\n");
            runtime = zdl::DlSystem::Runtime_t::CPU_FLOAT32;
        }
    }
    runtime_list.add(runtime);
    return runtime_list;
}


std::unique_ptr<zdl::SNPE::SNPE> InferenceHelperSnpe::CreateSnpe(const std::string& model_filename, bool use_user_supplied_buffers)
{
    zdl::DlSystem::RuntimeList runtime_list = GetSystemAvailability();

    std::unique_ptr<zdl::DlContainer::IDlContainer> container = zdl::DlContainer::IDlContainer::open(zdl::DlSystem::String(model_filename.c_str()));
    if (container == nullptr) {
        PRINT_E("Error while opening the container file.\n");
        return nullptr;
    }

    zdl::DlSystem::UDLFactoryFunc udl_func = UdlExample::MyUDLFactory;
    zdl::DlSystem::UDLBundle udl_bundle;
    udl_bundle.cookie = (void*)0xdeadbeaf, udl_bundle.func = udl_func; // 0xdeadbeaf to test cookie
    zdl::DlSystem::PlatformConfig platform_config;

    zdl::SNPE::SNPEBuilder snpe_builder(container.get());
    std::unique_ptr<zdl::SNPE::SNPE> snpe = snpe_builder.setOutputLayers({})
        .setRuntimeProcessorOrder(runtime_list)
        .setUdlBundle(udl_bundle)
        .setUseUserSuppliedBuffers(use_user_supplied_buffers)
        .setPlatformConfig(platform_config)
        .setInitCacheMode(false)
        .build();
    if (snpe == nullptr) {
        PRINT_E("Error while building SNPE object.\n");
        return nullptr;
    }

    auto logger_opt = snpe->getDiagLogInterface();
    if (!logger_opt) {
        PRINT_E("SNPE failed to obtain logging interface.\n");
    }
    auto logger = *logger_opt;
    auto opts = logger->getOptions();
    opts.LogFileDirectory = model_filename + "_log/";
    if(!logger->setOptions(opts)) {
        PRINT_E("Failed to set options\n");
        return nullptr;
    }
    if (!logger->start()) {
        PRINT_E("Failed to start logger\n");
        return nullptr;
    }
    return snpe;
}

int32_t InferenceHelperSnpe::GetTensorInfo(std::unique_ptr<zdl::SNPE::SNPE> const& snpe, const std::string& name, std::vector<int32_t>& dims)
{
    auto buffer_attributes_opt = snpe->getInputOutputBufferAttributes(name.c_str());
    if (!buffer_attributes_opt) {
        PRINT_E("Error obtaining attributes for input tensor. %s\n", name.c_str());
        return kRetErr;
    }
    zdl::DlSystem::TensorShape tensor_shape = buffer_attributes_opt->getDims();

    dims.clear();
    for (int32_t d = 0; d < tensor_shape.rank(); d++) {
        dims.push_back(static_cast<int32_t>(tensor_shape.getDimensions()[d]));
    }
    return kRetOk;
}

int32_t InferenceHelperSnpe::GetAllTensorInfo(std::unique_ptr<zdl::SNPE::SNPE> const& snpe, std::vector<InputTensorInfo>& input_tensor_info_list, std::vector<OutputTensorInfo>& output_tensor_info_list)
{
    for (auto& input_tensor_info : input_tensor_info_list) {
        std::vector<int32_t> dims;
        if (GetTensorInfo(snpe, input_tensor_info.name, dims) != 0) {
            return kRetErr;
        }
        if (dims.size() == 0 && input_tensor_info.tensor_dims.size() == 0) {
            PRINT_E("Tensor is undefined\n");
            return kRetErr;
        } else if (dims.size() == 0 && input_tensor_info.tensor_dims.size() > 0) {
            // do nothing
        } else if (dims.size() > 0 && input_tensor_info.tensor_dims.size() == 0) {
            input_tensor_info.tensor_dims = dims;
        } else {
            for (int32_t d = 0; d < dims.size(); d++) {
                if (dims[d] != input_tensor_info.tensor_dims[d]) {
                    PRINT_E("%s: Dim size doesn't match: %d vs %d\n", input_tensor_info.name.c_str(), dims[d], input_tensor_info.tensor_dims[d]);
                    return kRetErr;
                }
            }
        }
    }
    
    for (auto& output_tensor_info : output_tensor_info_list) {
        std::vector<int32_t> dims;
        if (GetTensorInfo(snpe, output_tensor_info.name, dims) != 0) {
            return kRetErr;
        }
        if (dims.size() == 0 && output_tensor_info.tensor_dims.size() == 0) {
            PRINT_E("Tensor is undefined\n");
            return kRetErr;
        } else if (dims.size() == 0 && output_tensor_info.tensor_dims.size() > 0) {
            // do nothing
        } else if (dims.size() > 0 && output_tensor_info.tensor_dims.size() == 0) {
            output_tensor_info.tensor_dims = dims;
        } else {
            for (int32_t d = 0; d < dims.size(); d++) {
                if (dims[d] != output_tensor_info.tensor_dims[d]) {
                    PRINT_E("%s: Dim size doesn't match: %d vs %d\n", output_tensor_info.name.c_str(), dims[d], output_tensor_info.tensor_dims[d]);
                    return kRetErr;
                }
            }
        }
    }

    return kRetOk;
}
