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
#include <memory>

/* for Armnn */
#include <armnn/ArmNN.hpp>
// #include "./install/include/arm_compute/runtime/Scheduler.h"
#include <armnnTfLiteParser/ITfLiteParser.hpp>
#include <armnnOnnxParser/IOnnxParser.hpp>

/* for My modules */
#include "common_helper.h"
#include "inference_helper_armnn.h"

/*** Macro ***/
#define TAG "InferenceHelperArmnn"
#define PRINT(...)   COMMON_HELPER_PRINT(TAG, __VA_ARGS__)
#define PRINT_E(...) COMMON_HELPER_PRINT_E(TAG, __VA_ARGS__)


/*** Function ***/

/* Note: Use Armnn via ArmnnWrapper class because the interfaces of tflite/onnx parser are not unified */
class ArmnnWrapper {
public:
    virtual ~ArmnnWrapper() {
        for(auto& buffer : list_buffer_in_) {
            delete[] (float*)(buffer);
        }
        for(auto& buffer : list_buffer_out_) {
            delete[] (float*)(buffer);
        }
    }

    virtual int32_t Initialize(const char* model_path, int32_t num_threads) = 0;

protected:
    virtual armnn::BindingPointInfo GetNetworkInputBindingInfo(const std::string& name) = 0;
    virtual armnn::BindingPointInfo GetNetworkOutputBindingInfo(const std::string& name) = 0;

protected:
    armnn::Status CreateRuntime(int32_t num_threads)
    {
        armnn::IRuntime::CreationOptions runtimeOptions;
        runtime_ = armnn::IRuntime::Create(runtimeOptions);
        if (!runtime_) {
            PRINT_E("Failed to create runtime\n");
            return armnn::Status::Failure;
        }
        PRINT("Supported Backends:\n")
        for (const auto& backend : runtime_->GetDeviceSpec().GetSupportedBackends()){
            const auto& backendStr = backend.Get();
            PRINT("  %s\n", backendStr.c_str());
        }

        armnn::OptimizerOptions optimizer_options;
        optimizer_options.m_ReduceFp32ToFp16 = false;
        std::vector<armnn::BackendId> preferred_backends;
        preferred_backends.push_back(armnn::Compute::GpuAcc);
        armnn::BackendOptions gpuAcc("GpuAcc",
                                        {
                                            { "FastMathEnabled", true },
                                            { "TuningLevel", 2},
                                        });
        optimizer_options.m_ModelOptions.push_back(gpuAcc);
        preferred_backends.push_back(armnn::Compute::CpuAcc);
        armnn::BackendOptions cpuAcc("CpuAcc",
                                        {
                                            { "FastMathEnabled", true },
                                            { "NumberOfThreads", num_threads },
                                        });
        optimizer_options.m_ModelOptions.push_back(cpuAcc);
        preferred_backends.push_back(armnn::Compute::CpuRef);

        armnn::IOptimizedNetworkPtr opt_net = armnn::Optimize(*network_, preferred_backends, runtime_->GetDeviceSpec(), optimizer_options);
        if (!opt_net) {
            PRINT_E("Failed to optimize network\n");
            return armnn::Status::Failure;
        }
        return runtime_->LoadNetwork(networkIdentifier_, std::move(opt_net));
    }

private:
    int32_t CheckTensorSize(const armnn::TensorInfo& armnn_tensor_info, TensorInfo& tensor_info)
    {
        bool is_size_assigned = true;
        if ( (tensor_info.tensor_dims.batch == -1) || (tensor_info.tensor_dims.height == -1) || (tensor_info.tensor_dims.width == -1) || (tensor_info.tensor_dims.channel == -1)) is_size_assigned = false;
        for (uint32_t i = 0; i < armnn_tensor_info.GetNumDimensions(); i++) {
            if (is_size_assigned) {
                if (((i == data_order_indices_[0]) && (tensor_info.tensor_dims.batch == static_cast<int32_t>(armnn_tensor_info.GetShape()[i])))
                    || ((i == data_order_indices_[1]) && (tensor_info.tensor_dims.height == static_cast<int32_t>(armnn_tensor_info.GetShape()[i])))
                    || ((i == data_order_indices_[2]) && (tensor_info.tensor_dims.width == static_cast<int32_t>(armnn_tensor_info.GetShape()[i])))
                    || ((i == data_order_indices_[3]) && (tensor_info.tensor_dims.channel == static_cast<int32_t>(armnn_tensor_info.GetShape()[i])))) {
                    /* OK */
                } else {
                    PRINT_E("Input Tensor size doesn't match\n");
                    return InferenceHelper::kRetErr;
                }
            } else {
                if (i == data_order_indices_[0]) tensor_info.tensor_dims.batch = armnn_tensor_info.GetShape()[i];
                if (i == data_order_indices_[1]) tensor_info.tensor_dims.channel = armnn_tensor_info.GetShape()[i];
                if (i == data_order_indices_[2]) tensor_info.tensor_dims.height = armnn_tensor_info.GetShape()[i];
                if (i == data_order_indices_[3]) tensor_info.tensor_dims.width = armnn_tensor_info.GetShape()[i];
            }
        }

        if (((tensor_info.tensor_type == TensorInfo::kTensorTypeUint8) && (armnn_tensor_info.GetDataType() == armnn::DataType::QAsymmU8))
            || ((tensor_info.tensor_type == TensorInfo::kTensorTypeInt8) && (armnn_tensor_info.GetDataType() == armnn::DataType::QAsymmS8))   // QSymmS8?
            || ((tensor_info.tensor_type == TensorInfo::kTensorTypeFp32) && (armnn_tensor_info.GetDataType() == armnn::DataType::Float32))
            || ((tensor_info.tensor_type == TensorInfo::kTensorTypeInt32) && (armnn_tensor_info.GetDataType() == armnn::DataType::Signed32))
            || ((tensor_info.tensor_type == TensorInfo::kTensorTypeInt64) && (armnn_tensor_info.GetDataType() == armnn::DataType::Signed64)) ) {
                /* OK */
        } else {
            PRINT_E("Input Tensor type doesn't match\n");
            return InferenceHelper::kRetErr;
        }
        return  InferenceHelper::kRetOk;
    }

    int32_t AllocateBuffer(const armnn::TensorInfo& armnn_tensor_info, std::vector<void*>& list_buffer)
    {
        switch(armnn_tensor_info.GetDataType()) {
        case armnn::DataType::QAsymmU8:
            list_buffer.push_back(new uint8_t[armnn_tensor_info.GetNumElements()]);
            break;
        case armnn::DataType::QAsymmS8:
            list_buffer.push_back(new int8_t[armnn_tensor_info.GetNumElements()]);
            break;
        case armnn::DataType::Float32:
            list_buffer.push_back(new float[armnn_tensor_info.GetNumElements()]);
            break;
        case armnn::DataType::Signed32:
            list_buffer.push_back(new int32_t[armnn_tensor_info.GetNumElements()]);
            break;
        case armnn::DataType::Signed64:
            list_buffer.push_back(new int64_t[armnn_tensor_info.GetNumElements()]);
            break;
        default:
            PRINT_E("Unsupported data type\n");
            return InferenceHelper::kRetErr;
        }
        return  InferenceHelper::kRetOk;
    }

public:
    int32_t AllocateTensor(std::vector<InputTensorInfo>& input_tensor_info_list, std::vector<OutputTensorInfo>& output_tensor_info_list)
    {
        for (auto& tensor_info : input_tensor_info_list) {
            armnn::BindingPointInfo armnn_info;
            try {
                armnn_info = GetNetworkInputBindingInfo(tensor_info.name.c_str());
            } catch(...) {
                PRINT_E("Failed to find tensor name: %s\n", tensor_info.name.c_str());
                return InferenceHelper::kRetErr;
            }
            tensor_info.id = armnn_info.first;
            armnn::TensorInfo armnn_tensor_info = armnn_info.second;
            if (CheckTensorSize(armnn_tensor_info, tensor_info) != InferenceHelper::kRetOk) {
                return InferenceHelper::kRetErr;
            }
            if (AllocateBuffer(armnn_tensor_info, list_buffer_in_) != InferenceHelper::kRetOk) {
                return InferenceHelper::kRetErr;
            }
            list_armnntensor_in_.push_back(std::make_pair(armnn_info.first, armnn::Tensor(armnn_info.second, list_buffer_in_.back())));
        }
 
         for (auto& tensor_info : output_tensor_info_list) {
            armnn::BindingPointInfo armnn_info;
            try {
                armnn_info = GetNetworkOutputBindingInfo(tensor_info.name.c_str());
            } catch(...) {
                PRINT_E("Failed to find tensor name: %s\n", tensor_info.name.c_str());
                return InferenceHelper::kRetErr;
            }
            tensor_info.id = armnn_info.first;
            armnn::TensorInfo armnn_tensor_info = armnn_info.second;
            if (CheckTensorSize(armnn_tensor_info, tensor_info) != InferenceHelper::kRetOk) {
                return InferenceHelper::kRetErr;
            }
            if (AllocateBuffer(armnn_tensor_info, list_buffer_out_) != InferenceHelper::kRetOk) {
                return InferenceHelper::kRetErr;
            }

            list_armnntensor_out_.push_back(std::make_pair(armnn_info.first, armnn::Tensor(armnn_info.second, list_buffer_out_.back())));

            tensor_info.data = list_buffer_out_.back();
            tensor_info.quant.zero_point = armnn_tensor_info.GetQuantizationOffset();
            tensor_info.quant.scale = armnn_tensor_info.GetQuantizationScale();
        }

        return InferenceHelper::kRetOk;
    }

    int32_t Process()
    {
        runtime_->EnqueueWorkload(networkIdentifier_, list_armnntensor_in_, list_armnntensor_out_);
        return InferenceHelper::kRetOk;
    }

public:
    std::vector<void*> list_buffer_in_;
    std::vector<void*> list_buffer_out_;

protected:
    armnn::INetworkPtr network_{nullptr, [](armnn::INetwork *){}};
    armnn::IRuntimePtr runtime_{nullptr, [](armnn::IRuntime *){}};
    armnn::NetworkId networkIdentifier_;
    armnn::InputTensors list_armnntensor_in_;
    armnn::OutputTensors list_armnntensor_out_;
    uint32_t data_order_indices_[4];

};


class ArmnnWrapperOnnx : public ArmnnWrapper {
public:
    ArmnnWrapperOnnx() {}
    ~ArmnnWrapperOnnx() override {}

    int32_t Initialize(const char* model_path, int32_t num_threads) override
    {
        parser_ = armnnOnnxParser::IOnnxParser::Create();
        network_ = parser_->CreateNetworkFromBinaryFile(model_path);
        // network_->PrintGraph();
        if(CreateRuntime(num_threads) != armnn::Status::Success) {
            PRINT_E("Failed to create network\n");
            return InferenceHelper::kRetErr;
        }

        data_order_indices_[0] = 0;   // N
        data_order_indices_[1] = 2;   // H
        data_order_indices_[2] = 3;   // W
        data_order_indices_[3] = 1;   // C

        return InferenceHelper::kRetOk;
    }

protected:
    armnn::BindingPointInfo GetNetworkInputBindingInfo(const std::string& name) override
    {
        return parser_->GetNetworkInputBindingInfo(name);
    }

    armnn::BindingPointInfo GetNetworkOutputBindingInfo(const std::string& name) override
    {
        return parser_->GetNetworkOutputBindingInfo(name);
    }

private:
    armnnOnnxParser::IOnnxParserPtr parser_{nullptr, [](armnnOnnxParser::IOnnxParser *){}};
};



class ArmnnWrapperTfLite : public ArmnnWrapper {
public:
    ArmnnWrapperTfLite() {}
    ~ArmnnWrapperTfLite() override {}

    int32_t Initialize(const char* model_path, int32_t num_threads) override
    {
        parser_ = armnnTfLiteParser::ITfLiteParser::Create();
        network_ = parser_->CreateNetworkFromBinaryFile(model_path);
        // network_->PrintGraph();
        DisplayTfLiteModelInfo();
        if(CreateRuntime(num_threads) != armnn::Status::Success) {
            PRINT_E("Failed to create network\n");
            return InferenceHelper::kRetErr;
        }

        data_order_indices_[0] = 0;   // N
        data_order_indices_[1] = 1;   // H
        data_order_indices_[2] = 2;   // W
        data_order_indices_[3] = 3;   // C

        return InferenceHelper::kRetOk;
    }

protected:
    armnn::BindingPointInfo GetNetworkInputBindingInfo(const std::string& name) override
    {
        return parser_->GetNetworkInputBindingInfo(0, name);
    }

    armnn::BindingPointInfo GetNetworkOutputBindingInfo(const std::string& name) override
    {
        return parser_->GetNetworkOutputBindingInfo(0, name);
    }

private:
    std::string get_string_of_data_type(armnn::DataType data_type) {
        switch(data_type) {
        case armnn::DataType::Float16: return "Float16";
        case armnn::DataType::Float32: return "Float32";
        case armnn::DataType::QAsymmU8: return "QAsymmU8";
        case armnn::DataType::Signed32: return "Signed32";
        case armnn::DataType::Boolean: return "Boolean";
        case armnn::DataType::QSymmS16: return "QSymmS16";
        // case armnn::DataType::QuantizedSymm8PerAxis: return "QuantizedSymm8PerAxis";
        case armnn::DataType::QSymmS8: return "QSymmS8";
        case armnn::DataType::QAsymmS8: return "QAsymmS8";
        case armnn::DataType::BFloat16: return "BFloat16";
        case armnn::DataType::Signed64: return "Signed64";
        default: return "Unkonwn";
        }
    }

    void DisplayTfLiteModelInfo()
    {
        std::vector<std::string> input_name_list = parser_->GetSubgraphInputTensorNames(0);
        std::vector<std::string> output_name_list = parser_->GetSubgraphOutputTensorNames(0);
        PRINT("Input Tensors: %ld\n", input_name_list.size());
        for(const auto& input_name : input_name_list) {
            armnn::BindingPointInfo info = parser_->GetNetworkInputBindingInfo(0, input_name);
            int32_t layer_binding_id = info.first;
            armnn::TensorInfo armnn_tensor_info = info.second;
            PRINT("  layer_binding_id: %d\n", layer_binding_id);
            PRINT("  dims: %d\n", armnn_tensor_info.GetNumDimensions());
            for (uint32_t i = 0; i < armnn_tensor_info.GetNumDimensions(); i++) {
                PRINT("  dims[%d]: %d\n", i, armnn_tensor_info.GetShape()[i]);
            }
            PRINT("  data_type: %s\n", get_string_of_data_type(armnn_tensor_info.GetDataType()).c_str());
            PRINT("  Q_offset: %d, Q_scale: %f\n", armnn_tensor_info.GetQuantizationOffset(), armnn_tensor_info.GetQuantizationScale());
        }

        PRINT("Output Tensors: %ld\n", output_name_list.size());
        for(const auto& output_name : output_name_list) {
            armnn::BindingPointInfo info = parser_->GetNetworkOutputBindingInfo(0, output_name);
            int32_t layer_binding_id = info.first;
            armnn::TensorInfo armnn_tensor_info = info.second;
            PRINT("  layer_binding_id: %d\n", layer_binding_id);
            PRINT("  dims: %d\n", armnn_tensor_info.GetNumDimensions());
            for (uint32_t i = 0; i < armnn_tensor_info.GetNumDimensions(); i++) {
                PRINT("  dims[%d]: %d\n", i, armnn_tensor_info.GetShape()[i]);
            }
            PRINT("  data_type: %s\n", get_string_of_data_type(armnn_tensor_info.GetDataType()).c_str());
            PRINT("  Q_offset: %d, Q_scale: %f\n", armnn_tensor_info.GetQuantizationOffset(), armnn_tensor_info.GetQuantizationScale());
        }
    }

private:
    armnnTfLiteParser::ITfLiteParserPtr parser_{nullptr, [](armnnTfLiteParser::ITfLiteParser *){}};
};





InferenceHelperArmnn::InferenceHelperArmnn()
{
    is_nhwc_ = true;
    num_threads_ = 1;
}

int32_t InferenceHelperArmnn::SetNumThreads(const int32_t num_threads)
{
    num_threads_ = num_threads;
    return kRetOk;
}

int32_t InferenceHelperArmnn::SetCustomOps(const std::vector<std::pair<const char*, const void*>>& custom_ops)
{
    PRINT("[WARNING] This method is not supported\n");
    return kRetOk;
}

int32_t InferenceHelperArmnn::Initialize(const std::string& model_filename, std::vector<InputTensorInfo>& input_tensor_info_list, std::vector<OutputTensorInfo>& output_tensor_info_list)
{
    armnn::ConfigureLogging(true, false, armnn::LogSeverity::Info);

    if (model_filename.find(".tflite") != std::string::npos) {
        armnn_wrapper_ = std::make_unique<ArmnnWrapperTfLite>();
        is_nhwc_ = true;
    } else if (model_filename.find(".onnx") != std::string::npos) {
        armnn_wrapper_ = std::make_unique<ArmnnWrapperOnnx>();
        is_nhwc_ = false;   // NCHW
    } else {
        PRINT_E("Invalid model filename (%s)\n", model_filename.c_str());
        return kRetErr;
    }

    if (armnn_wrapper_->Initialize(model_filename.c_str(), num_threads_) != InferenceHelper::kRetOk) {
        PRINT_E("Failed to initialize armnn\n");
        return kRetErr;
    }

    /* Allocate buffers and assign to tensor info */
    if(armnn_wrapper_->AllocateTensor(input_tensor_info_list, output_tensor_info_list) != InferenceHelper::kRetOk) {
        PRINT_E("Failed to AllocateTensor\n");
        return InferenceHelper::kRetErr;
    }

    /* Convert normalize parameter to speed up */
    for (auto& tensor_info : input_tensor_info_list) {
        ConvertNormalizeParameters(tensor_info);
    }

    return InferenceHelper::kRetOk;

}

int InferenceHelperArmnn::Finalize(void)
{
    return kRetOk;
}

int32_t InferenceHelperArmnn::PreProcess(const std::vector<InputTensorInfo>& input_tensor_info_list)
{
    int32_t buffer_index = 0;
    for (const auto& input_tensor_info : input_tensor_info_list) {
        if (input_tensor_info.data_type == InputTensorInfo::kDataTypeImage) {
            if ((input_tensor_info.image_info.width != input_tensor_info.image_info.crop_width) || (input_tensor_info.image_info.height != input_tensor_info.image_info.crop_height)) {
                PRINT_E("Crop is not supported\n");
                return  kRetErr;
            }
            if ((input_tensor_info.image_info.crop_width != input_tensor_info.tensor_dims.width) || (input_tensor_info.image_info.crop_height != input_tensor_info.tensor_dims.height)) {
                PRINT_E("Resize is not supported\n");
                return  kRetErr;
            }
            if (input_tensor_info.image_info.channel != input_tensor_info.tensor_dims.channel) {
                PRINT_E("Color conversion is not supported\n");
                return  kRetErr;
            }

            /* Normalize image */
            if (input_tensor_info.tensor_type == TensorInfo::kTensorTypeFp32) {
                float *dst = (float*)(armnn_wrapper_->list_buffer_in_[buffer_index]);
                uint8_t *src = (uint8_t*)(input_tensor_info.data);
                if (is_nhwc_) {
#pragma omp parallel for num_threads(num_threads_)
                    for (int32_t c = 0; c < input_tensor_info.tensor_dims.channel; c++) {
                        for (int32_t i = 0; i < input_tensor_info.tensor_dims.width * input_tensor_info.tensor_dims.height; i++) {
                            dst[i * input_tensor_info.tensor_dims.channel + c] = 
                                (src[i * input_tensor_info.tensor_dims.channel + c] - input_tensor_info.normalize.mean[c]) * input_tensor_info.normalize.norm[c];
                        }
                    }
                } else {
#pragma omp parallel for num_threads(num_threads_)
                    for (int32_t c = 0; c < input_tensor_info.tensor_dims.channel; c++) {
                        for (int32_t i = 0; i < input_tensor_info.tensor_dims.width * input_tensor_info.tensor_dims.height; i++) {
                            dst[c * input_tensor_info.tensor_dims.width * input_tensor_info.tensor_dims.height + i] = 
                                (src[i * input_tensor_info.tensor_dims.channel + c] - input_tensor_info.normalize.mean[c]) * input_tensor_info.normalize.norm[c];
                        }
                    }
                }
            } else if (input_tensor_info.tensor_type == TensorInfo::kTensorTypeUint8 || input_tensor_info.tensor_type == TensorInfo::kTensorTypeInt8) {
                uint8_t *dst = (uint8_t*)(armnn_wrapper_->list_buffer_in_[buffer_index]);
                uint8_t *src = (uint8_t*)(input_tensor_info.data);

                if (is_nhwc_) {
#pragma omp parallel for num_threads(num_threads_)
                    for (int32_t c = 0; c < input_tensor_info.tensor_dims.channel; c++) {
                        for (int32_t i = 0; i < input_tensor_info.tensor_dims.width * input_tensor_info.tensor_dims.height; i++) {
                            dst[i * input_tensor_info.tensor_dims.channel + c] = src[i * input_tensor_info.tensor_dims.channel + c];
                        }
                    }
                } else {
#pragma omp parallel for num_threads(num_threads_)
                    for (int32_t c = 0; c < input_tensor_info.tensor_dims.channel; c++) {
                        for (int32_t i = 0; i < input_tensor_info.tensor_dims.width * input_tensor_info.tensor_dims.height; i++) {
                            dst[c * input_tensor_info.tensor_dims.width * input_tensor_info.tensor_dims.height + i] = src[i * input_tensor_info.tensor_dims.channel + c];
                        }
                    }
                }
            } else {
                PRINT_E("Unsupported tensor_type (%d)\n", input_tensor_info.tensor_type);
                return kRetErr;
            }

        } else if (input_tensor_info.data_type == InputTensorInfo::kDataTypeBlobNhwc) {
            uint8_t *dst = (uint8_t*)(armnn_wrapper_->list_buffer_in_[buffer_index]);
            uint8_t *src = (uint8_t*)(input_tensor_info.data);
            if (is_nhwc_) {
                memcpy(dst, src, input_tensor_info.tensor_dims.batch * input_tensor_info.tensor_dims.channel * input_tensor_info.tensor_dims.height * input_tensor_info.tensor_dims.width);
            } else {
#pragma omp parallel for num_threads(num_threads_)
                for (int32_t c = 0; c < input_tensor_info.tensor_dims.channel; c++) {
                    for (int32_t i = 0; i < input_tensor_info.tensor_dims.width * input_tensor_info.tensor_dims.height; i++) {
                        dst[c * input_tensor_info.tensor_dims.width * input_tensor_info.tensor_dims.height + i] = src[i * input_tensor_info.tensor_dims.channel + c];
                    }
                }
            }
        } else if (input_tensor_info.data_type == InputTensorInfo::kDataTypeBlobNchw) {
            uint8_t *dst = (uint8_t*)(armnn_wrapper_->list_buffer_in_[buffer_index]);
            uint8_t *src = (uint8_t*)(input_tensor_info.data);
            if (is_nhwc_) {
                for (int32_t c = 0; c < input_tensor_info.tensor_dims.channel; c++) {
                    for (int32_t i = 0; i < input_tensor_info.tensor_dims.width * input_tensor_info.tensor_dims.height; i++) {
                        dst[i * input_tensor_info.tensor_dims.channel + c] = src[c * input_tensor_info.tensor_dims.width * input_tensor_info.tensor_dims.height + i];
                    }
                }
            } else {
                memcpy(dst, src, input_tensor_info.tensor_dims.batch * input_tensor_info.tensor_dims.channel * input_tensor_info.tensor_dims.height * input_tensor_info.tensor_dims.width);
            }
        } else {
            PRINT_E("Unsupported tensor_type (%d)\n", input_tensor_info.tensor_type);
            return kRetErr;
        }

        buffer_index++;
    }
    return kRetOk;
}

int32_t InferenceHelperArmnn::Process(std::vector<OutputTensorInfo>& output_tensor_info_list)
{
    armnn_wrapper_->Process();
    (void)output_tensor_info_list;	// no need to set output data, because the ptr to output data is already set at initialize
    return kRetOk;
}

void InferenceHelperArmnn::ConvertNormalizeParameters(InputTensorInfo& tensor_info)
{
    if (tensor_info.data_type != InputTensorInfo::kDataTypeImage) return;

#if 0
    /* Convert to speeden up normalization:  ((src / 255) - mean) / norm  = src * 1 / (255 * norm) - (mean / norm) */
    for (int32_t i = 0; i < 3; i++) {
        tensor_info.normalize.mean[i] /= tensor_info.normalize.norm[i];
        tensor_info.normalize.norm[i] *= 255.0f;
        tensor_info.normalize.norm[i] = 1.0f / tensor_info.normalize.norm[i];
    }
#endif
#if 1
    /* Convert to speeden up normalization:  ((src / 255) - mean) / norm = (src  - (mean * 255))  * (1 / (255 * norm)) */
    for (int32_t i = 0; i < 3; i++) {
        tensor_info.normalize.mean[i] *= 255.0f;
        tensor_info.normalize.norm[i] *= 255.0f;
        tensor_info.normalize.norm[i] = 1.0f / tensor_info.normalize.norm[i];
    }
#endif
}

