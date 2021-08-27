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
#include <fstream>
#include <iostream>

/* for Tensorflow Lite */
#ifdef INFERENCE_HELPER_ENABLE_TFLITE_DELEGATE_EDGETPU
#include "edgetpu.h"
#include "edgetpu_c.h"
#endif

#ifdef INFERENCE_HELPER_ENABLE_TFLITE_DELEGATE_GPU
#include "tensorflow/lite/delegates/gpu/delegate.h"
#endif

#ifdef INFERENCE_HELPER_ENABLE_TFLITE_DELEGATE_XNNPACK
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"
#endif

#ifdef INFERENCE_HELPER_ENABLE_TFLITE_DELEGATE_NNAPI
#include "tensorflow/lite/delegates/nnapi/nnapi_delegate.h"
#endif

/* for My modules */
#include "inference_helper_log.h"
#include "inference_helper_tensorflow_lite.h"

/*** Macro ***/
#define TAG "InferenceHelperTensorflowLite"
#define PRINT(...)   INFERENCE_HELPER_LOG_PRINT(TAG, __VA_ARGS__)
#define PRINT_E(...) INFERENCE_HELPER_LOG_PRINT_E(TAG, __VA_ARGS__)

/*** Function ***/
InferenceHelperTensorflowLite::InferenceHelperTensorflowLite()
{
    num_threads_ = 1;
    resolver_.reset(new tflite::ops::builtin::BuiltinOpResolver());
}

InferenceHelperTensorflowLite::~InferenceHelperTensorflowLite()
{
}

int32_t InferenceHelperTensorflowLite::SetNumThreads(const int32_t num_threads)
{
    num_threads_ = num_threads;
    return kRetOk;
}

int32_t InferenceHelperTensorflowLite::SetCustomOps(const std::vector<std::pair<const char*, const void*>>& custom_ops)
{
    for (auto op : custom_ops) {
        resolver_->AddCustom(op.first, (const TfLiteRegistration*)op.second);
    }
    return kRetOk;
}

int32_t InferenceHelperTensorflowLite::Initialize(const std::string& model_filename, std::vector<InputTensorInfo>& input_tensor_info_list, std::vector<OutputTensorInfo>& output_tensor_info_list)
{
    /*** Create network ***/
#if 0
    model_ = tflite::FlatBufferModel::BuildFromFile(model_filename.c_str());
#else
    std::ifstream ifs(model_filename, std::ios::binary);
    if (ifs) {
        ifs >> std::noskipws;
        (void)std::copy(std::istream_iterator<char>(ifs), std::istream_iterator<char>(), back_inserter(model_buffer_));
    } else {
        PRINT_E("Failed to read model (%s)\n", model_filename.c_str());
        return kRetErr;
    }
    ifs.close();
    model_ = tflite::FlatBufferModel::BuildFromBuffer(model_buffer_.data(), model_buffer_.size());
#endif

    if (model_ == nullptr) {
        PRINT_E("Failed to build model (%s)\n", model_filename.c_str());
        return kRetErr;
    }

    tflite::InterpreterBuilder builder(*model_, *resolver_);
    builder(&interpreter_);
    if (interpreter_ == nullptr) {
        PRINT_E("Failed to build interpreter (%s)\n", model_filename.c_str());
        return kRetErr;
    }

    interpreter_->SetNumThreads(num_threads_);

#ifdef INFERENCE_HELPER_ENABLE_TFLITE_DELEGATE_XNNPACK
    if (helper_type_ == kTensorflowLiteXnnpack) {
        auto options = TfLiteXNNPackDelegateOptionsDefault();
        options.num_threads = num_threads_;
        delegate_ = TfLiteXNNPackDelegateCreate(&options);
        interpreter_->ModifyGraphWithDelegate(delegate_);
    }
#endif
#ifdef INFERENCE_HELPER_ENABLE_TFLITE_DELEGATE_GPU
    if (helper_type_ == kTensorflowLiteGpu) {
        auto options = TfLiteGpuDelegateOptionsV2Default();
        options.inference_preference = TFLITE_GPU_INFERENCE_PREFERENCE_SUSTAINED_SPEED;
        options.inference_priority1 = TFLITE_GPU_INFERENCE_PRIORITY_MIN_LATENCY;
        delegate_ = TfLiteGpuDelegateV2Create(&options);
        interpreter_->ModifyGraphWithDelegate(delegate_);
    }
#endif
#ifdef INFERENCE_HELPER_ENABLE_TFLITE_DELEGATE_EDGETPU
    if (helper_type_ == kTensorflowLiteEdgetpu) {
        size_t num_devices;
        std::unique_ptr<edgetpu_device, decltype(&edgetpu_free_devices)> devices(edgetpu_list_devices(&num_devices), &edgetpu_free_devices);
        if (num_devices > 0) {
            const auto& device = devices.get()[0];
            delegate_ = edgetpu_create_delegate(device.type, device.path, nullptr, 0);
            if (delegate_) {
                interpreter_->ModifyGraphWithDelegate(delegate_);
            } else {
                PRINT_E("[WARNING] Failed to create Edge TPU delegate\n");
            }
        } else {
            PRINT_E("[WARNING] Edge TPU is not found\n");
        }
    }
#endif
#ifdef INFERENCE_HELPER_ENABLE_TFLITE_DELEGATE_NNAPI
    if (helper_type_ == kTensorflowLiteNnapi) {
        interpreter_->SetNumThreads(1);
        tflite::StatefulNnApiDelegate::Options options;
        //options.execution_preference = tflite::StatefulNnApiDelegate::Options::kSustainedSpeed;
        //options.disallow_nnapi_cpu = true;
        //options.allow_fp16 = true;
        //options.accelerator_name = "qti-default";
        //options.accelerator_name = "qti-dsp";
        //options.accelerator_name = "qti-gpu";
        //options.accelerator_name = "nnapi-reference";
        delegate_ = new tflite::StatefulNnApiDelegate(options);
        if (delegate_) {
            interpreter_->ModifyGraphWithDelegate(delegate_);
            //interpreter_->SetAllowFp16PrecisionForFp32(true);
            auto actualOptions = tflite::StatefulNnApiDelegate::GetOptions(delegate_);
            PRINT("[INFO] NNAPI options.accelerator_name = %s\n", actualOptions.accelerator_name);
        } else {
            PRINT_E("[WARNING] Failed to create NNAPI delegate\n");
        }
    }
#endif
    /* Memo: If you get error around here in Visual Studio, please make sure you don't use Debug */
    if (interpreter_->AllocateTensors() != kTfLiteOk) {
        PRINT_E("Failed to allocate tensors (%s)\n", model_filename.c_str());
        return kRetErr;
    }

    /* Get model information */
    DisplayModelInfo(*interpreter_);

    /* Check if input tensor name exists anddims are the same as described in the model. In case dims is unfixed, resize tensor size. Get id and type */
    for (auto& input_tensor_info : input_tensor_info_list) {
        if (GetInputTensorInfo(input_tensor_info) != kRetOk) {
            PRINT_E("Invalid input tensor info (%s)\n", input_tensor_info.name.c_str());
            return kRetErr;
        }
    }
    
    /* Check if output tensor name exists and get info (id, ptr to data, dims, type) */
    for (auto& output_tensor_info : output_tensor_info_list) {
        if (GetOutputTensorInfo(output_tensor_info) != kRetOk) {
            PRINT_E("Invalid output tensor info (%s)\n", output_tensor_info.name.c_str());
            return kRetErr;
        }
    }

    /* Convert normalize parameter to speed up */
    for (auto& input_tensor_info : input_tensor_info_list) {
        ConvertNormalizeParameters(input_tensor_info);
    }

    return kRetOk;
};


int32_t InferenceHelperTensorflowLite::Finalize(void)
{
    model_.reset();
    resolver_.reset();
    interpreter_.reset();

#ifdef INFERENCE_HELPER_ENABLE_TFLITE_DELEGATE_EDGETPU
    if (helper_type_ == kTensorflowLiteEdgetpu) {
        edgetpu_free_delegate(delegate_);
    }
#endif
#ifdef INFERENCE_HELPER_ENABLE_TFLITE_DELEGATE_GPU
    if (helper_type_ == kTensorflowLiteGpu) {
        TfLiteGpuDelegateV2Delete(delegate_);
    }
#endif
#ifdef INFERENCE_HELPER_ENABLE_TFLITE_DELEGATE_XNNPACK
    if (helper_type_ == kTensorflowLiteXnnpack) {
        TfLiteXNNPackDelegateDelete(delegate_);
    }
#endif
#ifdef INFERENCE_HELPER_ENABLE_TFLITE_DELEGATE_NNAPI
    if (helper_type_ == kTensorflowLiteNnapi) {
        delete reinterpret_cast<tflite::StatefulNnApiDelegate*>(delegate_);
    }
#endif
    return kRetOk;
}

int32_t InferenceHelperTensorflowLite::PreProcess(const std::vector<InputTensorInfo>& input_tensor_info_list)
{
    if (interpreter_ == nullptr) {
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

            /* Normalize image */
            if (input_tensor_info.tensor_type == TensorInfo::kTensorTypeFp32) {
                float* dst = interpreter_->typed_tensor<float>(input_tensor_info.id);
                PreProcessImage(num_threads_, input_tensor_info, dst);
            } else if (input_tensor_info.tensor_type == TensorInfo::kTensorTypeUint8) {
                uint8_t* dst = interpreter_->typed_tensor<uint8_t>(input_tensor_info.id);
                PreProcessImage(num_threads_, input_tensor_info, dst);
            } else if (input_tensor_info.tensor_type == TensorInfo::kTensorTypeInt8) {
                int8_t* dst = interpreter_->typed_tensor<int8_t>(input_tensor_info.id);
                PreProcessImage(num_threads_, input_tensor_info, dst);
            } else {
                PRINT_E("Unsupported tensor_type (%d)\n", input_tensor_info.tensor_type);
                return kRetErr;
            }
        } else if ((input_tensor_info.data_type == InputTensorInfo::kDataTypeBlobNhwc) || (input_tensor_info.data_type == InputTensorInfo::kDataTypeBlobNchw)) {
            if (input_tensor_info.tensor_type == TensorInfo::kTensorTypeFp32) {
                float* dst = interpreter_->typed_tensor<float>(input_tensor_info.id);
                PreProcessBlob<float>(num_threads_, input_tensor_info, dst);
            } else if (input_tensor_info.tensor_type == TensorInfo::kTensorTypeUint8 || input_tensor_info.tensor_type == TensorInfo::kTensorTypeInt8) {
                uint8_t* dst = interpreter_->typed_tensor<uint8_t>(input_tensor_info.id);
                PreProcessBlob<uint8_t>(num_threads_, input_tensor_info, dst);
            } else if (input_tensor_info.tensor_type == TensorInfo::kTensorTypeInt32) {
                int32_t* dst = interpreter_->typed_tensor<int32_t>(input_tensor_info.id);
                PreProcessBlob<int32_t>(num_threads_, input_tensor_info, dst);
            } else {
                PRINT_E("Unsupported tensor_type (%d)\n", input_tensor_info.tensor_type);
                return kRetErr;
            }
        } else {
            PRINT_E("Unsupported data_type (%d)\n", input_tensor_info.data_type);
            return kRetErr;
        }
    }
    return kRetOk;
}

int32_t InferenceHelperTensorflowLite::Process(std::vector<OutputTensorInfo>& output_tensor_info_list)
{
    if (interpreter_->Invoke() != kTfLiteOk) {
        PRINT_E("Failed to invoke\n");
        return kRetErr;
    }
    return kRetOk;
}

void InferenceHelperTensorflowLite::DisplayModelInfo(const tflite::Interpreter& interpreter)
{
    /* Memo: If you get error around here in Visual Studio, please make sure you don't use Debug */
    const auto& input_indices = interpreter.inputs();
    int32_t input_num = static_cast<int32_t>(input_indices.size());
    PRINT("Input num = %d\n", input_num);
    for (int32_t i = 0; i < input_num; i++) {
        auto* tensor = interpreter.tensor(input_indices[i]);
        PRINT("    tensor[%d]->name: %s\n", i, tensor->name);
        for (int32_t j = 0; j < tensor->dims->size; j++) {
            PRINT("    tensor[%d]->dims->size[%d]: %d\n", i, j, tensor->dims->data[j]);
        }
        if (tensor->type == kTfLiteUInt8 || tensor->type == kTfLiteInt8) {
            PRINT("    tensor[%d]->type: quantized\n", i);
            PRINT("    tensor[%d]->params.zero_point, scale: %d, %f\n", i, tensor->params.zero_point, tensor->params.scale);
        } else {
            PRINT("    tensor[%d]->type: not quantized\n", i);
        }
    }

    const auto& output_indices = interpreter.outputs();
    int32_t output_num = static_cast<int32_t>(output_indices.size());
    PRINT("Output num = %d\n", output_num);
    for (int32_t i = 0; i < output_num; i++) {
        auto* tensor = interpreter.tensor(output_indices[i]);
        PRINT("    tensor[%d]->name: %s\n", i, tensor->name);
        for (int32_t j = 0; j < tensor->dims->size; j++) {
            PRINT("    tensor[%d]->dims->size[%d]: %d\n", i, j, tensor->dims->data[j]);
        }
        if (tensor->type == kTfLiteUInt8 || tensor->type == kTfLiteInt8) {
            PRINT("    tensor[%d]->type: quantized\n", i);
            PRINT("    tensor[%d]->params.zero_point, scale: %d, %f\n", i, tensor->params.zero_point, tensor->params.scale);
        } else {
            PRINT("    tensor[%d]->type: not quantized\n", i);
        }
    }
}


int32_t InferenceHelperTensorflowLite::GetInputTensorInfo(InputTensorInfo& tensor_info)
{
    for (auto i : interpreter_->inputs()) {
        TfLiteTensor* tensor = interpreter_->tensor(i);
        if (std::string(tensor->name) == tensor_info.name) {
            tensor_info.id = i;
            
            bool is_model_size_fixed = true;
            for (int32_t i = 0; i < tensor->dims->size; i++) {
                if (tensor->dims->data[i] == -1) is_model_size_fixed = false;
            }
            bool is_size_assigned = true;
            if (tensor_info.tensor_dims.empty()) is_size_assigned = false;
            
            if (!is_model_size_fixed && !is_size_assigned) {
                PRINT_E("Model input size is not set\n");
                return kRetErr;
            }
            if (is_model_size_fixed && is_size_assigned) {
                bool is_size_ok = true;
                if (static_cast<int32_t>(tensor_info.tensor_dims.size()) == tensor->dims->size) {
                    for (int32_t d = 0; d < static_cast<int32_t>(tensor_info.tensor_dims.size()); d++) {
                        if (tensor_info.tensor_dims[d] != tensor->dims->data[d]) is_size_ok = false;
                    }
                } else {
                    is_size_ok = false;
                }
                if (!is_size_ok) {
                    PRINT_E("Invalid input size\n");
                    //return kRetErr;
                    /* Try to resize tensor size */
                    is_model_size_fixed = false;
                }
            }
            if (is_model_size_fixed && !is_size_assigned) {
                PRINT("Input tensor size is set from the model\n");
                tensor_info.tensor_dims.clear();
                for (int32_t d = 0; d < tensor->dims->size; d++) {
                    tensor_info.tensor_dims.push_back(tensor->dims->data[d]);
                }
            }
            if (!is_model_size_fixed && is_size_assigned) {
                PRINT("[WARNING] ResizeInputTensor is not tested\n");
                interpreter_->ResizeInputTensor(i, tensor_info.tensor_dims);
                if (interpreter_->AllocateTensors() != kTfLiteOk) {
                    PRINT_E("Failed to allocate tensors\n");
                    return kRetErr;
                }
            }

            if (tensor->type == kTfLiteUInt8) tensor_info.tensor_type = TensorInfo::kTensorTypeUint8;
            if (tensor->type == kTfLiteInt8) tensor_info.tensor_type = TensorInfo::kTensorTypeInt8;
            if (tensor->type == kTfLiteFloat32) tensor_info.tensor_type = TensorInfo::kTensorTypeFp32;
            if (tensor->type == kTfLiteInt32) tensor_info.tensor_type = TensorInfo::kTensorTypeInt32;
            if (tensor->type == kTfLiteInt64) tensor_info.tensor_type = TensorInfo::kTensorTypeInt64;
            return kRetOk;
        }
    }

    PRINT_E("Invalid name (%s) \n", tensor_info.name.c_str());
    return kRetErr;
}

int32_t InferenceHelperTensorflowLite::GetOutputTensorInfo(OutputTensorInfo& tensor_info)
{
    for (auto i : interpreter_->outputs()) {
        const TfLiteTensor* tensor = interpreter_->tensor(i);
        if (std::string(tensor->name) == tensor_info.name) {
            tensor_info.id = i;
            tensor_info.tensor_dims.clear();
            for (int32_t d = 0; d < tensor->dims->size; d++) {
                tensor_info.tensor_dims.push_back(tensor->dims->data[d]);
            }

            switch (tensor->type) {
            case kTfLiteUInt8:
                tensor_info.tensor_type = TensorInfo::kTensorTypeUint8;
                tensor_info.data = interpreter_->typed_tensor<uint8_t>(i);
                tensor_info.quant.scale = tensor->params.scale;
                tensor_info.quant.zero_point = tensor->params.zero_point;
                break;
            case kTfLiteInt8:
                tensor_info.tensor_type = TensorInfo::kTensorTypeInt8;
                tensor_info.data = interpreter_->typed_tensor<int8_t>(i);
                tensor_info.quant.scale = tensor->params.scale;
                tensor_info.quant.zero_point = tensor->params.zero_point;
                break;
            case kTfLiteFloat32:
                tensor_info.tensor_type = TensorInfo::kTensorTypeFp32;
                tensor_info.data = interpreter_->typed_tensor<float>(i);
                break;
            case kTfLiteInt32:
                tensor_info.tensor_type = TensorInfo::kTensorTypeInt32;
                tensor_info.data = interpreter_->typed_tensor<int32_t>(i);
                break;
            case kTfLiteInt64:
                tensor_info.tensor_type = TensorInfo::kTensorTypeInt64;
                tensor_info.data = interpreter_->typed_tensor<int64_t>(i);
                break;
            default:
                return kRetErr;
            }
            return kRetOk;;
        }
    }
    PRINT_E("Invalid name (%s) \n", tensor_info.name.c_str());
    return kRetErr;

}


static TfLiteFloatArray* TfLiteFloatArrayCopy(const TfLiteFloatArray* src)
{
    if (!src) return nullptr;
    TfLiteFloatArray* ret = static_cast<TfLiteFloatArray*>(malloc(TfLiteFloatArrayGetSizeInBytes(src->size)));
    if (!ret) return nullptr;
    ret->size = src->size;
    std::memcpy(ret->data, src->data, src->size * sizeof(float));
    return ret;
}

int32_t InferenceHelperTensorflowLite::SetBufferToTensor(int32_t index, void *data)
{
    const TfLiteTensor* tensor = interpreter_->tensor(index);
    const int32_t model_input_height = tensor->dims->data[1];
    const int32_t model_input_width = tensor->dims->data[2];
    const int32_t model_input_channel = tensor->dims->data[3];

    if (tensor->type == kTfLiteUInt8) {
        int32_t data_size = sizeof(int8_t) * 1 * model_input_height * model_input_width * model_input_channel;
        /* Need deep copy quantization parameters */
        /* reference: https://github.com/google-coral/edgetpu/blob/master/src/cpp/basic/basic_engine_native.cc */
        /* todo: do I need to release allocated memory ??? */
        const TfLiteAffineQuantization* input_quant_params = reinterpret_cast<TfLiteAffineQuantization*>(tensor->quantization.params);
        TfLiteQuantization input_quant_clone;
        input_quant_clone = tensor->quantization;
        TfLiteAffineQuantization* input_quant_params_clone = reinterpret_cast<TfLiteAffineQuantization*>(malloc(sizeof(TfLiteAffineQuantization)));
        input_quant_params_clone->scale = TfLiteFloatArrayCopy(input_quant_params->scale);
        input_quant_params_clone->zero_point = TfLiteIntArrayCopy(input_quant_params->zero_point);
        input_quant_params_clone->quantized_dimension = input_quant_params->quantized_dimension;
        input_quant_clone.params = input_quant_params_clone;

        interpreter_->SetTensorParametersReadOnly(
            index, tensor->type, tensor->name,
            std::vector<int32_t>(tensor->dims->data, tensor->dims->data + tensor->dims->size),
            input_quant_clone,	// use copied parameters
            (const char*)data, data_size);
    } else {
        int32_t data_size = sizeof(float) * 1 * model_input_height * model_input_width * model_input_channel;
        interpreter_->SetTensorParametersReadOnly(
            index, tensor->type, tensor->name,
            std::vector<int32_t>(tensor->dims->data, tensor->dims->data + tensor->dims->size),
            tensor->quantization,
            (const char*)data, data_size);
    }
    return 0;
}

