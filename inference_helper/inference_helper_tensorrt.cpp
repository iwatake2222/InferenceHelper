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

/* for TensorRT */
#include <cuda_runtime_api.h>
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include "tensorrt/common.h"

/* for My modules */
#include "inference_helper_log.h"
#include "inference_helper_tensorrt.h"

/*** Macro ***/
#define TAG "InferenceHelperTensorRt"
#define PRINT(...)   INFERENCE_HELPER_LOG_PRINT(TAG, __VA_ARGS__)
#define PRINT_E(...) INFERENCE_HELPER_LOG_PRINT_E(TAG, __VA_ARGS__)

/* Setting */
#define USE_FP16
// #define USE_INT8_WITHOUT_CALIBRATION
// #define USE_INT8_WITH_CALIBRATION

#define OPT_MAX_WORK_SPACE_SIZE ((size_t)1 << 30)
#define OPT_AVG_TIMING_ITERATIONS 1
#define OPT_MIN_TIMING_ITERATIONS 1

#ifdef USE_INT8_WITH_CALIBRATION
/* ★ Modify the following (use the same parameter as the model. Also, ppm must be the same size but not normalized.) */
#define CAL_DIR        "../../InferenceHelper/inference_helper/tensorrt/calibration/sample_ppm"
#define CAL_LIST_FILE  "list.txt"
#define CAL_BATCH_SIZE 10
#define CAL_NB_BATCHES 2
#define CAL_IMAGE_C    3
#define CAL_IMAGE_H    224
#define CAL_IMAGE_W    224
/* 0 ~ 1.0 */
// #define CAL_SCALE      (1.0 / 255.0)
// #define CAL_BIAS       (0.0)
/* -2.25 ~ 2.25 */
#define CAL_SCALE      (1.0 / (255.0 * 0.225))
#define CAL_BIAS       (0.45 / 0.225)

/* include BatchStream.h after defining parameters */
#include "tensorrt/BatchStream.h"
#include "tensorrt/EntropyCalibrator.h"
#endif

/*** Function ***/
InferenceHelperTensorRt::~InferenceHelperTensorRt() = default;

InferenceHelperTensorRt::InferenceHelperTensorRt()
{
    num_threads_ = 1;
    dla_core_ = -1;
}

int32_t InferenceHelperTensorRt::SetNumThreads(const int32_t num_threads)
{
    num_threads_ = num_threads;
    return kRetOk;
}

int32_t InferenceHelperTensorRt::SetCustomOps(const std::vector<std::pair<const char*, const void*>>& custom_ops)
{
    PRINT("[WARNING] This method is not supported\n");
    return kRetOk;
}

int32_t InferenceHelperTensorRt::Initialize(const std::string& model_filename, std::vector<InputTensorInfo>& input_tensor_info_list, std::vector<OutputTensorInfo>& output_tensor_info_list)
{
    /*** check model format ***/
    bool is_trt_model = false;
    bool is_onnx_model = false;
    // bool isUffModel = false;	// todo
    std::string trt_model_filename = std::string(model_filename);
    if (model_filename.find(".onnx") != std::string::npos) {
        is_onnx_model = true;
        trt_model_filename = trt_model_filename.replace(trt_model_filename.find(".onnx"), std::string(".onnx").length(), ".trt\0");
        std::ifstream ifs(trt_model_filename);
        if (ifs.is_open()) {
            is_onnx_model = false;
            is_trt_model = true;
        }
    } else if (trt_model_filename.find(".trt") != std::string::npos) {
        is_trt_model = true;
    } else {
        PRINT_E("unsupoprted file format (%s)\n", model_filename.c_str());
        return kRetErr;
    }

    /*** create runtime ***/
    runtime_ = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(sample::gLogger.getTRTLogger()));
    if (!runtime_) {
        PRINT_E("Failed to create runtime (%s)\n", model_filename.c_str());
        return kRetErr;
    }

    /*** create engine from model file ***/
    if (is_trt_model) {
        /* Just load TensorRT model (serialized model) */
        std::string buffer;
        std::ifstream stream(trt_model_filename, std::ios::binary);
        if (stream) {
            stream >> std::noskipws;
            copy(std::istream_iterator<char>(stream), std::istream_iterator<char>(), back_inserter(buffer));
        }

        engine_ = std::unique_ptr<nvinfer1::ICudaEngine>(runtime_->deserializeCudaEngine(buffer.data(), buffer.size()));
        stream.close();
        if (!engine_) {
            PRINT_E("Failed to create engine (%s)\n", trt_model_filename.c_str());
            return kRetErr;
        }
    } else if (is_onnx_model) {
        /* Create a TensorRT model from another format */
        auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()));
        const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
        auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
        auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());

        auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, sample::gLogger.getTRTLogger()));
        if (!parser->parseFromFile(model_filename.c_str(), (int)nvinfer1::ILogger::Severity::kWARNING)) {
            PRINT_E("Failed to parse onnx file (%s)", model_filename.c_str());
            return kRetErr;
        }

        // builder->setMaxBatchSize(1);
        config->setMaxWorkspaceSize(OPT_MAX_WORK_SPACE_SIZE);
        // config->setAvgTimingIterations(OPT_AVG_TIMING_ITERATIONS);
        // config->setMinTimingIterations(OPT_MIN_TIMING_ITERATIONS);

#if defined(USE_FP16)
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
#elif defined(USE_INT8_WITHOUT_CALIBRATION)
        config->setFlag(nvinfer1::BuilderFlag::kINT8);
        samplesCommon::setAllDynamicRanges(network.get(), 2.5f, 2.5f);
#elif defined(USE_INT8_WITH_CALIBRATION)
        config->setFlag(nvinfer1::BuilderFlag::kINT8);
        std::vector<std::string> data_dirs;
        data_dirs.push_back(CAL_DIR);
        nvinfer1::Dims4  image_dims{CAL_BATCH_SIZE, CAL_IMAGE_C, CAL_IMAGE_H, CAL_IMAGE_W};
        BatchStream calibration_stream(CAL_BATCH_SIZE, CAL_NB_BATCHES, image_dims, CAL_LIST_FILE, data_dirs);
        auto calibrator = std::unique_ptr<nvinfer1::IInt8Calibrator>(new Int8EntropyCalibrator2<BatchStream>(calibration_stream, 0, "_cal.txt", input_tensor_info_list[0].name.c_str()));
        config->setInt8Calibrator(calibrator.get());
#endif 

        if (dla_core_ >= 0) {
            PRINT("Use DLA: %d\n", dla_core_);
            samplesCommon::enableDLA(builder.get(), config.get(), dla_core_);
        }

        auto plan = std::unique_ptr<nvinfer1::IHostMemory>(builder->buildSerializedNetwork(*network, *config));
        if (!plan) {
            PRINT_E("Failed to create plan (%s)\n", model_filename.c_str());
            return kRetErr;
        }

        engine_ = std::unique_ptr<nvinfer1::ICudaEngine>(runtime_->deserializeCudaEngine(plan->data(), plan->size()));
        if (!engine_) {
            PRINT_E("Failed to create engine (%s)\n", model_filename.c_str());
            return kRetErr;
        }

        /* save serialized model for next time */
        std::ofstream ofs(std::string(trt_model_filename), std::ios::out | std::ios::binary);
        ofs.write((char*)(plan->data()), plan->size());
        ofs.close();
    }


    context_ = std::unique_ptr<nvinfer1::IExecutionContext>(engine_->createExecutionContext());
    if (!context_) {
        PRINT_E("Failed to create context (%s)\n", model_filename.c_str());
        return kRetErr;
    }

    /* Allocate host/device buffers and assign to tensor info */
    for (auto& input_tensor_info : input_tensor_info_list) {
        input_tensor_info.id = -1;	// not assigned
    }
    for (auto& output_tensor_info : output_tensor_info_list) {
        output_tensor_info.id = -1;	// not assigned
    }
    if (AllocateBuffers(input_tensor_info_list, output_tensor_info_list) != kRetOk) {
        return kRetErr;
    }
    /* Check if the tensor is assigned (exists in the model) */
    for (auto& input_tensor_info : input_tensor_info_list) {
        if (input_tensor_info.id == -1) {
            PRINT_E("Input tensor doesn't exist in the model (%s)\n", input_tensor_info.name.c_str());
            return kRetErr;
        }
    }
    for (auto& output_tensor_info : output_tensor_info_list) {
        if (output_tensor_info.id == -1) {
            PRINT_E("Output tensor doesn't exist in the model (%s)\n", output_tensor_info.name.c_str());
            return kRetErr;
        }
    }

    /* Convert normalize parameter to speed up */
    for (auto& input_tensor_info : input_tensor_info_list) {
        ConvertNormalizeParameters(input_tensor_info);
    }

    return kRetOk;
}

int InferenceHelperTensorRt::Finalize(void)
{
    int num_of_in_out = engine_->getNbBindings();
    for (int i = 0; i < num_of_in_out; i++) {
        const auto data_type = engine_->getBindingDataType(i);
        switch (data_type) {
        case nvinfer1::DataType::kFLOAT:
        case nvinfer1::DataType::kHALF:
        case nvinfer1::DataType::kINT32:
            delete[] (float*)(buffer_list_cpu_[i].first);
            break;
        case nvinfer1::DataType::kINT8:
            delete[] (int*)(buffer_list_cpu_[i].first);
            break;
        default:
            return kRetErr;
        }
    }

    for (auto p : buffer_list_gpu_) {
        cudaFree(p);
    }

    return kRetOk;
}

int32_t InferenceHelperTensorRt::PreProcess(const std::vector<InputTensorInfo>& input_tensor_info_list)
{
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
                float *dst = (float*)(buffer_list_cpu_[input_tensor_info.id].first);
                PreProcessImage(num_threads_, input_tensor_info, dst);
            } else if (input_tensor_info.tensor_type == TensorInfo::kTensorTypeUint8) {
                uint8_t *dst = (uint8_t*)(buffer_list_cpu_[input_tensor_info.id].first);
                PreProcessImage(num_threads_, input_tensor_info, dst);
            } else if (input_tensor_info.tensor_type == TensorInfo::kTensorTypeInt8) {
                int8_t *dst = (int8_t*)(buffer_list_cpu_[input_tensor_info.id].first);
                PreProcessImage(num_threads_, input_tensor_info, dst);
            } else {
                PRINT_E("Unsupported tensor_type (%d)\n", input_tensor_info.tensor_type);
                return kRetErr;
            }
        } else if ((input_tensor_info.data_type == InputTensorInfo::kDataTypeBlobNhwc) || (input_tensor_info.data_type == InputTensorInfo::kDataTypeBlobNchw)) {
            if (input_tensor_info.tensor_type == TensorInfo::kTensorTypeFp32) {
                float *dst = (float*)(buffer_list_cpu_[input_tensor_info.id].first);
                PreProcessBlob<float>(num_threads_, input_tensor_info, dst);
            } else if (input_tensor_info.tensor_type == TensorInfo::kTensorTypeUint8 || input_tensor_info.tensor_type == TensorInfo::kTensorTypeInt8) {
                uint8_t *dst = (uint8_t*)(buffer_list_cpu_[input_tensor_info.id].first);
                PreProcessBlob<uint8_t>(num_threads_, input_tensor_info, dst);
            } else if (input_tensor_info.tensor_type == TensorInfo::kTensorTypeInt32) {
                int32_t *dst = (int32_t*)(buffer_list_cpu_[input_tensor_info.id].first);
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

int32_t InferenceHelperTensorRt::Process(std::vector<OutputTensorInfo>& output_tensor_info_list)
{
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    for (int i = 0; i < (int)buffer_list_cpu_.size(); i++) {
        if (engine_->bindingIsInput(i)) {
            cudaMemcpyAsync(buffer_list_gpu_[i], buffer_list_cpu_[i].first, buffer_list_cpu_[i].second, cudaMemcpyHostToDevice, stream);
        }
    }
    context_->enqueue(1, &buffer_list_gpu_[0], stream, NULL);
    for (int i = 0; i < (int)buffer_list_cpu_.size(); i++) {
        if (!engine_->bindingIsInput(i)) {
            cudaMemcpyAsync(buffer_list_cpu_[i].first, buffer_list_gpu_[i], buffer_list_cpu_[i].second, cudaMemcpyDeviceToHost, stream);
        }
    }
    cudaStreamSynchronize(stream);

    cudaStreamDestroy(stream);

    (void)output_tensor_info_list;	// no need to set output data, because the ptr to output data is already set at initialize

    return kRetOk;
}

int32_t InferenceHelperTensorRt::AllocateBuffers(std::vector<InputTensorInfo>& input_tensor_info_list, std::vector<OutputTensorInfo>& output_tensor_info_list)
{
    int32_t num_of_in_out = engine_->getNbBindings();
    PRINT("num_of_in_out = %d\n", num_of_in_out);

    for (int32_t i = 0; i < num_of_in_out; i++) {
        PRINT("tensor[%d]->name: %s\n", i, engine_->getBindingName(i));
        PRINT("  is input = %d\n", engine_->bindingIsInput(i));
        int32_t data_size = 1;
        const auto dims = engine_->getBindingDimensions(i);
        for (int32_t i = 0; i < dims.nbDims; i++) {
            PRINT("  dims.d[%d] = %d\n", i, dims.d[i]);
            data_size *= dims.d[i];
        }
        const auto data_type = engine_->getBindingDataType(i);
        PRINT("  data_type = %d\n", static_cast<int32_t>(data_type));

        void* buffer_cpu = nullptr;
        void* buffer_gpu = nullptr;
        switch (data_type) {
        case nvinfer1::DataType::kFLOAT:
        case nvinfer1::DataType::kHALF:
        case nvinfer1::DataType::kINT32:
            buffer_cpu = new float[data_size];
            buffer_list_cpu_.push_back(std::pair<void*,int32_t>(buffer_cpu, data_size * sizeof(float)));
            cudaMalloc(&buffer_gpu, data_size * sizeof(float));
            buffer_list_gpu_.push_back(buffer_gpu);
            break;
        case nvinfer1::DataType::kINT8:
            buffer_cpu = new int8_t[data_size];
            buffer_list_cpu_.push_back(std::pair<void*,int32_t>(buffer_cpu, data_size * sizeof(int8_t)));
            cudaMalloc(&buffer_gpu, data_size * sizeof(int8_t));
            buffer_list_gpu_.push_back(buffer_gpu);
            break;
        default:
            PRINT_E("Unsupported datatype (%d)\n", static_cast<int32_t>(data_type));
            return kRetErr;
        }

        if(engine_->bindingIsInput(i)) {
            for (auto& input_tensor_info : input_tensor_info_list) {
                int32_t id = engine_->getBindingIndex(input_tensor_info.name.c_str());
                if (id == i) {
                    input_tensor_info.id = id;
                    if (input_tensor_info.tensor_dims.empty()) {
                        /* tensor size is not assigned. so get from the model */
                        for (int32_t i = 0; i < dims.nbDims; i++) {
                            input_tensor_info.tensor_dims.push_back(dims.d[i]);
                        }
                    } else {
                        /* tensor size is assigned. so check if it's the same as size defined in the model */
                        if (static_cast<int32_t>(input_tensor_info.tensor_dims.size()) != dims.nbDims) {
                            PRINT_E("Input Tensor dims doesn't match\n");
                            return kRetErr;
                        }
                        for (int32_t i = 0; i < dims.nbDims; i++) {
                            if (input_tensor_info.tensor_dims[i] != dims.d[i]) {
                                PRINT_E("Input Tensor size doesn't match\n");
                                return kRetErr;
                            }
                        }
                    }
                    if (((input_tensor_info.tensor_type == TensorInfo::kTensorTypeUint8) && (data_type == nvinfer1::DataType::kINT8))
                        || ((input_tensor_info.tensor_type == TensorInfo::kTensorTypeFp32) && (data_type == nvinfer1::DataType::kFLOAT))
                        || ((input_tensor_info.tensor_type == TensorInfo::kTensorTypeInt32) && (data_type == nvinfer1::DataType::kINT32))) {
                            /* OK */
                    } else {
                        PRINT_E("Input Tensor type doesn't match\n");
                        return kRetErr;
                    }
                }
            }
        } else {
            for (auto& output_tensor_info : output_tensor_info_list) {
                int32_t id = engine_->getBindingIndex(output_tensor_info.name.c_str());
                if (id == i) {
                    output_tensor_info.id = id;
                    if (output_tensor_info.tensor_dims.empty()) {
                        /* tensor size is not assigned. so get from the model */
                        for (int32_t i = 0; i < dims.nbDims; i++) {
                            output_tensor_info.tensor_dims.push_back(dims.d[i]);
                        }
                    } else {
                        /* tensor size is assigned. so check if it's the same as size defined in the model */
                        if (static_cast<int32_t>(output_tensor_info.tensor_dims.size()) != dims.nbDims) {
                            PRINT_E("Output Tensor dims doesn't match\n");
                            return kRetErr;
                        }
                        for (int32_t i = 0; i < dims.nbDims; i++) {
                            if (output_tensor_info.tensor_dims[i] != dims.d[i]) {
                                PRINT_E("Output Tensor size doesn't match\n");
                                return kRetErr;
                            }
                        }
                    }

                    if (((output_tensor_info.tensor_type == TensorInfo::kTensorTypeUint8) && (data_type == nvinfer1::DataType::kINT8))
                        || ((output_tensor_info.tensor_type == TensorInfo::kTensorTypeFp32) && (data_type == nvinfer1::DataType::kFLOAT))
                        || ((output_tensor_info.tensor_type == TensorInfo::kTensorTypeInt32) && (data_type == nvinfer1::DataType::kINT32))) {
                            /* OK */
                    } else {
                        PRINT_E("Output Tensor type doesn't match\n");
                        return kRetErr;
                    }
                    if (data_type == nvinfer1::DataType::kINT8) {
                        output_tensor_info.quant.scale = 1.0;			// todo
                        output_tensor_info.quant.zero_point = 0.0;
                    }
                    output_tensor_info.data = buffer_cpu;
                }
            }
        }
    }

    return kRetOk;
}
