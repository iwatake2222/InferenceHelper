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
#include "common_helper.h"
#include "inference_helper_tensorrt.h"

/*** Macro ***/
#define TAG "InferenceHelperTensorRt"
#define PRINT(...)   COMMON_HELPER_PRINT(TAG, __VA_ARGS__)
#define PRINT_E(...) COMMON_HELPER_PRINT_E(TAG, __VA_ARGS__)

/* Setting */
#define USE_FP16
// #define USE_INT8

#define OPT_MAX_WORK_SPACE_SIZE (1 << 30)
#define OPT_AVG_TIMING_ITERATIONS 4
#define OPT_MIN_TIMING_ITERATIONS 2

#ifdef USE_INT8
/* ★ Modify the following (use the same parameter as the model. Also, ppm must be the same size but not normalized.) */
#define CAL_DIR        "../../InferenceHelper/inference_helper/tensorrt//calibration/sample_ppm"
#define CAL_LIST_FILE  "list.txt"
#define CAL_INPUT_NAME "data"
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
InferenceHelperTensorRt::InferenceHelperTensorRt()
{
    num_threads_ = 1;
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
    /* check model format */
    bool is_trt_model = false;
    bool is_onnx_model = false;
    // bool isUffModel = false;	// todo
    std::string trt_model_filename = std::string(model_filename);
    if (model_filename.find(".onnx") != std::string::npos) {
        is_onnx_model = true;
        trt_model_filename = trt_model_filename.replace(trt_model_filename.find(".onnx"), std::string(".onnx").length(), ".trt\0");
    } else if (trt_model_filename.find(".trt") != std::string::npos) {
        is_trt_model = true;
    } else {
        PRINT_E("unsupoprted file format (%s)\n", model_filename.c_str());
        return kRetErr;
    }

    /* create runtime and engine from model file */
    if (is_trt_model) {
        std::string buffer;
        std::ifstream stream(model_filename, std::ios::binary);
        if (stream) {
            stream >> std::noskipws;
            copy(std::istream_iterator<char>(stream), std::istream_iterator<char>(), back_inserter(buffer));
        }
        runtime_ = std::shared_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(sample::gLogger.getTRTLogger()), samplesCommon::InferDeleter());
        engine_ = std::shared_ptr<nvinfer1::ICudaEngine>(runtime_->deserializeCudaEngine(buffer.data(), buffer.size(), NULL), samplesCommon::InferDeleter());
        stream.close();
        if (!engine_) {
            PRINT_E("Failed to create engine (%s)\n", model_filename.c_str());
            return kRetErr;
        }
        context_ = std::shared_ptr<nvinfer1::IExecutionContext>(engine_->createExecutionContext(), samplesCommon::InferDeleter());
        if (!context_) {
            PRINT_E("Failed to create context (%s)\n", model_filename.c_str());
            return kRetErr;
        }
    } else if (is_onnx_model) {
        /* create a TensorRT model from another format */
        auto builder = std::shared_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()), samplesCommon::InferDeleter());
#if 0
        /* For older version of JetPack */
        auto network = std::shared_ptr<nvinfer1::INetworkDefinition>(builder->createNetwork(), samplesCommon::InferDeleter());
#else
        const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
        auto network = std::shared_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch), samplesCommon::InferDeleter());
#endif
        auto config = std::shared_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig(), samplesCommon::InferDeleter());

        auto parser_onnx = std::shared_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, sample::gLogger.getTRTLogger()), samplesCommon::InferDeleter());
        if (!parser_onnx->parseFromFile(model_filename.c_str(), (int)nvinfer1::ILogger::Severity::kWARNING)) {
            PRINT_E("Failed to parse onnx file (%s)", model_filename.c_str());
            return kRetErr;
        }

        builder->setMaxBatchSize(1);
        config->setMaxWorkspaceSize(OPT_MAX_WORK_SPACE_SIZE);
        config->setAvgTimingIterations(OPT_AVG_TIMING_ITERATIONS);
        config->setMinTimingIterations(OPT_MIN_TIMING_ITERATIONS) ;

#if defined(USE_FP16)
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
#elif defined(USE_INT8)
        config->setFlag(nvinfer1::BuilderFlag::kINT8);
        std::vector<std::string> data_dirs;
        data_dirs.push_back(CAL_DIR);
        nvinfer1::DimsNCHW image_dims{CAL_BATCH_SIZE, CAL_IMAGE_C, CAL_IMAGE_H, CAL_IMAGE_W};
        BatchStream calibration_stream(CAL_BATCH_SIZE, CAL_NB_BATCHES, image_dims, CAL_LIST_FILE, data_dirs);
        auto calibrator = std::unique_ptr<nvinfer1::IInt8Calibrator>(new Int8EntropyCalibrator2<BatchStream>(calibration_stream, 0, "my_model", CAL_INPUT_NAME));
        config->setInt8Calibrator(calibrator.get());
#endif 

        engine_ = std::shared_ptr<nvinfer1::ICudaEngine>(builder->buildEngineWithConfig(*network, *config), samplesCommon::InferDeleter());
        if (!engine_) {
            PRINT_E("Failed to create engine (%s)\n", model_filename.c_str());
            return kRetErr;
        }
        context_ = std::shared_ptr<nvinfer1::IExecutionContext>(engine_->createExecutionContext(), samplesCommon::InferDeleter());
        if (!context_) {
            PRINT_E("Failed to create context (%s)\n", model_filename.c_str());
            return kRetErr;
        }
#if 1
        /* save serialized model for next time */
        nvinfer1::IHostMemory* trt_model_stream = engine_->serialize();
        std::ofstream ofs(std::string(trt_model_filename), std::ios::out | std::ios::binary);
        ofs.write((char*)(trt_model_stream->data()), trt_model_stream->size());
        ofs.close();
        trt_model_stream->destroy();
#endif
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
            delete[] (float*)(buffer_list_cpu_reserved_[i].first);
            break;
        case nvinfer1::DataType::kINT8:
            delete[] (int*)(buffer_list_cpu_reserved_[i].first);
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
        if (input_tensor_info.data_type == InputTensorInfo::kDataTypeImage) {
            if ((input_tensor_info.image_info.width != input_tensor_info.image_info.crop_width) || (input_tensor_info.image_info.height != input_tensor_info.image_info.crop_height)) {
                PRINT_E("Crop is not supported\n");
                return  kRetErr;
            }
            if ((input_tensor_info.image_info.crop_width != input_tensor_info.GetWidth()) || (input_tensor_info.image_info.crop_height != input_tensor_info.GetHeight())) {
                PRINT_E("Resize is not supported\n");
                return  kRetErr;
            }
            if (input_tensor_info.image_info.channel != input_tensor_info.GetChannel()) {
                PRINT_E("Color conversion is not supported\n");
                return  kRetErr;
            }

            /* Normalize image */
            if (input_tensor_info.tensor_type == TensorInfo::kTensorTypeFp32) {
                /* convert NHWC to NCHW */
                float *dst = (float*)(buffer_list_cpu_[input_tensor_info.id].first);
                uint8_t *src = (uint8_t*)(input_tensor_info.data);
                if (buffer_list_cpu_[input_tensor_info.id].second != 4 * input_tensor_info.image_info.width * input_tensor_info.image_info.height * input_tensor_info.image_info.channel) {
                    PRINT_E("Data size doesn't match\n");
                    return  kRetErr;
                }
#pragma omp parallel for num_threads(num_threads_)
                for (int32_t c = 0; c < input_tensor_info.GetChannel(); c++) {
                    for (int32_t i = 0; i < input_tensor_info.GetWidth() * input_tensor_info.GetHeight(); i++) {
#if 1
                        dst[c * input_tensor_info.GetWidth() * input_tensor_info.GetHeight() + i] = 
                            (src[i * input_tensor_info.GetChannel() + c] - input_tensor_info.normalize.mean[c]) * input_tensor_info.normalize.norm[c];
#else
                        dst[c * input_tensor_info.GetWidth() * input_tensor_info.GetHeight() + i] = 
                            (src[i * input_tensor_info.GetChannel() + c] / 255.0f - input_tensor_info.normalize.mean[c]) / input_tensor_info.normalize.norm[c];
#endif
                    }
                }
            } else if (input_tensor_info.tensor_type == TensorInfo::kTensorTypeUint8) {
                /* convert NHWC to NCHW */
                uint8_t *dst = (uint8_t*)(buffer_list_cpu_[input_tensor_info.id].first);
                uint8_t *src = (uint8_t*)(input_tensor_info.data);
                if (buffer_list_cpu_[input_tensor_info.id].second != 1 * input_tensor_info.image_info.width * input_tensor_info.image_info.height * input_tensor_info.image_info.channel) {
                    PRINT_E("Data size doesn't match\n");
                    return  kRetErr;
                }
#pragma omp parallel for num_threads(num_threads_)
                for (int32_t c = 0; c < input_tensor_info.GetChannel(); c++) {
                    for (int32_t i = 0; i < input_tensor_info.GetWidth() * input_tensor_info.GetHeight(); i++) {
                        dst[c * input_tensor_info.GetWidth() * input_tensor_info.GetHeight() + i] = src[i * input_tensor_info.GetChannel() + c];
                    }
                }
            } else {
                PRINT_E("Unsupported tensor_type (%d)\n", input_tensor_info.tensor_type);
                return kRetErr;
            }

        } else if (input_tensor_info.data_type == InputTensorInfo::kDataTypeBlobNhwc) {
                /* convert NHWC to NCHW */
                uint8_t *dst = (uint8_t*)(buffer_list_cpu_[input_tensor_info.id].first);
                uint8_t *src = (uint8_t*)(input_tensor_info.data);
#pragma omp parallel for num_threads(num_threads_)
                for (int32_t c = 0; c < input_tensor_info.GetChannel(); c++) {
                    for (int32_t i = 0; i < input_tensor_info.GetWidth() * input_tensor_info.GetHeight(); i++) {
                        dst[c * input_tensor_info.GetWidth() * input_tensor_info.GetHeight() + i] = src[i * input_tensor_info.GetChannel() + c];
                    }
                }
        } else if (input_tensor_info.data_type == InputTensorInfo::kDataTypeBlobNchw) {
            uint8_t *dst = (uint8_t*)(buffer_list_cpu_[input_tensor_info.id].first);
            uint8_t *src = (uint8_t*)(input_tensor_info.data);
            memcpy(dst, src, buffer_list_cpu_[input_tensor_info.id].second);
        } else {
            PRINT_E("Unsupported tensor_type (%d)\n", input_tensor_info.tensor_type);
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

void InferenceHelperTensorRt::ConvertNormalizeParameters(InputTensorInfo& tensor_info)
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
