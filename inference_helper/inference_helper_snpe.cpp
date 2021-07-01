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

/* for SNPE */
#include "snpe/SNPE.hpp"
#include "snpe/SNPEFactory.hpp"
#include "snpe/SNPEBuilder.hpp"
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
#include "udlExample.hpp"
#include "CreateUserBuffer.hpp"

/* for My modules */
#include "common_helper.h"
#include "inference_helper_snpe.h"

/*** Macro ***/
#define TAG "InferenceHelperSnpe"
#define PRINT(...)   COMMON_HELPER_PRINT(TAG, __VA_ARGS__)
#define PRINT_E(...) COMMON_HELPER_PRINT_E(TAG, __VA_ARGS__)


/*** Function ***/
InferenceHelperSnpe::InferenceHelperSnpe()
{
	m_numThread = 1;
	m_inputMap.reset(new zdl::DlSystem::UserBufferMap());
	m_outputMap.reset(new zdl::DlSystem::UserBufferMap());
}

InferenceHelperSnpe::~InferenceHelperSnpe()
{
}

int32_t InferenceHelperSnpe::setNumThread(const int32_t numThread)
{
	m_numThread = numThread;
	return RET_OK;
}

int32_t InferenceHelperSnpe::setCustomOps(const std::vector<std::pair<const char*, const void*>>& customOps)
{
	PRINT("[WARNING] This method is not supported\n");
	return RET_OK;
}

int32_t InferenceHelperSnpe::initialize(const std::string& modelFilename, std::vector<InputTensorInfo>& inputTensorInfoList, std::vector<OutputTensorInfo>& outputTensorInfoList)
{
	/* Settings for SNPE */
	int32_t userBufferSourceType = CPUBUFFER;
	int32_t bufferType = USERBUFFER_FLOAT;
	int32_t bitWidth = 0;
	switch (inputTensorInfoList[0].tensorType) {
	case InputTensorInfo::TENSOR_TYPE_FP32:
		bufferType = USERBUFFER_FLOAT;
		break;
	case InputTensorInfo::TENSOR_TYPE_UINT8:
		bufferType = USERBUFFER_TF8;
		bitWidth = 8;
		break;
	default:
		PRINT_E("Unsupported tensor type\n");
		return RET_ERR;
	}
	bool useUserSuppliedBuffers = (bufferType == USERBUFFER_FLOAT || bufferType == USERBUFFER_TF8 || bufferType == USERBUFFER_TF16);

	/* Create network */
	m_snpe = createSnpe(modelFilename, useUserSuppliedBuffers);
	if (!m_snpe) {
		PRINT_E("Failed to create SNPE\n");
		return RET_ERR;
	}

	getAllTensorInfo(m_snpe, inputTensorInfoList, outputTensorInfoList);


	/* Allocate buffer memory for input/output */
	if (useUserSuppliedBuffers) {
		if (bufferType == USERBUFFER_TF8 || bufferType == USERBUFFER_TF16) {
			PRINT_E("Not tested\n");
			createOutputBufferMap(*m_outputMap, m_applicationOutputBuffers, m_snpeUserBackedOutputBuffers, m_snpe, true, bitWidth);
			createInputBufferMap(*m_inputMap, m_applicationInputBuffers, m_snpeUserBackedInputBuffers, m_snpe, true, bitWidth);
		} else if (bufferType == USERBUFFER_FLOAT) {
			createOutputBufferMap(*m_outputMap, m_applicationOutputBuffers, m_snpeUserBackedOutputBuffers, m_snpe, false, bitWidth);
			if (userBufferSourceType == CPUBUFFER) {
				createInputBufferMap(*m_inputMap, m_applicationInputBuffers, m_snpeUserBackedInputBuffers, m_snpe, false, bitWidth);
			} else {
				PRINT_E("Not supported\n");
				return RET_ERR;
			}
		}
	} else {
		PRINT_E("Not supported\n");
		return RET_ERR;
	}

	/* Convert normalize parameter to speed up */
	for (auto& inputTensorInfo : inputTensorInfoList) {
		convertNormalizeParameters(inputTensorInfo);
	}

	return RET_OK;
};


int32_t InferenceHelperSnpe::finalize(void)
{
	// m_snpeWrapper.reset();
	// m_inputBuffer.reset();
	return RET_ERR;
}

int32_t InferenceHelperSnpe::preProcess(const std::vector<InputTensorInfo>& inputTensorInfoList)
{
	if (!m_snpe || !m_inputMap || !m_outputMap) {
		PRINT_E("Interpreter is not built yet\n");
		return RET_ERR;
	}

	for (const auto& inputTensorInfo : inputTensorInfoList) {
		if (inputTensorInfo.dataType == InputTensorInfo::DATA_TYPE_IMAGE) {
			if ((inputTensorInfo.imageInfo.width != inputTensorInfo.imageInfo.cropWidth) || (inputTensorInfo.imageInfo.height != inputTensorInfo.imageInfo.cropHeight)) {
				PRINT_E("Crop is not supported\n");
				return  RET_ERR;
			}
			if ((inputTensorInfo.imageInfo.cropWidth != inputTensorInfo.tensorDims.width) || (inputTensorInfo.imageInfo.cropHeight != inputTensorInfo.tensorDims.height)) {
				PRINT_E("Resize is not supported\n");
				return  RET_ERR;
			}
			if (inputTensorInfo.imageInfo.channel != inputTensorInfo.tensorDims.channel) {
				PRINT_E("Color conversion is not supported\n");
				return  RET_ERR;
			}

			/* Normalize image (NHWC to NHWC)*/
			uint8_t* src = static_cast<uint8_t*>(inputTensorInfo.data);
			if (inputTensorInfo.tensorType == TensorInfo::TENSOR_TYPE_UINT8) {
				PRINT_E("TENSOR_TYPE_UINT8 is not supported\n");
			} else if (inputTensorInfo.tensorType == TensorInfo::TENSOR_TYPE_FP32) {
				float* dst = reinterpret_cast<float*> (&m_applicationInputBuffers.at(inputTensorInfo.name)[0]);
#pragma omp parallel for num_threads(m_numThread)
				for (int32_t i = 0; i < inputTensorInfo.tensorDims.width * inputTensorInfo.tensorDims.height; i++) {
					for (int32_t c = 0; c < inputTensorInfo.tensorDims.channel; c++) {
#if 1
						dst[i * inputTensorInfo.tensorDims.channel + c] = (src[i * inputTensorInfo.tensorDims.channel + c] - inputTensorInfo.normalize.mean[c]) * inputTensorInfo.normalize.norm[c];
#else
						dst[i * inputTensorInfo.tensorDims.channel + c] = (src[i * inputTensorInfo.tensorDims.channel + c] / 255.0f - inputTensorInfo.normalize.mean[c]) / inputTensorInfo.normalize.norm[c];
#endif
					}
				}
			} else {
				PRINT_E("Unsupported tensorType (%d)\n", inputTensorInfo.tensorType);
				return RET_ERR;
			}

		} else if ( (inputTensorInfo.dataType == InputTensorInfo::DATA_TYPE_BLOB_NHWC) || (inputTensorInfo.dataType == InputTensorInfo::DATA_TYPE_BLOB_NCHW) ){
			PRINT_E("DATA_TYPE_BLOB_NHWC (DATA_TYPE_BLOB_NCHW) is not supported\n");
			return RET_ERR;
		} else {
			PRINT_E("Unsupported data type (%d)\n", inputTensorInfo.dataType);
			return RET_ERR;
		}
	}
	return RET_OK;
}

int32_t InferenceHelperSnpe::invoke(std::vector<OutputTensorInfo>& outputTensorInfoList)
{
	if (!m_snpe || !m_inputMap || !m_outputMap) {
		PRINT_E("Interpreter is not built yet\n");
		return RET_ERR;
	}

	bool execStatus = m_snpe->execute(*m_inputMap, *m_outputMap);
	if (execStatus == false) {
		PRINT_E("Error while executing the network.\n");
		return RET_ERR;
	}

	for (auto& outputTensorInfo : outputTensorInfoList) {
		outputTensorInfo.data = m_applicationOutputBuffers.at(outputTensorInfo.name).data();
	}

	return RET_OK;
}

void InferenceHelperSnpe::convertNormalizeParameters(InputTensorInfo& inputTensorInfo)
{
	if (inputTensorInfo.dataType != InputTensorInfo::DATA_TYPE_IMAGE) return;

#if 0
	/* Convert to speeden up normalization:  ((src / 255) - mean) / norm  = src * 1 / (255 * norm) - (mean / norm) */
	for (int32_t i = 0; i < 3; i++) {
		inputTensorInfo.normalize.mean[i] /= inputTensorInfo.normalize.norm[i];
		inputTensorInfo.normalize.norm[i] *= 255.0f;
		inputTensorInfo.normalize.norm[i] = 1.0f / inputTensorInfo.normalize.norm[i];
	}
#endif
#if 1
	/* Convert to speeden up normalization:  ((src / 255) - mean) / norm = (src  - (mean * 255))  * (1 / (255 * norm)) */
	for (int32_t i = 0; i < 3; i++) {
		inputTensorInfo.normalize.mean[i] *= 255.0f;
		inputTensorInfo.normalize.norm[i] *= 255.0f;
		inputTensorInfo.normalize.norm[i] = 1.0f / inputTensorInfo.normalize.norm[i];
	}
#endif
}


static zdl::DlSystem::RuntimeList getSystemAvailability(void)
{
	zdl::DlSystem::Version_t version = zdl::SNPE::SNPEFactory::getLibraryVersion();
	PRINT("SNPE Version: %s\n", version.asString().c_str());

	zdl::DlSystem::RuntimeList runtimeList;
	zdl::DlSystem::Runtime_t runtime = zdl::DlSystem::Runtime_t::DSP_FIXED8_TF;
	if (zdl::SNPE::SNPEFactory::isRuntimeAvailable(runtime) == false) {
		PRINT_E("DSP is not available. Falling back to GPU.\n");
		runtime = zdl::DlSystem::Runtime_t::GPU_FLOAT32_16_HYBRID;
		if (zdl::SNPE::SNPEFactory::isRuntimeAvailable(runtime) == false) {
			PRINT_E("GPU is not availagle. Falling back to CPU.\n");
			runtime = zdl::DlSystem::Runtime_t::CPU_FLOAT32;
		}
	}
	runtimeList.add(runtime);
	return runtimeList;
}


std::unique_ptr<zdl::SNPE::SNPE> InferenceHelperSnpe::createSnpe(const std::string& modelFilename, bool useUserSuppliedBuffers)
{
	zdl::DlSystem::RuntimeList runtimeList = getSystemAvailability();

	std::unique_ptr<zdl::DlContainer::IDlContainer> container = zdl::DlContainer::IDlContainer::open(zdl::DlSystem::String(modelFilename.c_str()));
	if (container == nullptr) {
		PRINT_E("Error while opening the container file.\n");
		return nullptr;
	}

	zdl::DlSystem::UDLFactoryFunc udlFunc = UdlExample::MyUDLFactory;
	zdl::DlSystem::UDLBundle udlBundle; udlBundle.cookie = (void*)0xdeadbeaf, udlBundle.func = udlFunc; // 0xdeadbeaf to test cookie
	zdl::DlSystem::PlatformConfig platformConfig;

	zdl::SNPE::SNPEBuilder snpeBuilder(container.get());
	std::unique_ptr<zdl::SNPE::SNPE> snpe = snpeBuilder.setOutputLayers({})
		.setRuntimeProcessorOrder(runtimeList)
		.setUdlBundle(udlBundle)
		.setUseUserSuppliedBuffers(useUserSuppliedBuffers)
		.setPlatformConfig(platformConfig)
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
	opts.LogFileDirectory = modelFilename + "_log/";
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

int32_t InferenceHelperSnpe::getTensorInfo(std::unique_ptr<zdl::SNPE::SNPE> const& snpe, const std::string& name, int32_t& batch, int32_t& height, int32_t& width, int32_t& channel)
{
	auto bufferAttributesOpt = snpe->getInputOutputBufferAttributes(name.c_str());
	if (!bufferAttributesOpt) {
		PRINT_E("Error obtaining attributes for input tensor. %s\n", name.c_str());
		return RET_ERR;
	}
	zdl::DlSystem::TensorShape tensorShape = bufferAttributesOpt->getDims();

	batch = -1;
	height = -1;
	width = -1;
	channel = -1;

	batch = static_cast<int32_t>(tensorShape.getDimensions()[0]);
	if (tensorShape.rank() > 1) {
		height = static_cast<int32_t>(tensorShape.getDimensions()[1]);
	}
	if (tensorShape.rank() > 2) {
		width = static_cast<int32_t>(tensorShape.getDimensions()[2]);
	}
	if (tensorShape.rank() > 3) {
		channel = static_cast<int32_t>(tensorShape.getDimensions()[3]);
	}
	PRINT("%s: batch = %d, height = %d, width = %d, channel = %d\n", name.c_str(), batch, height, width, channel);

	return RET_OK;
}

int32_t InferenceHelperSnpe::getAllTensorInfo(std::unique_ptr<zdl::SNPE::SNPE> const& snpe, std::vector<InputTensorInfo>& inputTensorInfoList, std::vector<OutputTensorInfo>& outputTensorInfoList)
{
	for (auto& inputTensorInfo : inputTensorInfoList) {
		int32_t batch, height, width, channel;
		if (getTensorInfo(snpe, inputTensorInfo.name, batch, height, width, channel) != 0) {
			return RET_ERR;
		}
		if (batch == -1 && inputTensorInfo.tensorDims.batch == -1) {
			PRINT_E("%s: Batch size is undefined\n", inputTensorInfo.name.c_str());
			return RET_ERR;
		} else if (batch != -1 && inputTensorInfo.tensorDims.batch == -1) {
			inputTensorInfo.tensorDims.batch = batch;
		} else if (batch == -1 && inputTensorInfo.tensorDims.batch != -1) {
			// do nothing
		} else {
			if (batch != inputTensorInfo.tensorDims.batch) {
				PRINT_E("%s: Batch size doesn't match: %d vs %d\n", inputTensorInfo.name.c_str(), batch, inputTensorInfo.tensorDims.batch);
				return RET_ERR;
			}
		}
		if (height == -1 && inputTensorInfo.tensorDims.height == -1) {
			// PRINT_E("%s: Height size is undefined\n", inputTensorInfo.name.c_str());
			// return RET_ERR;
			inputTensorInfo.tensorDims.height = 1;
		} else if (height != -1 && inputTensorInfo.tensorDims.height == -1) {
			inputTensorInfo.tensorDims.height = height;
		} else if (height == -1 && inputTensorInfo.tensorDims.height != -1) {
			// do nothing
		} else {
			if (height != inputTensorInfo.tensorDims.height) {
				PRINT_E("%s: Height size doesn't match: %d vs %d\n", inputTensorInfo.name.c_str(), height, inputTensorInfo.tensorDims.height);
				return RET_ERR;
			}
		}
		if (width == -1 && inputTensorInfo.tensorDims.width == -1) {
			// PRINT_E("%s: Width size is undefined\n", inputTensorInfo.name.c_str());
			// return RET_ERR;
			inputTensorInfo.tensorDims.width = 1;
		} else if (width != -1 && inputTensorInfo.tensorDims.width == -1) {
			inputTensorInfo.tensorDims.width = width;
		} else if (width == -1 && inputTensorInfo.tensorDims.width != -1) {
			// do nothing
		} else {
			if (width != inputTensorInfo.tensorDims.width) {
				PRINT_E("%s: Width size doesn't match: %d vs %d\n", inputTensorInfo.name.c_str(), width, inputTensorInfo.tensorDims.width);
				return RET_ERR;
			}
		}
		if (channel == -1 && inputTensorInfo.tensorDims.channel == -1) {
			// PRINT_E("%s: Channel size is undefined\n", inputTensorInfo.name.c_str());
			// return RET_ERR;
			inputTensorInfo.tensorDims.channel = 1;
		} else if (channel != -1 && inputTensorInfo.tensorDims.channel == -1) {
			inputTensorInfo.tensorDims.channel = channel;
		} else if (channel == -1 && inputTensorInfo.tensorDims.channel != -1) {
			// do nothing
		} else {
			if (channel != inputTensorInfo.tensorDims.channel) {
				PRINT_E("%s: Channel size doesn't match: %d vs %d\n", inputTensorInfo.name.c_str(), channel, inputTensorInfo.tensorDims.channel);
				return RET_ERR;
			}
		}
	}
	
	for (auto& outputTensorInfo : outputTensorInfoList) {
		int32_t batch, height, width, channel;
		getTensorInfo(snpe, outputTensorInfo.name, batch, height, width, channel);

		if (batch == -1 && outputTensorInfo.tensorDims.batch == -1) {
			PRINT_E("%s: Batch size is undefined\n", outputTensorInfo.name.c_str());
			return RET_ERR;
		} else if (batch != -1 && outputTensorInfo.tensorDims.batch == -1) {
			outputTensorInfo.tensorDims.batch = batch;
		} else if (batch == -1 && outputTensorInfo.tensorDims.batch != -1) {
			// do nothing
		} else {
			if (batch != outputTensorInfo.tensorDims.batch) {
				PRINT_E("%s: Batch size doesn't match: %d vs %d\n", outputTensorInfo.name.c_str(), batch, outputTensorInfo.tensorDims.batch);
				return RET_ERR;
			}
		}
		if (height == -1 && outputTensorInfo.tensorDims.height == -1) {
			// PRINT_E("%s: Height size is undefined\n", outputTensorInfo.name.c_str());
			// return RET_ERR;
			outputTensorInfo.tensorDims.height = 1;
		} else if (height != -1 && outputTensorInfo.tensorDims.height == -1) {
			outputTensorInfo.tensorDims.height = height;
		} else if (height == -1 && outputTensorInfo.tensorDims.height != -1) {
			// do nothing
		} else {
			if (height != outputTensorInfo.tensorDims.height) {
				PRINT_E("%s: Height size doesn't match: %d vs %d\n", outputTensorInfo.name.c_str(), height, outputTensorInfo.tensorDims.height);
				return RET_ERR;
			}
		}
		if (width == -1 && outputTensorInfo.tensorDims.width == -1) {
			// PRINT_E("%s: Width size is undefined\n", outputTensorInfo.name.c_str());
			// return RET_ERR;
			outputTensorInfo.tensorDims.width = 1;
		} else if (width != -1 && outputTensorInfo.tensorDims.width == -1) {
			outputTensorInfo.tensorDims.width = width;
		} else if (width == -1 && outputTensorInfo.tensorDims.width != -1) {
			// do nothing
		} else {
			if (width != outputTensorInfo.tensorDims.width) {
				PRINT_E("%s: Width size doesn't match: %d vs %d\n", outputTensorInfo.name.c_str(), width, outputTensorInfo.tensorDims.width);
				return RET_ERR;
			}
		}
		if (channel == -1 && outputTensorInfo.tensorDims.channel == -1) {
			// PRINT_E("%s: Channel size is undefined\n", outputTensorInfo.name.c_str());
			// return RET_ERR;
			outputTensorInfo.tensorDims.channel = 1;
		} else if (channel != -1 && outputTensorInfo.tensorDims.channel == -1) {
			outputTensorInfo.tensorDims.channel = channel;
		} else if (channel == -1 && outputTensorInfo.tensorDims.channel != -1) {
			// do nothing
		} else {
			if (channel != outputTensorInfo.tensorDims.channel) {
				PRINT_E("%s: Channel size doesn't match: %d vs %d\n", outputTensorInfo.name.c_str(), channel, outputTensorInfo.tensorDims.channel);
				return RET_ERR;
			}
		}
	}

	return RET_OK;
}
