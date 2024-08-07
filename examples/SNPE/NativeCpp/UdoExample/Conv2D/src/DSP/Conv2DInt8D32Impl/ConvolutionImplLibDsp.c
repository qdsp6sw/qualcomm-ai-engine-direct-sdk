//=============================================================================
//
//  Copyright (c) 2020-2021, 2024 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//=============================================================================

#include <string.h>
#include <float.h>
#include <stdlib.h>
#include <math.h>
#include <algorithm>
#include "DSP/QnnDspOpPackage.h"
#include "DspOps.hpp"

// operations info
char g_convOpType [] = "Conv";
uint32_t g_convStaticParamsNum = 5;
uint32_t g_convInputsNum = 3;
uint32_t g_convOutputsNum = 1;
Udo_QuantizationType_t g_convInputQuantizationTypes [] = {UDO_QUANTIZATION_TF,UDO_QUANTIZATION_TF,UDO_QUANTIZATION_TF};
Udo_QuantizationType_t g_convOutputQuantizationTypes [] =  {UDO_QUANTIZATION_TF};
Udo_HexNNTensorLayout_t* g_convLayout = NULL;

Udo_ErrorType_t
conv_createOpFactory (QnnOpPackage_GlobalInfrastructure_t globalInfra,
    Udo_CoreType_t udoCoreType, void *perFactoryInfrastructure,
    Udo_String_t operationType, uint32_t numOfStaticParams,
    Udo_Param_t *staticParams, Udo_OpFactory_t *opFactory)
{
    if(operationType == NULL || opFactory == NULL) {
        return UDO_INVALID_ARGUMENT;
    }
    if(strcmp(operationType, g_convOpType) == 0) {
        convOpFactory_t* thisFactory = (convOpFactory_t *)(*(globalInfra->dspGlobalInfra->hexNNv2Infra.udoMalloc))(sizeof(convOpFactory_t));
        int size = strlen(operationType) + 1; // +1 to hold the '\0' character
        thisFactory->opType = (Udo_String_t)(*(globalInfra->dspGlobalInfra->hexNNv2Infra.udoMalloc))(size);
        strlcpy((thisFactory->opType), operationType, size);
        thisFactory->numOfStaticParams = numOfStaticParams;
        size = sizeof(Udo_Param_t) * numOfStaticParams;
        thisFactory->staticParams = (Udo_Param_t *)(*(globalInfra->dspGlobalInfra->hexNNv2Infra.udoMalloc))(size);
        auto tempPtr = thisFactory->staticParams;
        for (int i = 0; i < numOfStaticParams; ++i)
        {
            thisFactory->staticParams->paramType = staticParams->paramType;
            size = strlen(staticParams->paramName) + 1; // +1 to hold the '\0' character
            thisFactory->staticParams->paramName = (Udo_String_t)(*(globalInfra->dspGlobalInfra->hexNNv2Infra.udoMalloc))(size);
            strlcpy((thisFactory->staticParams->paramName), staticParams->paramName, size);
            if (staticParams->paramType == UDO_PARAMTYPE_SCALAR)
            {
               thisFactory->staticParams->scalarParam = staticParams->scalarParam;
            }
            else if (staticParams->paramType == UDO_PARAMTYPE_TENSOR)
            {
               size = sizeof(int) * (*staticParams->tensorParam.maxDimensions);
               thisFactory->staticParams->tensorParam.tensorData = (int *)(*(globalInfra->dspGlobalInfra->hexNNv2Infra.udoMalloc))(size);
               memcpy((char *)thisFactory->staticParams->tensorParam.tensorData, (char *)staticParams->tensorParam.tensorData, size);
            }
            ++staticParams;
            ++thisFactory->staticParams;
        }
        thisFactory->staticParams = tempPtr;
        *opFactory = (Udo_OpFactory_t)thisFactory;
    } else {
        return UDO_INVALID_ARGUMENT;
    }
    return UDO_NO_ERROR;
}

Udo_ErrorType_t
conv_releaseOpFactory(QnnOpPackage_GlobalInfrastructure_t globalInfra,
                                              Udo_OpFactory_t opFactory)
{
    if(opFactory == NULL) {
        return UDO_INVALID_ARGUMENT;
    }
    convOpFactory_t* thisFactory = (convOpFactory_t *)(opFactory);
    (*(globalInfra->dspGlobalInfra->hexNNv2Infra.udoFree))((thisFactory->opType));
    auto tempPtr = thisFactory->staticParams;
    for (int i = 0; i < thisFactory->numOfStaticParams; ++i)
    {
         (*(globalInfra->dspGlobalInfra->hexNNv2Infra.udoFree))((thisFactory->staticParams->paramName));
         if (thisFactory->staticParams->paramType == UDO_PARAMTYPE_TENSOR)
         {
            (*(globalInfra->dspGlobalInfra->hexNNv2Infra.udoFree))((thisFactory->staticParams->tensorParam.tensorData));
         }
         ++thisFactory->staticParams;
    }
    thisFactory->staticParams = tempPtr;
    (*(globalInfra->dspGlobalInfra->hexNNv2Infra.udoFree))((thisFactory->staticParams));
    (*(globalInfra->dspGlobalInfra->hexNNv2Infra.udoFree))(thisFactory);
    return UDO_NO_ERROR;
}

Udo_ErrorType_t
conv_validateOperation (Udo_String_t operationType, uint32_t numOfStaticParams,
    const Udo_Param_t *staticParams) {
    if(strcmp(operationType, g_convOpType) == 0) {
        if (numOfStaticParams != g_convStaticParamsNum) {
            return UDO_INVALID_ARGUMENT;
        }
        /*
         * If this op should validate others, add code here
         */
    } else {
        return UDO_INVALID_ARGUMENT;
    }
    return UDO_NO_ERROR;
}

typedef struct{
  uint32_t B;
  uint32_t H;
  uint32_t W;
  uint32_t D;
  uint32_t heightPadBefore;
  uint32_t heightPadAfter;
  uint32_t widthPadBefore;
  uint32_t widthPadAfter;
  uint32_t depthPadBefore;
  uint32_t depthPadAfter;
} D32_Params;

typedef struct ConvOpInfo_t {
  uint8_t *input;
  uint8_t *filter;
  uint8_t *bias;
  uint8_t *output;
  uint32_t inputHeight;
  uint32_t inputWidth;
  uint32_t inputDepth;
  uint32_t filterHeight;
  uint32_t filterWidth;
  uint32_t filterDepth;
  uint32_t outputHeight;
  uint32_t outputWidth;
  uint32_t outputDepth;
  float inputMin;
  float inputMax;
  float filterMin;
  float filterMax;
  float outputMin;
  float outputMax;
  int32_t padH;
  int32_t padW;
  int32_t strideH;
  int32_t strideW;
  int32_t groups;
  D32_Params* in_d32prms;
  D32_Params* out_d32prms;
} ConvOpInfo_t;

static inline int32_t quantize_int(float val, float minval, float maxval) {
  /* We want 0.0 -- 255.0 to resize to 0..255 */
  float range     = fmaxf(1e-18f, maxval - minval);
  float resizeAmt = 255.0f / (range);
  float valueF    = (val - minval) * resizeAmt;
  int32_t value   = roundf(valueF);
  return (-1) * value;
}

void floatToInt(float realMultiplier, int* outputMultiplier, int* outputShift)
{
    if (realMultiplier == 0.) {
        *outputMultiplier = 0;
        *outputShift      = 0;
    } else {
        const float q = ::frexp(realMultiplier, outputShift);
        auto qFixed    = static_cast<int64_t>(::round(q * (1LL << 31)));
        if (qFixed == (1LL << 31)) {
            qFixed /= 2;
            *outputShift += 1;
        }
        if (*outputShift < -31) {
            *outputShift = 0;
            qFixed      = 0;
        }
        if (*outputShift > 30) {
            *outputShift = 30;
            qFixed      = (1LL << 31) - 1;
        }
        *outputMultiplier = static_cast<int32_t>(qFixed);
    }
}

size_t tensor_offset_for_d32(
        int32_t b,
        int32_t h,
        int32_t w,
        int32_t d,
        D32_Params* p)
{
    uint32_t h_before = p->heightPadBefore;
    uint32_t w_before = p->widthPadBefore;
    uint32_t d_before = p->depthPadBefore;
    uint32_t h_total = h_before + p->H + p->heightPadAfter;
    uint32_t w_total = w_before + p->W + p->widthPadAfter;
    uint32_t d_total = d_before + p->D + p->depthPadAfter;
    uint32_t d_pos = d + d_before;

    return b * h_total * w_total * d_total
           + h_total * w_total * (d_pos / 32) * 32
           + (h + h_before) * w_total * 32
           + (w + w_before) * 32
           + d_pos % 32 ;
}

void tensor_prepare_normal(uint8_t* src, uint32_t b, uint32_t h, uint32_t w, uint32_t d, uint8_t* dst, D32_Params* p)
{
    for (int i = 0; i < b; ++i)
    {
        for (int j = 0; j < h; ++j)
        {
            for (int k = 0; k < w; ++k)
            {
                for (int l = 0; l < d; ++l)
                {
                    int inputOffsetD32 = tensor_offset_for_d32(i, j, k, l, p);
                    *dst = src[inputOffsetD32];
                    ++dst;
                }
            }
        }
    }
}

void tensor_prepare_d32(uint8_t* src, uint32_t b, uint32_t h, uint32_t w, uint32_t d, uint8_t* dst, D32_Params* p)
{
    int inputOffset = 0;
    for (int i = 0; i < b; ++i)
    {
        for (int j = 0; j < h; ++j)
        {
            for (int k = 0; k < w; ++k)
            {
                for (int l = 0; l < d; ++l)
                {
                    int inputOffsetD32 = tensor_offset_for_d32(i, j, k, l, p);
                    dst[inputOffsetD32] = src[inputOffset++];
                }
            }
        }
    }
}

void Get_D32_Dimensions(uint32_t* dim, D32_Params* params){

    int b = dim[0];
    int h = dim[1];
    int w = dim[2];
    int d = dim[3];

    params->B = b;
    params->H = h;
    params->W = w;
    params->D = d;

    params->heightPadBefore = 4;
    params->heightPadAfter = 4;

    params->widthPadBefore = 4;
    params->widthPadAfter = (4 - w % 4) % 4;
    params->depthPadBefore = 0;
    params->depthPadAfter = (32 - d % 32) % 32;
}

int inline evalQuantizedMultiplier(int input, int output_offset,
    int quantized_multiplier, int shift)
{
    int unclamped_result = input;
    const int total_shift = 31 - shift;
    const int64_t round = static_cast<int64_t>(1) << (total_shift - 1);
    int64_t result = unclamped_result * static_cast<int64_t>(quantized_multiplier) + round;
    result = result >> total_shift;
    unclamped_result = static_cast<int>(result) - output_offset;
    return unclamped_result;
}

void workerThreadConvQuant(void *perOpInfrastructure, void *userData) {
    ConvOpInfo_t *data = (ConvOpInfo_t *)userData;
    D32_Params* in_d32prms = data->in_d32prms;
    D32_Params* out_d32prms = data->out_d32prms;

    uint8_t *input        = data->input;
    uint8_t* filter       = data->filter;
    uint8_t* bias         = data->bias;
    uint8_t *output       = data->output;
    float inputMin        = data->inputMin;
    float inputMax        = data->inputMax;
    float filterMin       = data->filterMin;
    float filterMax       = data->filterMax;
    float outputMin       = data->outputMin;
    float outputMax       = data->outputMax;
    int32_t numFilters = data->outputDepth;
    int32_t padH = data->padH;
    int32_t padW = data->padW;
    int32_t strideH = data->strideH;
    int32_t strideW = data->strideW;
    int32_t groups = data->groups;

    size_t inputHeight = data->inputHeight;
    size_t inputWidth = data->inputWidth;
    size_t filterHeight = data->filterHeight;
    size_t filterWidth = data->filterWidth;
    size_t filterDepth = data->filterDepth;
    size_t outputHeight = data->outputHeight;
    size_t outputWidth = data->outputWidth;
    size_t outputDepth = data->outputDepth;
    size_t outputGroupDepth = numFilters / groups;
    float inputDelta = (inputMax - inputMin) / 255;
    int32_t inputOffset = quantize_int(0.0f, inputMin, inputMax);
    float filterDelta = (filterMax - filterMin) / 255;
    int32_t filterOffset = quantize_int(0.0f, filterMin, filterMax);
    float outputDelta = (outputMax - outputMin) / 255;
    int32_t outputOffset = quantize_int(0.0f, outputMin, outputMax);
    float realMultiplier = 0.0;
    if (outputDelta)
    {
        realMultiplier = (inputDelta * filterDelta) / outputDelta;
    }
    int output_multiplier=0;
    int shift=0;
    floatToInt(realMultiplier, &output_multiplier, &shift);

    for (int32_t oh = 0; oh < outputHeight; oh++) {
        for(int32_t ow = 0; ow < outputWidth; ow++) {
            for (int32_t g = 0; g < groups; g++) {
                for (int32_t d = 0; d < outputGroupDepth; d++) {
                    int32_t offset = g * outputGroupDepth + d;
                    int32_t sum = 0;
                    for (int32_t fh = 0; fh < filterHeight; fh++) {
                        int32_t inputH = oh * strideH - padH + fh;
                        if (inputH < 0) {
                            continue;
                        }
                        if (inputH >= inputHeight) {
                            break;
                        }

                        for (int32_t fw = 0; fw < filterWidth; fw++) {
                            int32_t inputW = ow * strideW - padW + fw;
                            if (inputW < 0) {
                                continue;
                            }
                            if (inputW >= inputWidth) {
                                break;
                            }
                            for (int32_t fd = 0; fd < filterDepth; fd++) {
                                int32_t inOffset = tensor_offset_for_d32(0, inputH, inputW, filterDepth * g + fd, in_d32prms);
                                int32_t fOffset = (fh * filterWidth + fw) * filterDepth * outputDepth + fd * outputDepth;
                                int32_t inputVal = input[inOffset] + inputOffset;
                                int32_t filterVal = filter[fOffset + offset] + filterOffset;
                                sum += inputVal * filterVal;
                            }//fd
                        }//fw
                    }// end of loop fh
                    if (bias) {
                        int32_t biasVal = bias[offset];
                        sum += biasVal;
                    }
                    uint32_t outputIndex = tensor_offset_for_d32(0, oh, ow, g * outputGroupDepth + d, out_d32prms);
                    auto val = evalQuantizedMultiplier(sum, outputOffset, output_multiplier, shift);
                    output[outputIndex] = (val < 0) ? 0 : ((val > 255) ? 255 : val);
                }// d
            }//g
         }// end of loop ow
    }// end of loop oh
}

Udo_ErrorType_t
conv_executeOp (QnnOpPackage_GlobalInfrastructure_t globalInfra,
    Udo_Operation_t operation, bool blocking, const uint32_t ID,
    Udo_ExternalNotify_t notifyFunc) {
    if (operation == NULL) {
        return UDO_INVALID_ARGUMENT;
    }
    OpParams_t* m_Operation = (OpParams_t*) operation;
    const char* opType = ((convOpFactory_t*)(m_Operation->opFactory))->opType;
    if (opType == NULL) {
        return UDO_INVALID_ARGUMENT;
    }
    if (strcmp(opType, g_convOpType) == 0) {
        Udo_TensorParam_t* in_tensor = &(m_Operation->InputParams[0]);
        Udo_TensorParam_t* out_tensor = &(m_Operation->outputParams[0]);
        Udo_TensorParam_t* filt_tensor = &(m_Operation->InputParams[1]);
        Udo_TensorParam_t* bias_tensor = &(m_Operation->InputParams[2]);

        if (in_tensor->layout == UDO_LAYOUT_NULL || out_tensor->layout == UDO_LAYOUT_NULL) {
            return UDO_UNSUPPORTED_FEATURE;
        }

       D32_Params in_d32prms;
       Get_D32_Dimensions(in_tensor->currDimensions, &in_d32prms);
        float inputMin = in_tensor->quantizeParams.TFParams.minValue;
        float inputMax = in_tensor->quantizeParams.TFParams.maxValue;
        float filterMin = filt_tensor->quantizeParams.TFParams.minValue;
        float filterMax = filt_tensor->quantizeParams.TFParams.maxValue;
        float outputMin = out_tensor->quantizeParams.TFParams.minValue;
        float outputMax = out_tensor->quantizeParams.TFParams.maxValue;

        uint8_t *inputTensorData  = (uint8_t *)(in_tensor->tensorData);
        uint8_t *filterTensorData = (uint8_t *)(filt_tensor->tensorData);
        uint8_t *biasTensorData   = nullptr;
        uint8_t *outputTensorData = (uint8_t *)(out_tensor->tensorData);

        uint32_t inputHeight = in_tensor->currDimensions[1];
        uint32_t inputWidth = in_tensor->currDimensions[2];
        uint32_t inputDepth = in_tensor->currDimensions[3];
        uint32_t filterHeight = filt_tensor->currDimensions[0];
        uint32_t filterWidth = filt_tensor->currDimensions[1];
        uint32_t filterDepth = filt_tensor->currDimensions[2];
        uint32_t outputHeight = out_tensor->currDimensions[1];
        uint32_t outputWidth = out_tensor->currDimensions[2];
        uint32_t outputDepth = out_tensor->currDimensions[3];
        int padH = 0;
        int padW = 0;
        int strideH = 0;
        int strideW = 0;
        int groups = 1;

        uint32_t numOfStaticParams = ((convOpFactory_t*)(m_Operation->opFactory))->numOfStaticParams;
        Udo_Param_t* staticParams  = ((convOpFactory_t*)(m_Operation->opFactory))->staticParams;
        for (int i = 0; i < numOfStaticParams; i++) {
            Udo_Param_t* param = staticParams;
            if (param->paramType == UDO_PARAMTYPE_SCALAR)
            {
                if (strcmp(param->paramName, "group") == 0)
                {
                    groups = param->scalarParam.dataValue.int32Value;
                }
            }
            else if (param->paramType == UDO_PARAMTYPE_TENSOR)
            {
                if (strcmp(param->paramName, "strides") == 0)
                {
                    auto strides = (int32_t*)param->tensorParam.tensorData;
                    strideH = strides[0];
                    strideW = strides[1];
                }
                else if (strcmp(param->paramName, "pads") == 0)
                {
                    auto pads = (int32_t*)param->tensorParam.tensorData;
                    padH = pads[0];
                    padW = pads[1];
                }
            }
            ++staticParams;
        }
        if (bias_tensor) {
            biasTensorData = (uint8_t *)(bias_tensor->tensorData);
        }

        // set output height, width and depth to infra
        D32_Params out_d32prms;
        Get_D32_Dimensions(out_tensor->currDimensions, &out_d32prms);
        out_tensor->dataType = UDO_DATATYPE_FIXED_8;
        int size = (out_d32prms.H + out_d32prms.heightPadBefore + out_d32prms.heightPadAfter) *
                    (out_d32prms.W + out_d32prms.widthPadBefore + out_d32prms.widthPadAfter) *
        (out_d32prms.D + out_d32prms.depthPadBefore + out_d32prms.depthPadAfter);

        if ((*(globalInfra->dspGlobalInfra->hexNNv2Infra.udoSetOutputTensorSize))(
              m_Operation->opInfra, 0, outputHeight * outputWidth * outputDepth) != 0) {
            return UDO_UNSUPPORTED_FEATURE;
        }
        //Create input tensors in d32 format
        int d32Size = (in_d32prms.H + in_d32prms.heightPadBefore + in_d32prms.heightPadAfter) *
                      (in_d32prms.W + in_d32prms.widthPadBefore + in_d32prms.widthPadAfter) *
        (in_d32prms.D + in_d32prms.depthPadBefore + in_d32prms.depthPadAfter);
        uint8_t* inputTensorDataD32 = (uint8_t*)malloc(d32Size);
        memset(inputTensorDataD32, 0, d32Size);
        tensor_prepare_d32(inputTensorData, 1, in_d32prms.H, in_d32prms.W, in_d32prms.D, inputTensorDataD32, &in_d32prms);
        uint8_t* outputTensorDataD32 = (uint8_t*)malloc(size);
        memset(outputTensorDataD32, 0, size);

        ConvOpInfo_t workerThreadIn = {
           inputTensorDataD32, filterTensorData, biasTensorData, outputTensorDataD32,
           inputHeight, inputWidth, inputDepth, filterHeight, filterWidth,
           filterDepth, outputHeight, outputWidth, outputDepth, inputMin,
           inputMax, filterMin, filterMax, outputMin, outputMax, padH, padW,
           strideH, strideW, groups, &in_d32prms, &out_d32prms};
          (*(globalInfra->dspGlobalInfra->hexNNv2Infra.udoRunWorkerThreads))(
          m_Operation->opInfra, 1, workerThreadConvQuant, &workerThreadIn);
        tensor_prepare_normal(outputTensorDataD32, 1, outputHeight, outputWidth, outputDepth, outputTensorData, &out_d32prms);
        free(inputTensorDataD32);
        free(outputTensorDataD32);
        return UDO_NO_ERROR;
    } else {
        return UDO_INVALID_ARGUMENT;
    }
}

Udo_ErrorType_t conv_queryOperation (
    Udo_String_t operationType, uint32_t numOfStaticParams,
    const Udo_Param_t *staticParams, uint32_t *numOfInputs,
    Udo_QuantizationType_t **inputsQuantTypes,
    Udo_HexNNTensorLayout_t **inputsLayouts, uint32_t *numOfOutputs,
    Udo_QuantizationType_t **outputsQuantTypes,
    Udo_HexNNTensorLayout_t **outputsLayouts) {
    if (strcmp(operationType, g_convOpType) == 0) {
        *numOfInputs = g_convInputsNum;
        *inputsQuantTypes = g_convInputQuantizationTypes;
        *inputsLayouts = g_convLayout;
        *numOfOutputs = g_convOutputsNum;
        *outputsQuantTypes = g_convOutputQuantizationTypes;
        *outputsLayouts = g_convLayout;
    } else {
        return UDO_WRONG_OPERATION;
    }
    return UDO_NO_ERROR;
}

UdoDspShared* new_conv(QnnOpPackage_GlobalInfrastructure_t globalInfra) {
    UdoDspShared* pOpObj = (UdoDspShared*)(*(globalInfra->dspGlobalInfra->hexNNv2Infra.udoMalloc))(sizeof(UdoDspShared));
    if (pOpObj == NULL) {
        return NULL;
    }
    pOpObj->opType = g_convOpType;
    pOpObj->numOfStaticParams = g_convStaticParamsNum;
    pOpObj->numOfInputs = g_convInputsNum;
    pOpObj->numOfOutputs = g_convOutputsNum;

    pOpObj->createOpFactory = conv_createOpFactory;
    pOpObj->releaseOpFactory = conv_releaseOpFactory;
    pOpObj->validateOp = conv_validateOperation;
    pOpObj->executeOp = conv_executeOp;
    pOpObj->queryOp = conv_queryOperation;
    return pOpObj;
}

Udo_ErrorType_t free_conv(QnnOpPackage_GlobalInfrastructure_t globalInfra, UdoDspShared* opInfo) {
    if (opInfo == NULL) {
        return UDO_NO_ERROR;
    }
    (*(globalInfra->dspGlobalInfra->hexNNv2Infra.udoFree))(opInfo);
    return UDO_NO_ERROR;
}
