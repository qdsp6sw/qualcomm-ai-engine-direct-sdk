//=============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//=============================================================================

#include "QnnModel.hpp"
#include "QnnOpDef.h"

// Flag to determine if Backend should do node validation for each opNode added
#define DO_GRAPH_NODE_VALIDATIONS 1

#ifdef _MSC_VER
#define MODEL_LIB_EXPORT __declspec(dllexport)
#else
#define MODEL_LIB_EXPORT __attribute__((visibility("default")))
#endif

using namespace qnn_wrapper_api;
extern "C" {
MODEL_LIB_EXPORT ModelError_t QnnModel_GenAI_composeGraphs(Qnn_BackendHandle_t backendHandle,
                                          QNN_INTERFACE_VER_TYPE interface,
                                          Qnn_ContextHandle_t contextHandle,
                                          const GraphConfigInfo_t** graphsConfigInfo,
                                          const uint32_t numGraphsConfigInfo,
                                          uint32_t* inputDim,
                                          uint32_t inputRank,
                                          uint32_t* outputDim,
                                          uint32_t outputRank,
                                          uint32_t* kvDim,
                                          uint32_t kvRank,
                                          Qnn_Param_t* params,
                                          uint32_t numParams,
                                          GraphInfoPtr_t** graphsInfo,
                                          uint32_t* numGraphsInfo,
                                          bool debug,
                                          QnnLog_Callback_t logCallback,
                                          QnnLog_Level_t maxLogLevel) {
  (void) logCallback;
  (void) maxLogLevel;
  ModelError_t err = MODEL_NO_ERROR;

  /* model/graph for qnn_model*/
  QnnModel qnn_model;
  const QnnGraph_Config_t** graphConfigs = nullptr;
  VALIDATE(
      getQnnGraphConfigFromInfo("qnn_model", graphsConfigInfo, numGraphsConfigInfo, graphConfigs),
      err);
  VALIDATE(qnn_model.initialize(backendHandle,
                                interface,
                                contextHandle,
                                "qnn_model",
                                debug,
                                DO_GRAPH_NODE_VALIDATIONS,
                                graphConfigs),
           err);
  Qnn_Tensor_t tin;
  tin.version = QNN_TENSOR_VERSION_1;
  tin.v1.id = 0;
  tin.v1.name = "x0";
  tin.v1.type = QNN_TENSOR_TYPE_APP_WRITE;
  tin.v1.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
  tin.v1.dataType = QNN_DATATYPE_UINT_32;
  tin.v1.quantizeParams.encodingDefinition = QNN_DEFINITION_UNDEFINED;
  tin.v1.quantizeParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_UNDEFINED;
  tin.v1.quantizeParams.scaleOffsetEncoding = {.scale = 0.0000000000000000f,
                                               .offset = 0};
  tin.v1.rank = inputRank;
  tin.v1.dimensions = inputDim;
  tin.v1.memType = QNN_TENSORMEMTYPE_RAW;
  tin.v1.clientBuf = {.data = nullptr, .dataSize = 0};
  VALIDATE(qnn_model.addTensor(
               "x0",  // Node Name
               (Qnn_Tensor_t)tin),
           err);

  uint32_t input1Dim[1] = {1};
  Qnn_Tensor_t tin2;
  tin2.version = QNN_TENSOR_VERSION_1;
  tin2.v1.id = 0;
  tin2.v1.name = "x1";
  tin2.v1.type = QNN_TENSOR_TYPE_APP_WRITE;
  tin2.v1.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
  tin2.v1.dataType = QNN_DATATYPE_UINT_32;
  tin2.v1.quantizeParams.encodingDefinition = QNN_DEFINITION_UNDEFINED;
  tin2.v1.quantizeParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_UNDEFINED;
  tin2.v1.quantizeParams.scaleOffsetEncoding = {.scale = 0.0000000000000000f,
                                               .offset = 0};
  tin2.v1.rank = 1;
  tin2.v1.dimensions = input1Dim;
  tin2.v1.memType = QNN_TENSORMEMTYPE_RAW;
  tin2.v1.clientBuf = {.data = nullptr, .dataSize = 0};
  VALIDATE(qnn_model.addTensor(
    "x1",  // Node Name
    (Qnn_Tensor_t)tin2),
    err);

  uint32_t input2Dim[1] = {1};
  Qnn_Tensor_t tin3;
  tin3.version = QNN_TENSOR_VERSION_1;
  tin3.v1.id = 0;
  tin3.v1.name = "x2";
  tin3.v1.type = QNN_TENSOR_TYPE_APP_WRITE;
  tin3.v1.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
  tin3.v1.dataType = QNN_DATATYPE_UINT_32;
  tin3.v1.quantizeParams.encodingDefinition = QNN_DEFINITION_UNDEFINED;
  tin3.v1.quantizeParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_UNDEFINED;
  tin3.v1.quantizeParams.scaleOffsetEncoding = {.scale = 0.0000000000000000f,
                                               .offset = 0};
  tin3.v1.rank = 1;
  tin3.v1.dimensions = input2Dim;
  tin3.v1.memType = QNN_TENSORMEMTYPE_RAW;
  tin3.v1.clientBuf = {.data = nullptr, .dataSize = 0};
  VALIDATE(qnn_model.addTensor(
    "x2",  // Node Name
    (Qnn_Tensor_t)tin3),
           err);

  Qnn_Tensor_t tin4;
  tin4.version = QNN_TENSOR_VERSION_1;
  tin4.v1.id = 0;
  tin4.v1.name = "x3";
  tin4.v1.type = QNN_TENSOR_TYPE_APP_WRITE;
  tin4.v1.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
  tin4.v1.dataType = QNN_DATATYPE_UINT_32;
  tin4.v1.quantizeParams.encodingDefinition = QNN_DEFINITION_UNDEFINED;
  tin4.v1.quantizeParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_UNDEFINED;
  tin4.v1.quantizeParams.scaleOffsetEncoding = {.scale = 0.0000000000000000f,
          .offset = 0};
  tin4.v1.rank = kvRank;
  tin4.v1.dimensions = kvDim;
  tin4.v1.memType = QNN_TENSORMEMTYPE_RAW;
  tin4.v1.clientBuf = {.data = nullptr, .dataSize = 0};
  VALIDATE(qnn_model.addTensor(
                   "x3",  // Node Name
                   (Qnn_Tensor_t)tin4),
           err);

  Qnn_Tensor_t tin5;
  tin5.version = QNN_TENSOR_VERSION_1;
  tin5.v1.id = 0;
  tin5.v1.name = "x4";
  tin5.v1.type = QNN_TENSOR_TYPE_APP_WRITE;
  tin5.v1.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
  tin5.v1.dataType = QNN_DATATYPE_UINT_32;
  tin5.v1.quantizeParams.encodingDefinition = QNN_DEFINITION_UNDEFINED;
  tin5.v1.quantizeParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_UNDEFINED;
  tin5.v1.quantizeParams.scaleOffsetEncoding = {.scale = 0.0000000000000000f,
          .offset = 0};
  tin5.v1.rank = kvRank;
  tin5.v1.dimensions = kvDim;
  tin5.v1.memType = QNN_TENSORMEMTYPE_RAW;
  tin5.v1.clientBuf = {.data = nullptr, .dataSize = 0};
  VALIDATE(qnn_model.addTensor(
                   "x4",  // Node Name
                   (Qnn_Tensor_t)tin5),
           err);

  uint32_t input5Dim[1] = {1};
  Qnn_Tensor_t tin6;
  tin6.version = QNN_TENSOR_VERSION_1;
  tin6.v1.id = 0;
  tin6.v1.name = "x5";
  tin6.v1.type = QNN_TENSOR_TYPE_APP_WRITE;
  tin6.v1.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
  tin6.v1.dataType = QNN_DATATYPE_UINT_32;
  tin6.v1.quantizeParams.encodingDefinition = QNN_DEFINITION_UNDEFINED;
  tin6.v1.quantizeParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_UNDEFINED;
  tin6.v1.quantizeParams.scaleOffsetEncoding = {.scale = 0.0000000000000000f,
          .offset = 0};
  tin6.v1.rank = 1;
  tin6.v1.dimensions = input5Dim;
  tin6.v1.memType = QNN_TENSORMEMTYPE_RAW;
  tin6.v1.clientBuf = {.data = nullptr, .dataSize = 0};
  VALIDATE(qnn_model.addTensor(
                   "x5",  // Node Name
                   (Qnn_Tensor_t)tin6),
           err);

  /* ADDING NODE FOR genAI */
  const char* inputs_genAI[] = {"x0", "x1", "x2", "x3", "x4", "x5"};

  Qnn_Tensor_t tout;
  tout.version = QNN_TENSOR_VERSION_1;
  tout.v1.id = 0;
  tout.v1.name = "output_genAI";
  tout.v1.type = QNN_TENSOR_TYPE_APP_READ;
  tout.v1.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
  tout.v1.dataType = QNN_DATATYPE_FLOAT_32;
  tout.v1.quantizeParams.encodingDefinition = QNN_DEFINITION_UNDEFINED;
  tout.v1.quantizeParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_UNDEFINED;
  tout.v1.quantizeParams.scaleOffsetEncoding = {.scale = 0.0000000000000000f,
                                                .offset = 0};
  tout.v1.rank = outputRank;
  tout.v1.dimensions = outputDim;
  tout.v1.memType = QNN_TENSORMEMTYPE_RAW;
  tout.v1.clientBuf = {.data = nullptr, .dataSize = 0};

  uint32_t output1Dim[1] = {1};
  Qnn_Tensor_t tout1;
  tout1.version = QNN_TENSOR_VERSION_1;
  tout1.v1.id = 0;
  tout1.v1.name = "output_npast";
  tout1.v1.type = QNN_TENSOR_TYPE_APP_READ;
  tout1.v1.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
  tout1.v1.dataType = QNN_DATATYPE_UINT_32;
  tout1.v1.quantizeParams.encodingDefinition = QNN_DEFINITION_UNDEFINED;
  tout1.v1.quantizeParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_UNDEFINED;
  tout1.v1.quantizeParams.scaleOffsetEncoding = {.scale = 0.0000000000000000f,
                                                .offset = 0};
  tout1.v1.rank = 1;
  tout1.v1.dimensions = output1Dim;
  tout1.v1.memType = QNN_TENSORMEMTYPE_RAW;
  tout1.v1.clientBuf = {.data = nullptr, .dataSize = 0};

  Qnn_Tensor_t outputs_genAI[] = {(Qnn_Tensor_t)tout, (Qnn_Tensor_t)tout1};

  VALIDATE(qnn_model.addNode(QNN_OPCONFIG_VERSION_1,  // Op_Config_t Version
                             "LLM",                   // Node Name
                             "llm_engine.oppackage",  // Package Name
                             "LLM",                   // Qnn Node Type
                             params,                  // Node Params
                             numParams,               // Num Node Params
                             inputs_genAI,            // Input Tensor Names
                             6,                       // Num Input Tensor Names
                             outputs_genAI,           // Output Tensors
                             2                        // Num Output Tensors
                             ),
           err);

  // Add all models to array to get graphsInfo
  QnnModel* models[] = {&qnn_model};
  uint32_t numModels = 1;

  // Populate the constructed graphs in provided output variables
  VALIDATE(getGraphInfoFromModels(*models, numModels, graphsInfo), err);
  *numGraphsInfo = numModels;

  return err;

}  // PREPARE_GRAPHS

MODEL_LIB_EXPORT ModelError_t QnnModel_freeGraphsInfo(GraphInfoPtr_t** graphsInfo, uint32_t numGraphsInfo) {
  return qnn_wrapper_api::freeGraphsInfo(graphsInfo, numGraphsInfo);
}  // FREEGRAPHINFO
}
