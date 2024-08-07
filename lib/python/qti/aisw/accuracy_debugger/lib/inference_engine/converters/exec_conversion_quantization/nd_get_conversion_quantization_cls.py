# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

from qti.aisw.accuracy_debugger.lib.inference_engine.converters.exec_conversion_quantization.nd_exec_qnn_conversion_quantization import ExecuteQNNConversionAndQuantization
from qti.aisw.accuracy_debugger.lib.inference_engine.converters.exec_conversion_quantization.nd_exec_snpe_conversion_quantization import ExecuteSNPEConversion, ExecuteSNPEQuantization, ExecuteSNPEConversionAndQuantization
from qti.aisw.accuracy_debugger.lib.inference_engine.converters.exec_conversion_quantization.nd_exec_qairt_conversion_quantization import ExecuteQAIRTConversion, ExecuteQAIRTQuantization, ExecuteQAIRTConversionAndQuantization
from qti.aisw.accuracy_debugger.lib.utils.nd_constants import Engine


def get_exec_conversion_quantization_cls(engine: str, task: str):
    if engine == Engine.QNN.value:
        return ExecuteQNNConversionAndQuantization
    elif engine == Engine.SNPE.value:
        if task == "conversion":
            return ExecuteSNPEConversion
        elif task == "quantization":
            return ExecuteSNPEQuantization
        elif task == "conversion_quantization":
            return ExecuteSNPEConversionAndQuantization
    elif engine == Engine.QAIRT.value:
        if task == "conversion":
            return ExecuteQAIRTConversion
        elif task == "quantization":
            return ExecuteQAIRTQuantization
        elif task == "conversion_quantization":
            return ExecuteQAIRTConversionAndQuantization
