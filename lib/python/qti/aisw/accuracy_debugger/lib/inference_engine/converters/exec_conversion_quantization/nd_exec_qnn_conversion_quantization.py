# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import os
import subprocess
from qti.aisw.accuracy_debugger.lib.utils.nd_exceptions import InferenceEngineError
from qti.aisw.accuracy_debugger.lib.utils.nd_errors import get_message, get_progress_message
from qti.aisw.accuracy_debugger.lib.inference_engine.converters.exec_conversion_quantization.nd_exec_conversion_quantization import ExecuteConversionAndQuantization
from qti.aisw.accuracy_debugger.lib.utils.nd_constants import ComponentType


class ExecuteQNNConversionAndQuantization(ExecuteConversionAndQuantization):

    def __init__(self, framework: str, host_device: str, params: dict, output_directory: str,
                 logger=None, verbose: str = "info", engine_path: str = None) -> None:
        '''
        Performs quantization of the given model with qnn-{onnx/tflite/tf/pytorch}-converter
        :param framework: One of {onnx/tflite/tf/pytorch}
        :param host_device: Device on which quantization will be performed. one of ["x86_64-windows-msvc", "wos"]
        :param quantization_params: A dictionary with following keys:
            {
                "model_path", "input_tensors", "output_tensors", "input_list_txt"
                "quantization_overrides", "param_quantizer", "act_quantizer"
                "weight_bw", "bias_bw", "act_bw", "float_bias_bw", "restrict_quantization_steps"
                "algorithms", "ignore_encodings", "per_channel_quantization"
                "act_quantizer_calibration", "param_quantizer_calibration"
                "act_quantizer_schema", "param_quantizer_schema"
                "percentile_calibration_value", "extra_converter_args",
                "float_fallback"
            }
        :param output_directory: directory where quantized files will be dumped
        :param logger: logging handle
        :param verbose: logging level
        :param engine_path: QNN SDK path
        '''
        super().__init__(engine='QNN', framework=framework, host_device=host_device, params=params,
                         output_directory=output_directory, engine_path=engine_path, logger=logger,
                         verbose=verbose)
        self._get_coverter(self._config_data[ComponentType.converter.value][self._framework.value])

    def convert_and_quantize(self, quantized_model_save_dir, quantized_model_name):
        '''
        Given a quant Scheme generates the quantized model
        '''
        # Path for qnn model.cpp
        qnn_model_cpp_path = os.path.join(quantized_model_save_dir, quantized_model_name + '.cpp')

        # Quant scheme specific convert command
        convert_command = self._converter.build_convert_command(
            model_path=self._params["model_path"], input_tensors=self._params["input_tensors"],
            output_tensors=self._params["output_tensors"], output_path=qnn_model_cpp_path,
            input_list_txt=self._params["input_list_txt"],
            quantization_overrides=self._params["quantization_overrides"],
            param_quantizer=self._params["param_quantizer"],
            act_quantizer=self._params["act_quantizer"], weight_bw=self._params["weight_bw"],
            bias_bw=self._params["bias_bw"], act_bw=self._params["act_bw"],
            float_bias_bw=self._params["float_bias_bw"],
            restrict_quantization_steps=self._params["restrict_quantization_steps"],
            algorithms=self._params["algorithms"],
            ignore_encodings=self._params["ignore_encodings"],
            per_channel_quantization=self._params["per_channel_quantization"],
            act_quantizer_calibration=self._params["act_quantizer_calibration"],
            param_quantizer_calibration=self._params["param_quantizer_calibration"],
            act_quantizer_schema=self._params["act_quantizer_schema"],
            param_quantizer_schema=self._params["param_quantizer_schema"],
            percentile_calibration_value=self._params["percentile_calibration_value"],
            extra_converter_args=self._params["extra_converter_args"],
            float_fallback=self._params["float_fallback"])
        try:
            self._logger.debug('Model converter command : {}'.format(convert_command))
            code, _, err = self._environment.host_device_obj.execute(
                commands=[convert_command], cwd=self._environment.engine_path,
                env=self._environment.host_env)
            if code != 0:
                raise InferenceEngineError(
                    get_message('ERROR_INFERENCE_ENGINE_BASE_CONVERSION_FAILED'))
            self._logger.info(get_progress_message("PROGRESS_INFERENCE_ENGINE_CONVERSION_FINISHED"))
        except subprocess.CalledProcessError as exc:
            self._logger.error(str(exc))
            raise InferenceEngineError(get_message('ERROR_INFERENCE_ENGINE_CONVERSION_FAILED'))
