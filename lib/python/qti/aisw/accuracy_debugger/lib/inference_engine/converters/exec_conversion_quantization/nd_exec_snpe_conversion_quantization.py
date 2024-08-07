# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

from qti.aisw.accuracy_debugger.lib.utils.nd_exceptions import InferenceEngineError
from qti.aisw.accuracy_debugger.lib.utils.nd_errors import get_message
from qti.aisw.accuracy_debugger.lib.inference_engine.converters.exec_conversion_quantization.nd_exec_conversion_quantization import ExecuteConversionAndQuantization
from qti.aisw.accuracy_debugger.lib.utils.nd_constants import ComponentType

import os
import sys
import json
import subprocess
import zipfile
import copy
import importlib


class ExecuteSNPEConversion(ExecuteConversionAndQuantization):

    def __init__(self, framework: str, host_device: str, params: dict, output_directory: str,
                 logger=None, verbose: str = "info", engine_path: str = None):
        '''
        Performs quantization of the given model with snpe-{onnx/tflite/tf}-to-dlc
        :param framework: One of {onnx/tflite/tf}
        :param host_device: Device on which quantization will be performed. one of ["x86_64-windows-msvc", "wos"]
        :param quantization_params: A dictionary with following keys:
            {
                "model_path", "output_tensors",
                "quantization_overrides", "input_tensors",
                "extra_converter_args"
            }
        :param output_directory: directory where quantized files will be dumped
        :param logger: logging handle
        :param verbose: logging level
        :param engine_path: SNPE SDK path
        '''
        super().__init__(engine='SNPE', framework=framework, host_device=host_device,
                         output_directory=output_directory, params=params, engine_path=engine_path,
                         logger=logger, verbose=verbose)
        # set engine path if none specified
        self._get_coverter(self._config_data[ComponentType.converter.value][self._framework.value])

    def convert(self, model_dlc_save_dir, model_dlc_name):
        """Convert a model into a dlc.

        :param dlc_path: Path to save the new dlc
        :param inputs: Input names and dimensions to the model
        :param outputs: Output names of the model
        """
        snpe_base_dlc_path = os.path.join(model_dlc_save_dir, model_dlc_name + '.dlc')
        conversion_inputs = {
            input_name: dim
            for input_name, dim, _ in self._params["input_tensors"]
        }
        convert_command = self._converter.build_convert_command(
            self._params["model_path"], conversion_inputs, self._params["output_tensors"],
            snpe_base_dlc_path, self._params["quantization_overrides"],
            self._params["extra_converter_args"])
        log_string = 'Starting conversion with: ' + \
                     'Inputs: ' + str(list(conversion_inputs)) + ' ' + \
                     'Outputs: ' + str(self._params["output_tensors"])
        self._logger.info(log_string)

        try:
            code, _, err = self._environment.host_device_obj.execute(commands=[convert_command],
                                                                     env=self._environment.host_env)
            if code != 0:
                raise InferenceEngineError(
                    get_message('ERROR_INFERENCE_ENGINE_BASE_CONVERSION_FAILED'))

            self._logger.info('Model converted successfully')
        except subprocess.CalledProcessError as exc:
            self._logger.error(str(exc))
            raise InferenceEngineError(get_message('ERROR_INFERENCE_ENGINE_CONVERSION_FAILED'))
        return code


class ExecuteSNPEQuantization(ExecuteConversionAndQuantization):

    def __init__(self, framework: str, host_device: str, params: dict, output_directory: str,
                 logger=None, verbose: str = "info", engine_path: str = None):
        '''
        Performs quantization of the given model with snpe-{onnx/tflite/tf}-converter/quantizer
        :param framework: One of {onnx/tflite/tf}
        :param host_device: Device on which quantization will be performed. one of ["x86_64-windows-msvc", "wos"]
        :param quantization_params: A dictionary with following keys:
            {
                "input_dlc_path", "input_list", "quantizer",
                "weight_bw", "bias_bw", "act_bw", "float_bw", "htp_socs",
                "extra_quantizer_args", "override_params", "no_weight_quantization"
            }
        :param output_directory: directory where quantized files will be dumped
        :param logger: logging handle
        :param verbose: logging level
        :param engine_path: SNPE SDK path
        '''
        super().__init__(engine='SNPE', framework=framework, host_device=host_device, params=params,
                         output_directory=output_directory, engine_path=engine_path, logger=logger,
                         verbose=verbose)
        # set engine path if none specified

        self._snpe_quantizer_config = self._config_data['snpe_quantizer']
        if (self._host_device in ['x86_64-windows-msvc', 'wos']):
            self._executable = self._snpe_quantizer_config["windows_executable"]
        else:
            self._executable = self._snpe_quantizer_config["executable"]

        arguments = self._snpe_quantizer_config["arguments"]
        self._no_weight_quantization_flag = arguments["no_weight_quantization"]
        self._input_list_flag = arguments["input_list"]
        self._weights_bitwidth_flag = arguments["weights_bitwidth"]
        self._act_bitwidth_flag = arguments["act_bitwidth"]
        self._bias_bitwidth_flag = arguments["bias_bitwidth"]
        self._float_bitwidth_flag = arguments["float_bitwidth"]
        self._use_symmetric_quantize_weights_flag = arguments["use_symmetric_quantize_weights"]
        self._use_adjusted_weights_quantizer_flag = arguments["use_adjusted_weights_quantizer"]
        self._use_enhanced_quantizer_flag = arguments["use_enhanced_quantizer"]
        self._override_params_flag = arguments["override_params"]
        self._enable_htp_flag = arguments["enable_htp"]
        self._htp_socs_flag = arguments["htp_socs"]
        self._dlc_path_flag = arguments["dlc_path"]
        self._output_path_flag = arguments["output_path"]
        self._flags = arguments["flags"].copy()

    def quantize(self, quantized_model_save_dir, quantized_model_name):
        """Execute DLC quantization.

        :param quantized_dlc_path: Path to the converter dlc result
        :param quantization_variation: either of tf, enhanced, adjusted, symmetric
        :param cle: If not None, perform cross layer equalization
        """
        snpe_quantized_dlc_path = os.path.join(quantized_model_save_dir,
                                               quantized_model_name + '.dlc')
        try:
            convert_command = [
                self._executable, self._dlc_path_flag, self._params["input_dlc_path"]
            ]
            if self._params["input_list"]:
                convert_command += [self._input_list_flag, self._params["input_list"]]
            else:
                raise InferenceEngineError(
                    "snpe dlc quantization should be input the inputlist, but you miss it!")

            convert_command += [self._act_bitwidth_flag + "=" + str(self._params["act_bw"])]
            convert_command += [self._bias_bitwidth_flag + "=" + str(self._params["bias_bw"])]
            convert_command += [self._output_path_flag, snpe_quantized_dlc_path]
            if self._params["float_bw"]:
                convert_command += [self._float_bitwidth_flag + "=" + str(self._params["float_bw"])]
            if self._params["no_weight_quantization"]:
                convert_command += [self._no_weight_quantization_flag]
            else:
                if self._params["quantizer"] == "symmetric":
                    convert_command += [self._use_symmetric_quantize_weights_flag]
                else:
                    convert_command += [
                        self._weights_bitwidth_flag + "=" + str(self._params["weight_bw"])
                    ]

            if self._params["quantizer"] == "adjusted":
                convert_command += [self._use_adjusted_weights_quantizer_flag]
            if self._params["quantizer"] == "enhanced":
                convert_command += [self._use_enhanced_quantizer_flag]
            if self._params["override_params"]:
                convert_command += [self._override_params_flag]

            if self._executable != "snpe-dlc-quant.exe":
                if self._params["enable_htp"]:
                    convert_command += [self._enable_htp_flag]
                if self._params["htp_socs"]:
                    convert_command += [self._htp_socs_flag + "=" + self._params["htp_socs"]]
            if self._params["extra_quantizer_args"]:
                convert_command += [self._params["extra_quantizer_args"]]

            convert_command += self._flags
            convert_command_str = ' '.join(convert_command)
            log_string = 'Running DLC quantize with: ' + \
                        'Inputs: ' + str(self._params["input_dlc_path"]) + ' ' + \
                        'Outputs: ' + str(snpe_quantized_dlc_path)
            self._logger.info(log_string)
            code, _, err = self._environment.host_device_obj.execute(
                commands=[convert_command_str], cwd=self._environment.engine_path,
                env=self._environment.host_env)
            if code != 0:
                raise InferenceEngineError(
                    get_message('"ERROR_INFERENCE_ENGINE_SNPE_DLC_QUANTIZED_FAILED"'))
            self._logger.info('DLC model quantized successfully')
        except subprocess.CalledProcessError as exc:
            self._logger.error(str(exc))
            raise InferenceEngineError(
                get_message('"ERROR_INFERENCE_ENGINE_SNPE_DLC_QUANTIZED_FAILED"'))
        return code


class ExecuteSNPEConversionAndQuantization:

    def __init__(self, framework: str, host_device: str, params: dict, output_directory: str,
                 logger=None, verbose: str = "info", engine_path: str = None):
        '''
        Performs quantization of the given model with snpe-{onnx/tflite/tf}-converter/quantizer
        :param framework: One of {onnx/tflite/tf}
        :param host_device: Device on which quantization will be performed. one of ["x86_64-windows-msvc", "wos"]
        :param params: A dictionary with following keys:
            {
                "model_path", "input_list", "quantizer", "output_tensors",
                "quantization_overrides", "input_tensors",
                "weight_bw", "bias_bw", "act_bw", "float_bw",
                "extra_converter_args", "htp_socs",
                "extra_quantizer_args", "override_params", "no_weight_quantization"
            }
        :param output_directory: directory where quantized files will be dumped
        :param logger: logging handle
        :param verbose: logging level
        :param engine_path: SNPE SDK path
        '''
        # set engine path if none specified

        self._params = params
        self._framework = framework
        self._host_device = host_device
        self._output_directory = output_directory
        self._logger = logger
        self._verbose = verbose
        self._engine_path = engine_path

    def convert_and_quantize(self, quantized_model_save_dir, quantized_model_name):
        '''
        Given a quant Scheme generates the quantized model
        '''
        # Perform the dlc conversion
        conversion_params = {
            "model_path": self._params["model_path"],
            "output_tensors": self._params["output_tensors"],
            "quantization_overrides": self._params["quantization_overrides"],
            "input_tensors": self._params["input_tensors"],
            "extra_converter_args": self._params["extra_converter_args"]
        }
        self._converter = ExecuteSNPEConversion(framework=self._framework,
                                                host_device=self._host_device,
                                                output_directory=self._output_directory,
                                                params=conversion_params, logger=self._logger,
                                                verbose=self._verbose,
                                                engine_path=self._engine_path)
        self._converter.convert(model_dlc_save_dir=quantized_model_save_dir, model_dlc_name='base')

        # Perform dlc quantization
        input_dlc_path = os.path.join(quantized_model_save_dir, 'base.dlc')
        quantization_params = {
            "input_dlc_path": input_dlc_path,
            "input_list": self._params["input_list"],
            "quantizer": self._params["quantizer"],
            "quantization_overrides": self._params["quantization_overrides"],
            "weight_bw": self._params["weight_bw"],
            "bias_bw": self._params["bias_bw"],
            "act_bw": self._params["act_bw"],
            "float_bw": self._params["float_bw"],
            "htp_socs": self._params["htp_socs"],
            "enable_htp": self._params["enable_htp"],
            "extra_quantizer_args": self._params["extra_quantizer_args"],
            "override_params": self._params["override_params"],
            "no_weight_quantization": self._params["no_weight_quantization"]
        }
        self._quantizer = ExecuteSNPEQuantization(framework=self._framework,
                                                  host_device=self._host_device,
                                                  params=quantization_params,
                                                  output_directory=self._output_directory,
                                                  logger=self._logger, verbose=self._verbose,
                                                  engine_path=self._engine_path)
        self._quantizer.quantize(quantized_model_save_dir, quantized_model_name)
