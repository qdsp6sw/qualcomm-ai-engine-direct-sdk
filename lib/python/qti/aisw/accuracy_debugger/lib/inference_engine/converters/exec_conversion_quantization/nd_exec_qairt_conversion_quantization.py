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
from qti.aisw.accuracy_debugger.lib.utils.nd_errors import get_message
from qti.aisw.accuracy_debugger.lib.inference_engine.converters.exec_conversion_quantization.nd_exec_conversion_quantization import ExecuteConversionAndQuantization
from qti.aisw.accuracy_debugger.lib.utils.nd_constants import ComponentType


class ExecuteQAIRTConversion(ExecuteConversionAndQuantization):

    def __init__(self, host_device: str, params: dict, output_directory: str, logger=None,
                 verbose: str = "info", engine_path: str = None):
        '''
        Performs conversion of the given model with qairt-converter
        :param host_device: Device on which converter will be performed.
        :param params: A dictionary with following keys:
            {
                "input_network",
                "output_path",
                "input_dims",
                "output_tensors",
                "quantization_overrides",
                "io_config"
                "extra_converter_args"
            }
        }
        :param output_directory: host output directory
        :param logger: logging handle
        :param verbose: logging level
        :param engine_path: QAIRT SDK path
        '''
        super().__init__(engine='QAIRT', framework=None, host_device=host_device,
                         output_directory=output_directory, params=params, engine_path=engine_path,
                         logger=logger, verbose=verbose)
        self._get_coverter(self._config_data[ComponentType.converter.value])

    def convert(self):
        """
        Convert a model into a dlc.
        """
        convert_command = self._converter.build_convert_command(
            self._params["input_network"], inputs_dims=self._params["input_dims"],
            output_tensors=self._params["output_tensors"], output_path=self._params["output_path"],
            quantization_overrides=self._params["quantization_overrides"],
            io_config=self._params["io_config"],
            extra_converter_args=self._params["extra_converter_args"])

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


class ExecuteQAIRTQuantization(ExecuteConversionAndQuantization):

    def __init__(self, host_device: str, params: dict, output_directory: str, logger=None,
                 verbose: str = "info", engine_path: str = None):
        '''
        Performs quantization of the given model with qairt-quantizer
        :param host_device: Device on which quantization will be performed.
        :param params: A dictionary with following keys:
            {
                "input_dlc_path", "output_path", "calibration_input_list", use_per_channel_quantization",
                "weights_bitwidth", "bias_bitwidth", "act_bitwidth", "use_native_output_files",
                "act_quantizer_calibration", "param_quantizer_calibration",
                "act_quantizer_schema", "param_quantizer_schema", "percentile_calibration_value",
                "use_per_row_quantization", "extra_quantizer_args", "use_native_input_files,
                "float_bitwidth", "float_fallback"
            }
        :param output_directory: host output directory
        :param logger: logging handle
        :param verbose: logging level
        :param engine_path: QAIRT SDK path
        '''
        super().__init__(engine='QAIRT', framework=None, host_device=host_device, params=params,
                         output_directory=output_directory, engine_path=engine_path, logger=logger,
                         verbose=verbose)

        self._snpe_quantizer_config = self._config_data[ComponentType.quantizer.value]
        self._snpe_quantizer_arguments = self._snpe_quantizer_config["arguments"]

    def quantize(self):
        """
        Execute DLC quantization.
        """
        try:
            quantizer_command = [
                self._snpe_quantizer_config["executable"],
                self._snpe_quantizer_arguments["dlc_path"], self._params["input_dlc_path"],
                self._snpe_quantizer_arguments["output_path"], self._params["output_path"],
                self._snpe_quantizer_arguments["dump_encoding_json"]
            ]
            if self._params["calibration_input_list"]:
                quantizer_command += [
                    self._snpe_quantizer_arguments["input_list"],
                    self._params["calibration_input_list"]
                ]

                quantizer_command += [
                    self._snpe_quantizer_arguments["act_bitwidth"],
                    str(self._params["act_bitwidth"])
                ]
                quantizer_command += [
                    self._snpe_quantizer_arguments["bias_bitwidth"],
                    str(self._params["bias_bitwidth"])
                ]
                quantizer_command += [
                    self._snpe_quantizer_arguments["weights_bitwidth"],
                    str(self._params["weights_bitwidth"])
                ]
                quantizer_command += [
                    self._snpe_quantizer_arguments["act_quantizer_schema"],
                    self._params["act_quantizer_schema"]
                ]
                quantizer_command += [
                    self._snpe_quantizer_arguments["param_quantizer_schema"],
                    self._params["param_quantizer_schema"]
                ]

                if self._params["act_quantizer_calibration"]:
                    quantizer_command += [
                        self._snpe_quantizer_arguments["act_quantizer_calibration"],
                        self._params["act_quantizer_calibration"]
                    ]
                if self._params["param_quantizer_calibration"]:
                    quantizer_command += [
                        self._snpe_quantizer_arguments["param_quantizer_calibration"],
                        self._params["param_quantizer_calibration"]
                    ]
                if self._params["percentile_calibration_value"]:
                    quantizer_command += [
                        self._snpe_quantizer_arguments["percentile_calibration_value"],
                        str(self._params["percentile_calibration_value"])
                    ]
                if self._params["use_per_channel_quantization"]:
                    quantizer_command.append(
                        self._snpe_quantizer_arguments["use_per_channel_quantization"])

                if self._params["use_per_row_quantization"]:
                    quantizer_command.append(
                        self._snpe_quantizer_arguments["use_per_row_quantization"])
            elif self._params["float_fallback"]:
                quantizer_command.append(self._snpe_quantizer_arguments["float_fallback_flag"])

            if self._params["float_bitwidth"]:
                quantizer_command += [
                    self._snpe_quantizer_arguments["float_bitwidth_flag"],
                    str(self._params["float_bitwidth"])
                ]

            if self._params["use_native_input_files"]:
                quantizer_command.append(self._snpe_quantizer_arguments["use_native_input_files"])

            if self._params["use_native_output_files"]:
                quantizer_command.append(self._snpe_quantizer_arguments["use_native_output_files"])

            if self._params["extra_quantizer_args"]:
                quantizer_command.append(self._params["extra_quantizer_args"])

            quantizer_command_str = ' '.join(quantizer_command)

            log_string = 'Running DLC quantize with: ' + \
                        'Inputs: ' + str(self._params["input_dlc_path"]) + ' ' + \
                        'Outputs: ' + str(self._params["output_path"])
            self._logger.info(log_string)
            code, _, err = self._environment.host_device_obj.execute(
                commands=[quantizer_command_str], cwd=self._environment.engine_path,
                env=self._environment.host_env)
            if code != 0:
                raise InferenceEngineError(f"Quantization failed with error: {err}")
            self._logger.info('DLC model quantized successfully')
        except subprocess.CalledProcessError as exc:
            self._logger.error(str(exc))
            raise InferenceEngineError(f"Quantization failed with error: {str(exc)}")
        return code


class ExecuteQAIRTConversionAndQuantization:

    def __init__(self, host_device: str, params: dict, output_directory: str, logger=None,
                 verbose: str = "info", engine_path: str = None):
        '''
        Performs conversion and quantization of the given model
        :param host_device: Device on which conversion and quantization will be performed.
        :param params: A dictionary with following keys:
            {
                "input_network", "output_path", "input_dims", "output_tensors", "io_config",
                "quantization_overrides", "extra_converter_args",
                "calibration_input_list", "weights_bitwidth", "bias_bitwidth", "act_bitwidth",
                "param_quantizer_schema", "act_quantizer_schema", "param_quantizer_calibration",
                "act_quantizer_calibration", "percentile_calibration_value",
                "use_per_row_quantization", "use_per_channel_quantization", "float_fallback",
                "float_bitwidth", "use_native_input_files", use_native_output_files",
                "extra_quantizer_args"
            }
        :param output_directory: host output directory
        :param logger: logging handle
        :param verbose: logging level
        :param engine_path: QAIRT SDK path
        '''
        self._params = params
        self._host_device = host_device
        self._output_directory = output_directory
        self._logger = logger
        self._verbose = verbose
        self._engine_path = engine_path

    def convert_and_quantize(self):
        '''
        Given a quant Scheme generates the quantized model
        '''
        # Perform the dlc conversion
        converted_dlc_path = self._params["output_path"].rsplit('.', 1)[0] + "_converted.dlc"
        conversion_params = {
            "input_network": self._params["input_network"],
            "output_path": converted_dlc_path,
            "input_dims": self._params["input_dims"],
            "output_tensors": self._params["output_tensors"],
            "quantization_overrides": self._params["quantization_overrides"],
            "io_config": self._params["io_config"],
            "extra_converter_args": self._params["extra_converter_args"]
        }
        self._converter = ExecuteQAIRTConversion(host_device=self._host_device,
                                                 output_directory=self._output_directory,
                                                 params=conversion_params, logger=self._logger,
                                                 verbose=self._verbose,
                                                 engine_path=self._engine_path)
        self._converter.convert()

        # Perform dlc quantization
        quantization_params = {
            "input_dlc_path": converted_dlc_path,
            "output_path": self._params["output_path"],
            "calibration_input_list": self._params["calibration_input_list"],
            "weights_bitwidth": self._params["weights_bitwidth"],
            "bias_bitwidth": self._params["bias_bitwidth"],
            "act_bitwidth": self._params["act_bitwidth"],
            "param_quantizer_schema": self._params["param_quantizer_schema"],
            "act_quantizer_schema": self._params["act_quantizer_schema"],
            "param_quantizer_calibration": self._params["param_quantizer_calibration"],
            "act_quantizer_calibration": self._params["act_quantizer_calibration"],
            "percentile_calibration_value": self._params["percentile_calibration_value"],
            "use_per_row_quantization": self._params["use_per_row_quantization"],
            "use_per_channel_quantization": self._params["use_per_channel_quantization"],
            "float_fallback": self._params["float_fallback"],
            "float_bitwidth": self._params["float_bitwidth"],
            "use_native_input_files": self._params["use_native_input_files"],
            "use_native_output_files": self._params["use_native_output_files"],
            "extra_quantizer_args": self._params["extra_quantizer_args"]
        }
        self._quantizer = ExecuteQAIRTQuantization(host_device=self._host_device,
                                                   params=quantization_params,
                                                   output_directory=self._output_directory,
                                                   logger=self._logger, verbose=self._verbose,
                                                   engine_path=self._engine_path)
        self._quantizer.quantize()
