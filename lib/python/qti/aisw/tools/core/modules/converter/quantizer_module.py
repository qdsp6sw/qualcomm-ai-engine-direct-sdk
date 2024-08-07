# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import os
import logging
from argparse import Namespace
from typing import Literal
from pydantic import model_validator
from qti.aisw.converters.common.dlc_quantizer import DLCQuantizer
from qti.aisw.converters.common.utils import converter_utils
from qti.aisw.tools.core.modules.compliance.function_signature_compliance import *
from qti.aisw.tools.core.modules.definitions.schema import *
from qti.aisw.tools.core.modules.definitions.interface import *
from qti.aisw.tools.core.modules.converter.common import BackendInfoConfig
from qti.aisw.converters.common.backend_awareness import BackendInfo
from qti.aisw.tools.core.modules.converter.constants import LOGLEVEL


quant_calibration = Literal["min-max", "sqnr", "entropy", "mse", "percentile"]
quant_schema = Literal["asymmetric", "symmetric"]


class QuantizerInputConfig(AISWBaseModel):

    input_dlc: FilePath = Field(description="Path to input dlc file.")
    output_dlc: str = Field(default="",
                            description="Path to the dlc container containing the model for which fixed-point"
                                        " encoding metadata should be generated.")
    input_list: Optional[FilePath] = Field(default=None, description="Path to a file specifying the input raw data.")
    float_fallback: bool = Field(default=False,
                                 description="Option to enable fallback to floating point (FP) instead of fixed point.")
    algorithms: Optional[List[str]] = Field(default=[], description="Option to select optimization algorithm.")
    bias_bitwidth: Literal[8, 32] = Field(default=8,
                                          description="Option to select the bitwidth to use when "
                                                      "quantizing the biases, either 8 (default) or 32.")
    act_bitwidth: Literal[8, 16] = Field(default=8,
                                         description="Option to select the bitwidth to use when "
                                                     "quantizing the activations, either 8 (default) or 16.")
    weights_bitwidth: Literal[8, 4] = Field(default=8,
                                            description="Option to select the bitwidth to use when"
                                                        "quantizing the weights, either 4 or 8 (default).")
    float_bitwidth: Literal[32, 16] = Field(default=32,
                                            description="Convert the graph to specified float bitwidth.")
    float_bias_bitwidth: Optional[Literal[16, 32]] = Field(default=None,
                                                           description="Option to select the bitwidth to use for float "
                                                                       "bias tensor.")
    keep_weights_quantized: bool = Field(default=False, description="Use this option to keep the weights quantized even"
                                                                    " when the output of the op is in floating point.")
    use_aimet_quantizer: bool = Field(default=False,
                                      description="Use AIMET for Quantization instead of QNN IR quantizer.")

    config_file: Optional[FilePath] = Field(default=None, description="Use this argument to pass the path of the config"
                                                                      " YAML file with quantizer options.")

    ignore_encodings: bool = Field(default=False,
                                   description="Use only quantizer generated encodings, ignoring any user"
                                               " or model provided encodings.")
    use_per_channel_quantization: bool = Field(default=False,
                                               description="option to enable per-channel quantization for "
                                                           "convolution-based op weights.")
    use_per_row_quantization: bool = Field(default=False,
                                           description="option to enable row wise quantization of Matmul and "
                                                       "FullyConnected ops.")
    use_native_input_files: bool = Field(default=False,
                                         description="Boolean flag to indicate how to read input files.\n"
                                                     "False(default): Reads inputs as floats and quantizes if necessary based on"
                                                     " quantization parameters in the model.\n"
                                                     "True: Reads inputs assuming the data type to be native to the Model.")
    use_native_output_files: bool = Field(default=False,
                                          description="Boolean flag to indicate the data type of the output files.\n"
                                                      "False(default): Output the file as floats.\n"
                                                      "True: Outputs the file that is native to the model.")
    restrict_quantization_steps: Optional[List[str]] = Field(default=[],
                                                             description="Specifies the number of steps to use for "
                                                                         "computing quantization encodings")
    act_quantizer_calibration: quant_calibration = Field(default="min-max",
                                                         description="Specify which quantization calibration method to "
                                                                     "use for activations supported values: min-max "
                                                                     "(default), sqnr, entropy, mse, percentile.")
    param_quantizer_calibration: quant_calibration = Field(default="min-max",
                                                           description="Specify which quantization calibration method "
                                                                       " to use for activations supported values: min-max "
                                                                       "(default), sqnr, entropy, mse, percentile.")
    act_quantizer_schema: quant_schema = Field(default="asymmetric",
                                               description="Specify which quantization schema to use for "
                                                           "activations supported values: asymmetric (default), symmetric")
    param_quantizer_schema: quant_schema = Field(default="asymmetric",
                                                 description="Specify which quantization schema to use for "
                                                             "activations supported values: asymmetric (default), symmetric")
    percentile_calibration_value: float = Field(default=99.99, ge=90.0, le=100,
                                                description="Specify the percentile value to be used with "
                                                            "Percentile calibration method. The specified "
                                                            "float value must lie within 90 and 100, default: 99.99")
    op_package_lib: Optional[List[str]] = Field(default=[],
                                                description="Option to pass list of op package library for quantization.")
    backend_info: Optional[BackendInfoConfig] = Field(default=None, description="Backend information.")

    @model_validator(mode="after")
    def validate_input_arguments(self):

        if self.float_fallback and (self.input_list or self.ignore_encodings):
            raise ValueError("'input_list' or 'ignore_encodings' option must not be provided when 'float_fallback' "
                             "option enabled.")

        if not (self.float_fallback or self.use_aimet_quantizer) and not self.input_list:
            raise ValueError("Please supply input to quantize model.")

        if self.config_file:
            if not self.use_aimet_quantizer:
                raise ValueError("Config file cannot be passed without use_aimet_quantizer flag")
            else:
                # Try loading YAML file
                import yaml
                try:
                    with open(str(self.config_file), "r") as conf:
                        yaml.safe_load(conf)
                except Exception as e:
                    self._logger.error("Failed to load config file with error: {}".format(e))
                    raise
        return self


class QuantizerOutputConfig(AISWBaseModel):
    dlc_output: str


class QuantizerModuleSchemaV1(ModuleSchema):
    _VERSION = ModuleSchemaVersion(major=0, minor=1, patch=0)
    _BACKENDS = None
    name: Literal["QuantizerModule"] = "QuantizerModule"
    path: Literal[str(inspect.getfile(os))] = str(inspect.getfile(os))
    arguments: QuantizerInputConfig
    outputs: Optional[QuantizerOutputConfig] = None
    backends: Optional[List[str]] = _BACKENDS
    version: ModuleSchemaVersion = _VERSION


@expect_module_compliance
class QAIRTQuantizer(Module):
    """
    User interface class for quantizer API
    """
    _SCHEMA = QuantizerModuleSchemaV1
    _PREVIOUS_SCHEMAS = []

    def __init__(self, logger=None):
        if not logger:
            logger = logging.getLogger("QuantizerLogger")
        converter_utils.LOGGER = logger
        super().__init__(logger)
        self._debug_level = LOGLEVEL.INFO

    @staticmethod
    def _get_args(config: QuantizerInputConfig) -> Namespace:
        """
        This method accepts quantizer input config and return arguments in dictionary object.
        1. Converter arguments names to quantizer internal names.
        2. Sets default values for suppressed arguments.
        Args:
            config: Converter input arguments config.
        Returns:
            Dictionary containing arguments.
        """
        option_dict = config.model_dump()
        option_dict["input_dlc"] = str(option_dict["input_dlc"])
        option_dict["input_list"] = str(option_dict["input_list"])
        # Unpack backend options
        backend_info = option_dict.pop("backend_info")
        if backend_info:
            option_dict["backend"] = backend_info["backend"]
            option_dict["soc_model"] = backend_info["soc_model"]
        else:
            option_dict["backend"] = None
            option_dict["soc_model"] = None

        if option_dict["float_bias_bitwidth"] is None:
            option_dict["float_bias_bitwidth"] = 0
        if option_dict["op_package_lib"]:
            option_dict["op_package_lib"] = ",".join(config.op_package_lib)
        else:
            option_dict["op_package_lib"] = ""

        # Check if output dlc path is supplied. If not dump output dlc at input dlc location.
        if not ("output_dlc" in option_dict and option_dict["output_dlc"]):
            input_dlc_dir, input_dlc_file_name = os.path.split(os.path.abspath(option_dict["input_dlc"]))
            output_dlc = os.path.join(input_dlc_dir, "quantized_"+input_dlc_file_name)
            option_dict["output_dlc"] = output_dlc
        # Disable legacy quantizer
        option_dict["disable_legacy_quantizer"] = True

        return option_dict

    def quantize(self, config: QuantizerInputConfig) -> QuantizerOutputConfig:
        """
        This is quantizer API method. It accepts input arguments in "AISWBaseModel" object type and returns
        outputs in "AISWBaseModel" object type.
        Args:
            config: "QuantizerInputConfig" object containing converter module input arguments.

        Returns:
            QuantizerOutputConfig object containing quantized DLC file path.
        """
        # Convert pydantic model to dictionary
        args_dict = self._get_args(config)
        args_dict["backend_info_obj"] = BackendInfo.get_instance(args_dict.pop("backend"), args_dict.pop("soc_model"))
        quantizer_command = converter_utils.sanitize_args(args_dict, args_to_ignore=['input_dlc', 'i', 'output_dlc', 'o'])
        try:
            dlc_quantizer = DLCQuantizer(**args_dict)
            dlc_quantizer.quantize()
            dlc_quantizer.save(quantizer_command)
        except Exception as e:
            self._logger.error("Quantization failed")
            raise e
        output_config = QuantizerOutputConfig(dlc_output=args_dict["output_dlc"])
        return output_config

    def enable_debug(self, debug_level: int, **kwargs: Unpack[TypedDict]) -> Optional[bool]:
        """
        Sets quantizer log level.
        Args:
            debug_level: LOGLEVEL.VERBOSE enables VERBOSE and higher level messages.
               LOGLEVEL.DEBUG enables DEBUG and higher level messages.
               LOGLEVEL.DEBUG_3 enables DEBUG_3 and higher level messages.
               LOGLEVEL.DEBUG_2 enables DEBUG_2 and higher level messages.
               LOGLEVEL.DEBUG_1 enables DEBUG_1 and higher level messages.
               LOGLEVEL.INFO enables INFO and higher level messages.
            **kwargs:

        Returns:
            bool: 'True' if debugging is enabled else return 'False'.
        """
        if debug_level < LOGLEVEL.INFO or debug_level > LOGLEVEL.VERBOSE:
            return False
        self._debug_level = debug_level
        converter_utils.setup_logging(self._debug_level)
        return True

    @property
    def _schema(self):
        return self._SCHEMA

    def get_logger(self) -> Any:
        pass

    def properties(self) -> Dict[str, Any]:
        return self._schema.model_json_schema()

    @classmethod
    def get_schema(cls, version: Optional[Union[str, ModuleSchemaVersion]] = None) -> Type[ModuleSchema]:
        if not version:
            return cls._SCHEMA

        if not isinstance(version, (str, ModuleSchemaVersion)):
            raise TypeError(
                f'Unknown type passed for version:{version!r} expected {type(str)} or {ModuleSchemaVersion!r} ')

        if isinstance(version, str):
            previous_schema_match = list(filter(lambda previous_schema: previous_schema.check_version_str(version),
                                                cls._PREVIOUS_SCHEMAS))
        else:
            previous_schema_match = list(filter(lambda previous_schema: version == previous_schema.get_version(),
                                                cls._PREVIOUS_SCHEMAS))

        if previous_schema_match:
            cls._LOGGER.info(f'Requested version: {version} matches previous schema version')
            return previous_schema_match[0]
        elif isinstance(version, str) and cls._SCHEMA.check_version_str(version):
            return cls._SCHEMA
        elif version == cls._SCHEMA.get_version():
            return cls._SCHEMA
        else:
            raise ValueError(f'Unknown version provided: {version}')

    @classmethod
    def get_schemas(cls) -> List[Type[ModuleSchema]]:
        if cls._PREVIOUS_SCHEMAS:
            return [cls._SCHEMA, *cls._PREVIOUS_SCHEMAS]
        return [cls._SCHEMA]