# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
from qti.aisw.accuracy_debugger.lib.inference_engine.converters.nd_converter import Converter
from qti.aisw.accuracy_debugger.lib.inference_engine import inference_engine_repository
from qti.aisw.accuracy_debugger.lib.utils.nd_constants import ComponentType, Engine


@inference_engine_repository.register(cls_type=ComponentType.converter, framework=None,
                                      engine=Engine.QAIRT, engine_version="1.0.0")
class QAIRTConverter(Converter):

    def __init__(self, context):
        super(QAIRTConverter, self).__init__(context)
        self.context = context

    def build_convert_command(self, model_path, inputs_dims=None, output_tensors=None, output_path=None,
                              quantization_overrides=None, io_config=None, extra_converter_args=None):
        convert_command_list = [self.context.executable, self.context.arguments["model_path_flag"],
                                model_path, self.context.arguments["output_path_flag"], output_path]

        if inputs_dims:
            for input_tensor, dimension in inputs_dims:
                convert_command_list.extend(
                    [self.context.arguments["input_shape_flag"], '"{}"'.format(input_tensor), dimension])

        if output_tensors:
            for output_tensor in output_tensors:
                convert_command_list.extend([self.context.arguments["output_tensor_flag"], output_tensor])

        if io_config:
            convert_command_list.extend([self.context.arguments["io_config_flag"], io_config])

        if quantization_overrides:
            convert_command_list.extend([self.context.arguments["quantization_overrides_flag"], quantization_overrides])

        if extra_converter_args:
            convert_command_list.extend([extra_converter_args])

        convert_command_str = ' '.join(convert_command_list)

        return convert_command_str

