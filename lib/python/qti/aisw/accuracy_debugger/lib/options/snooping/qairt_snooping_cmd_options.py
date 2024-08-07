# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import argparse
import os

from packaging import version

import numpy as np

from qti.aisw.accuracy_debugger.lib.options.cmd_options import CmdOptions
from qti.aisw.accuracy_debugger.lib.utils.nd_constants import Architecture_Target_Types, Engine, Runtime, \
    Android_Architectures, X86_Architectures, \
    Device_type, Qnx_Architectures, Windows_Architectures, X86_windows_Architectures, Aarch64_windows_Architectures
from qti.aisw.accuracy_debugger.lib.utils.nd_exceptions import ParameterError, UnsupportedError
from qti.aisw.accuracy_debugger.lib.utils.nd_path_utility import get_absolute_path
from qti.aisw.accuracy_debugger.lib.utils.nd_namespace import update_layer_options


class QAIRTSnoopingCmdOptions(CmdOptions):

    def __init__(self, args, type, validate_args=True):
        super().__init__(component="snooping", args=args, engine="QAIRT",
                         validate_args=validate_args)
        self.type = type

    def _base_initialize(self):

        required_args = self.parser.add_argument_group('Required Arguments')
        converter_args = self.parser.add_argument_group('QAIRT Converter Arguments')
        quantizer_args = self.parser.add_argument_group('QAIRT Quantizer Arguments')
        net_run_args = self.parser.add_argument_group('Net-run Arguments')
        optional = self.parser.add_argument_group('Other optional Arguments')

        required_args.add_argument(
            '-f', '--framework', nargs='+', type=str, required=True,
            help='Framework type to be used, followed optionally by framework '
            'version.')

        required_args.add_argument(
            '-r', '--runtime', type=str.lower, required=True,
            choices=[r.value for r in Runtime if r.value not in ['htp']], help='Runtime to be used.\
                Note: In case of SNPE execution(--executor_type snpe), aic runtime is not supported.'
        )

        required_args.add_argument(
            '-a', '--architecture', type=str, required=True,
            choices=Architecture_Target_Types.target_types.value,
            help='Name of the architecture to use for inference engine.\
                Note: In case of SNPE execution(--executor_type snpe), aarch64-qnx architecture is not supported.'
        )

        required_args.add_argument(
            '-l', '--input_list', type=str, required=True,
            help="Path to the input list text file to run inference(used with net-run). \
                              Note: When having multiple entries in text file, in order to save memory and time, \
                                  you can pass --debug_mode_off to skip intermediate outputs dump.")

        converter_args.add_argument('--input_network', '--model_path', dest='model_path', type=str,
                                    default=None, required=False, help='Path to the model file(s).')

        converter_args.add_argument(
            '--input_tensor', nargs='+', action='append', required=False,
            help='The name and dimension of all the input buffers to the network \
            specified in the format [input_name comma-separated-dimensions], \
            for example: \'data\' 1,224,224,3. Note that the quotes should always be included in \
            order to handle special characters, spaces, etc. \
            For multiple inputs, specify multiple --desired_input_shape on the command line like:  \
            --desired_input_shape "data1" 1,224,224,3 \
            --desired_input_shape "data2" 1,50,100,3 \
            NOTE: Required for TensorFlow and PyTorch. Optional for Onnx and Tflite. \
            In case of Onnx, this feature works only with Onnx 1.6.0 and above.')

        converter_args.add_argument(
            '--out_tensor_node', '--output_tensor', dest='output_tensor', type=str, required=False,
            action='append', help='Name of the graph\'s output Tensor Names. \
            Multiple output names should be provided separately like: \
            --out_tensor_node out_1 --out_tensor_node out_2 \
            NOTE: Required for TensorFlow. Optional for Onnx, Tflite and PyTorch')

        converter_args.add_argument(
            '--io_config', type=str, required=False, default=None,
            help="Use this option to specify a yaml file for input and output options.")

        converter_args.add_argument(
            '-qo', '--quantization_overrides', type=str, required=False, default=None,
            help="Path to quantization overrides json file.")

        converter_args.add_argument(
            '--extra_converter_args', type=str, required=False, default=None,
            help="additional converter arguments in a quoted string. \
                 example: --extra_converter_args 'arg1=value1;arg2=value2'")

        quantizer_args.add_argument(
            '--calibration_input_list', type=str, required=False, default=None,
            help='Path to the inputs list text file to run quantization(used with qairt-quantizer).')

        quantizer_args.add_argument(
            '-bbw', '--bias_bitwidth', type=int, required=False, default=8, choices=[8, 32],
            help="option to select the bitwidth to use when quantizing the bias. default 8")
        quantizer_args.add_argument(
            '-abw', '--act_bitwidth', type=int, required=False, default=8, choices=[8, 16],
            help="option to select the bitwidth to use when quantizing the activations. default 8")
        quantizer_args.add_argument(
            '-wbw', '--weights_bitwidth', type=int, required=False, default=8, choices=[8, 16],
            help="option to select the bitwidth to use when quantizing the weights. default 8")

        quantizer_args.add_argument(
            '-nif', '--use_native_input_files', action="store_true", default=False, required=False,
            help="Specifies that the input files will be parsed in the data type native to the graph.\
                                    If not specified, input files will be parsed in floating point."
        )
        quantizer_args.add_argument(
            '-nof', '--use_native_output_files', action="store_true", default=False, required=False,
            help="Specifies that the output files will be generated in the data \
                                    type native to the graph. If not specified, output files will \
                                    be generated in floating point.")

        quantizer_args.add_argument(
            '--act_quantizer_calibration', type=str.lower, required=False, default="min-max",
            choices=['min-max', 'sqnr', 'entropy', 'mse', 'percentile'],
            help="Specify which quantization calibration method to use for activations. \
                                    This option has to be paired with --act_quantizer_schema.")

        quantizer_args.add_argument(
            '--param_quantizer_calibration', type=str.lower, required=False, default="min-max",
            choices=['min-max', 'sqnr', 'entropy', 'mse', 'percentile'],
            help="Specify which quantization calibration method to use for parameters.\
                                    This option has to be paired with --param_quantizer_schema.")

        quantizer_args.add_argument(
            '--act_quantizer_schema', type=str.lower, required=False, default='asymmetric',
            choices=['asymmetric',
                     'symmetric'], help="Specify which quantization schema to use for activations. \
                                    Can not be used together with act_quantizer.")

        quantizer_args.add_argument(
            '--param_quantizer_schema', type=str.lower, required=False, default='asymmetric',
            choices=['asymmetric',
                     'symmetric'], help="Specify which quantization schema to use for parameters. \
                                    Can not be used together with param_quantizer.")

        quantizer_args.add_argument('--percentile_calibration_value', type=float, required=False,
                                    default=99.99,
                                    help="Value must lie between 90 and 100. Default is 99.99")

        quantizer_args.add_argument(
            '--use_per_channel_quantization', action="store_true", default=False,
            help="Use per-channel quantization for convolution-based op weights.\
            Note: This will replace built-in model QAT encodings when used for a given weight.")

        quantizer_args.add_argument(
            '--use_per_row_quantization', action="store_true", default=False,
            help="Use this option to enable rowwise quantization of Matmul and FullyConnected ops.")

        quantizer_args.add_argument(
            '--float_fallback', action="store_true", default=False,
            help='Use this option to enable fallback to floating point (FP) instead of fixed point. '
            'This option can be paired with --float_bitwidth to indicate the bitwidth for '
            'FP (by default 32). If this option is enabled, then input list must '
            'not be provided and --ignore_encodings must not be provided. '
            'The external quantization encodings (encoding file/FakeQuant encodings) '
            'might be missing quantization parameters for some interim tensors. '
            'First it will try to fill the gaps by propagating across math-invariant '
            'functions. If the quantization params are still missing, '
            'then it will apply fallback to nodes to floating point.')

        quantizer_args.add_argument(
            '--float_bitwidth', type=int, required=False, default=None, choices=[32, 16],
            help='Use this option to enable fallback to floating point (FP) instead of fixed \
                    point. \
                    This option can be paired with --float_bitwidth to indicate the bitwidth for \
                    FP (by default 32). \
                    If this option is enabled, then input list must not be provided and \
                    --ignore_encodings must not be provided. \
                    The external quantization encodings (encoding file/FakeQuant encodings) \
                    might be missing quantization parameters for some interim tensors. \
                    First it will try to fill the gaps by propagating across math-invariant \
                    functions. If the quantization params are still missing, \
                    then it will apply fallback to nodes to floating point.')

        quantizer_args.add_argument(
            '--extra_quantizer_args', type=str, required=False, default=None,
            help="additional quantizer arguments in a quoted string. \
                    example: --extra_quantizer_args 'arg1=value1;arg2=value2'")

        net_run_args.add_argument(
            '--perf_profile', type=str.lower, required=False, default="balanced", choices=[
                'low_balanced', 'balanced', 'default', 'high_performance',
                'sustained_high_performance', 'burst', 'low_power_saver', 'power_saver',
                'high_power_saver', 'extreme_power_saver', 'system_settings'
            ], help=
            'Specifies perf profile to set. Valid settings are "low_balanced" , "balanced" , "default", \
                    "high_performance" ,"sustained_high_performance", "burst", "low_power_saver", "power_saver", \
                    "high_power_saver", "extreme_power_saver", and "system_settings". \
                    Note: perf_profile argument is now deprecated for \
                    HTP backend, user can specify performance profile \
                    through backend extension config now.')

        net_run_args.add_argument(
            '--profiling_level', type=str.lower, required=False, default=None,
            help='Enables profiling and sets its level. \
                                    For QNN executor, valid settings are "basic", "detailed" and "client" \
                                    For SNPE executor, valid settings are "off", "basic", "moderate", "detailed", and "linting". \
                                    Default is detailed.')

        net_run_args.add_argument(
            '--userlogs', type=str.lower, required=False, default=None,
            choices=["warn", "verbose", "info", "error", "fatal"], help="Enable verbose logging. \
                                        Note: This argument is applicable only when --executor_type snpe"
        )

        net_run_args.add_argument(
            '--log_level', type=str.lower, required=False, default=None,
            choices=['error', 'warn', 'info', 'debug', 'verbose'], help="Enable verbose logging. \
                                        Note: This argument is applicable only when --executor_type qnn"
        )

        net_run_args.add_argument(
            '--extra_runtime_args', type=str, required=False, default=None,
            help="additional net runner arguments in a quoted string. \
                example: --extra_runtime_args 'arg1=value1;arg2=value2'")

        optional.add_argument('--precision', choices=['int8', 'fp16', 'fp32'], default='int8',
                              help='Select precision for model(default is int8)')

        optional.add_argument(
            '--executor_type', type=str.lower, required=False, default=None,
            choices=['qnn', 'snpe'],
            help='Choose between qnn(qnn-net-run) and snpe(snpe-net-run) execution. \
                If not provided, qnn-net-run will be executed for QAIRT or QNN SDK, \
                or else snpe-net-run will be executed for SNPE SDK.')

        optional.add_argument(
            '--stage', type=str.lower, required=False, choices=['source', 'converted',
                                                                'quantized'], default='source',
            help='Specifies the starting stage in the Accuracy Debugger pipeline. \
            source: starting with a source framework model, \
            converted: starting with a converted model, \
            quantized: starting with a quantized model. \
            Default is source.')

        optional.add_argument('-p', '--engine_path', type=str, required=False,
                              help="Path to SDK folder.")

        optional.add_argument(
            '--deviceId', required=False, default=None,
            help='The serial number of the device to use. If not passed, '
            'the first in a list of queried devices will be used for validation.')

        optional.add_argument('-v', '--verbose', action="store_true", default=False,
                              help="Set verbose logging at debugger tool level")

        optional.add_argument(
            '--host_device', type=str, required=False, default='x86',
            choices=['x86', 'x86_64-windows-msvc', 'wos'],
            help='The device that will be running conversion. Set to x86 by default.')

        optional.add_argument('-w', '--working_dir', type=str, required=False,
                                default='working_directory',
                                help='Working directory for the {} to store temporary files. '.format(self.component) + \
                                    'Creates a new directory if the specified working directory does not exist')
        optional.add_argument('--output_dirname', type=str, required=False,
                                default='<curr_date_time>',
                                help='output directory name for the {} to store temporary files under <working_dir>/{} .'.format(self.component,self.component) + \
                                    'Creates a new directory if the specified working directory does not exist')

        optional.add_argument('--debug_mode_off', dest="debug_mode", action="store_false",
                              required=False,
                              help="This option can be used to avoid dumping intermediate outputs.")
        optional.set_defaults(debug_mode=True)

        optional.add_argument(
            '--args_config', type=str, required=False,
            help="Path to a config file with arguments. This can be used to feed arguments to "
            "the AccuracyDebugger as an alternative to supplying them on the command line.")

        optional.add_argument('--remote_server', type=str, required=False, default=None,
                              help="ip address of remote machine")
        optional.add_argument('--remote_username', type=str, required=False, default=None,
                              help="username of remote machine")
        optional.add_argument('--remote_password', type=str, required=False, default=None,
                              help="password of remote machine")

        optional.add_argument(
            '--golden_output_reference_directory', '--golden_dir_for_mapping',
            dest='golden_output_reference_directory', type=str, required=False, default=None,
            help="Optional parameter to indicate the directory of the goldens, \
                it's used for tensor mapping without running model with framework runtime.")

        optional.add_argument(
            '--disable_offline_prepare', action="store_true", default=False,
            help=f"Use this option to disable offline preparation. \
                                  Note: By default offline preparation will be done for DSP/HTP runtimes."
        )

        optional.add_argument(
            '--backend_extension_config', type=str, required=False, default=None,
            help="Path to config to be used with qnn-context-binary-generator. \
                Note: This argument is applicable only when --executor_type qnn")

        optional.add_argument(
            '--context_config_params', type=str, default=None, required=False,
            help="optional context config params in a quoted string. \
                example: --context_config_params 'context_priority=high; cache_compatibility_mode=strict' \
                    Note: This argument is applicable only when --executor_type qnn")

        optional.add_argument(
            '--graph_config_params', type=str, default=None, required=False,
            help="optional graph config params in a quoted string. \
                example: --graph_config_params 'graph_priority=low; graph_profiling_num_executions=10'"
        )

        optional.add_argument(
            '--extra_contextbin_args', type=str, required=False, default=None, help=
            "Additional context binary generator arguments in a quoted string(applicable only when --executor_type qnn). \
                example: --extra_contextbin_args 'arg1=value1;arg2=value2'")

        optional.add_argument(
            '--start_layer', type=str, default=None, required=False, help=
            "save all intermediate layer outputs from provided start layer to bottom layer of model. \
                                    Can be used in conjunction with --end_layer.")
        optional.add_argument(
            '--end_layer', type=str, default=None, required=False, help=
            "save all intermediate layer outputs from top layer to  provided end layer of model. \
                                  Can be used in conjunction with --start_layer.")

        optional.add_argument(
            '--add_layer_outputs', default=[], help="Output layers to be dumped. \
                                e.g: node1,node2")

        optional.add_argument(
            '--add_layer_types', default=[],
            help='outputs of layer types to be dumped. e.g :Resize,Transpose.\
                                All enabled by default.')

        optional.add_argument(
            '--skip_layer_types', default=[],
            help='comma delimited layer types to skip dumping. e.g :Resize,Transpose')

        optional.add_argument(
            '--skip_layer_outputs', default=[],
            help='comma delimited layer output names to skip dumping. e.g: node1,node2')
        optional.add_argument('--disable_graph_optimization', action="store_true",
                              help="Disables basic model optimization")
        optional.add_argument('--onnx_custom_op_lib', default=None,
                              help="path to onnx custom operator library")

    def _verify_update_base_parsed_args(self, parsed_args):

        if parsed_args.golden_output_reference_directory:
            parsed_args.golden_output_reference_directory = get_absolute_path(
                parsed_args.golden_output_reference_directory)
        parsed_args.engine_path = get_absolute_path(parsed_args.engine_path)

        # get framework and framework version if possible
        parsed_args.framework_version = None
        if len(parsed_args.framework) > 2:
            raise ParameterError("Maximum two arguments required for framework.")
        elif len(parsed_args.framework) == 2:
            parsed_args.framework_version = parsed_args.framework[1]

        parsed_args.framework = parsed_args.framework[0]
        if parsed_args.framework != 'onnx' and self.type != "oneshot-layerwise":
            raise UnsupportedError("Layerwise snooping supports only onnx framework")

        if parsed_args.runtime == Runtime.htp.value and parsed_args.architecture != 'x86_64-linux-clang':
            raise ParameterError("Runtime htp supports only x86_64-linux-clang architecture")

        if parsed_args.add_layer_types:
            parsed_args.add_layer_types = parsed_args.add_layer_types.split(',')

        if parsed_args.skip_layer_types:
            parsed_args.skip_layer_types = parsed_args.skip_layer_types.split(',')

        if parsed_args.add_layer_types or parsed_args.skip_layer_types or parsed_args.skip_layer_outputs:
            parsed_args.add_layer_outputs = update_layer_options(parsed_args)

        user_provided_dtypes = []

        if parsed_args.input_tensor is not None:
            # get proper input_tensor format
            for tensor in parsed_args.input_tensor:
                if len(tensor) < 3:
                    raise argparse.ArgumentTypeError(
                        "Invalid format for input_tensor, format as "
                        "--input_tensor \"INPUT_NAME\" INPUT_DIM INPUT_DATA.")
                elif len(tensor) == 3:
                    user_provided_dtypes.append('float32')
                elif len(tensor) == 4:
                    user_provided_dtypes.append(tensor[-1])
                tensor[2] = get_absolute_path(tensor[2])

        if parsed_args.calibration_input_list and parsed_args.precision in ['fp16', 'fp32']:
            raise ParameterError("When calibration_input_list is provided and precision can not be {}".format(parsed_args.precision))

        if not parsed_args.use_native_input_files:
            # Since underlying QNN tools are not supporting 64 bit inputs but our framework
            # diagnosis works on 64 bit, if model accepts 64 bit inputs
            # we need to reconvert the user provided 64 bit inputs through the input_list
            # into 32 bits. Here, we loop over each line in the input_list.txt
            # and convert the 64 bit input and dump them into the working directory
            # and finally we create new input_list containing 32 bit input paths
            converted_input_file_dump_path = os.path.join(parsed_args.working_dir, "inputs_32")
            os.makedirs(converted_input_file_dump_path, exist_ok=True)
            new_input_list_file_path = os.path.join(converted_input_file_dump_path,
                                                    'input_list.txt')
            new_input_list_file = open(new_input_list_file_path, 'w')
            with open(parsed_args.input_list, 'r') as file:
                for line in file.readlines():
                    line = line.rstrip().lstrip().split('\n')[0]
                    if line:
                        new_file_name_and_path = []
                        file_name_and_paths = [
                            file_name_and_path.split(':=')
                            if ':=' in file_name_and_path else [None, file_name_and_path]
                            for file_name_and_path in line.split()
                        ]
                        for user_provided_dtype, file_name_and_path in zip(
                                user_provided_dtypes, file_name_and_paths):
                            user_provided_tensor = np.fromfile(file_name_and_path[1],
                                                               dtype=user_provided_dtype)
                            if user_provided_dtype == "int64":
                                converted_tensor = user_provided_tensor.astype(np.int32)
                            elif user_provided_dtype == "float64":
                                converted_tensor = user_provided_tensor.astype(np.float32)
                            else:
                                converted_tensor = user_provided_tensor
                            file_name = os.path.join(converted_input_file_dump_path,
                                                     os.path.basename(file_name_and_path[1]))
                            converted_tensor.tofile(file_name)
                            if file_name_and_path[0] is not None:
                                new_file_name_and_path.append(file_name_and_path[0] + ":=" +
                                                              file_name)
                            else:
                                new_file_name_and_path.append(file_name)
                        new_input_list_file.write(" ".join(new_file_name_and_path) + "\n")
            new_input_list_file.close()
            parsed_args.input_list = new_input_list_file_path

        return parsed_args

    def get_all_associated_parsers(self):
        parsers_to_be_validated = [self.parser]

        return parsers_to_be_validated
