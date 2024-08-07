# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import argparse

from qti.aisw.accuracy_debugger.lib.options.cmd_options import CmdOptions
from qti.aisw.accuracy_debugger.lib.utils.nd_constants import Engine, Runtime, Android_Architectures, \
    X86_Architectures, X86_windows_Architectures, Device_type, Architecture_Target_Types, Qnx_Architectures, \
        Windows_Architectures, FrameworkExtension, Aarch64_windows_Architectures
from qti.aisw.accuracy_debugger.lib.utils.nd_exceptions import ParameterError
from qti.aisw.accuracy_debugger.lib.utils.nd_path_utility import get_absolute_path, format_args, format_params
from qti.aisw.accuracy_debugger.lib.utils.nd_path_utility import retrieveSdkDir, get_sdk_type
from qti.aisw.accuracy_debugger.lib.utils.nd_framework_utility import get_framework_info
from qti.aisw.accuracy_debugger.lib.utils.nd_namespace import update_layer_options


class QAIRTInferenceEngineCmdOptions(CmdOptions):

    def __init__(self, engine, args, validate_args=True):
        super().__init__('inference_engine', args, engine, validate_args=validate_args)

    def initialize(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                              description="Script to run inference engine.")

        required_args = self.parser.add_argument_group('Required Arguments')
        converter_args = self.parser.add_argument_group('QAIRT Converter Arguments')
        quantizer_args = self.parser.add_argument_group('QAIRT Quantizer Arguments')
        net_run_args = self.parser.add_argument_group('Net-run Arguments')
        optional = self.parser.add_argument_group('Other optional Arguments')

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

        converter_args.add_argument(
            '--input_network', '--model_path', dest='input_network', type=str, default=None,
            required=False, help='Path to the model file(s). \
            This argument is mandatory when --stage is source(which is default).')

        converter_args.add_argument(
            '--desired_input_shape', '--input_tensor', dest='desired_input_shape', nargs='+',
            action='append', required=False,
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
            '--out_tensor_node', '--output_tensor', dest='out_tensor_node', type=str,
            required=False, action='append', help='Name of the graph\'s output Tensor Names. \
            Multiple output names should be provided separately like: \
            --out_tensor_node out_1 --out_tensor_node out_2 \
            NOTE: Required for TensorFlow. Optional for Onnx, Tflite and PyTorch')

        converter_args.add_argument(
            '--io_config', type=str, required=False, default=None,
            help="Use this option to specify a yaml file for input and output options.")

        converter_args.add_argument(
            '-qo', '--quantization_overrides', type=str, required=False, default=None,
            help="Path to quantization overrides json file."
        )

        converter_args.add_argument(
            '--extra_converter_args', type=str, required=False, default=None,
            help="additional converter arguments in a quoted string. \
                 example: --extra_converter_args 'arg1=value1;arg2=value2'")

        quantizer_args.add_argument(
            '--input_dlc', type=str, required=False, default=None,
            help='Path to the dlc container containing the model for which fixed-point encoding \
            metadata should be generated. \
            This argument is mandatory when --stage is either converted or quantized.')

        quantizer_args.add_argument(
            '--calibration_input_list', type=str, required=False, default=None,
            help='Path to the inputs list text file to run quantization(used with qairt-quantizer)')

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
            '--act_quantizer_calibration', type=str.lower, required=False, default='min-max',
            choices=['min-max', 'sqnr', 'entropy', 'mse', 'percentile'],
            help='Specify which quantization calibration method to use for activations '
                    'supported values: min-max (default), sqnr, entropy, mse, percentile '
                    'This option can be paired with --act_quantizer_schema to override the '
                    'quantization schema to use for activations otherwise default '
                    'schema(asymmetric) will be used')

        quantizer_args.add_argument(
            '--param_quantizer_calibration', type=str.lower, required=False, default='min-max',
            choices=['min-max', 'sqnr', 'entropy', 'mse', 'percentile'],
            help='Specify which quantization calibration method to use for parameters '
                    'supported values: min-max (default), sqnr, entropy, mse, percentile '
                    'This option can be paired with --act_quantizer_schema to override the '
                    'quantization schema to use for activations otherwise default '
                    'schema(asymmetric) will be used')

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

        self.initialized = True

    def verify_update_parsed_args(self, parsed_args):
        parsed_args.engine = self.engine
        parsed_args.engine_version = None
        parsed_args.input_list = get_absolute_path(parsed_args.input_list)
        parsed_args.calibration_input_list = get_absolute_path(parsed_args.calibration_input_list)
        parsed_args.engine_path = get_absolute_path(parsed_args.engine_path)
        parsed_args.input_network = get_absolute_path(parsed_args.input_network)
        parsed_args.input_dlc = get_absolute_path(parsed_args.input_dlc)

        # set engine path if not specified
        if parsed_args.engine_path is None:
            parsed_args.engine_path = retrieveSdkDir()
            if parsed_args.engine_path is None:
                raise ParameterError(
                    "Please pass valid QAIRT/SNPE SDK path with --engine_path argument.")

        # Find SDK type(QAIRT/QNN/SNPE)
        parsed_args.sdk_type = get_sdk_type(parsed_args.engine_path)

        if parsed_args.sdk_type not in [Engine.QAIRT.value, Engine.SNPE.value]:
            raise ParameterError(
                f"Invalid SDK type={parsed_args.sdk_type} is used or passed, supported SDKs are {[Engine.QAIRT.value, Engine.SNPE.value]}"
            )

        if (parsed_args.architecture in [a.value for a in Android_Architectures]):
            target_device = Device_type.android.value
        elif (parsed_args.architecture in [a.value for a in X86_Architectures]):
            target_device = Device_type.x86.value
        elif (parsed_args.architecture in [a.value for a in X86_windows_Architectures]):
            target_device = Device_type.x86_64_windows_msvc.value
        elif (parsed_args.architecture in [a.value for a in Aarch64_windows_Architectures]):
            target_device = Device_type.wos.value
        elif (parsed_args.architecture in [a.value for a in Qnx_Architectures]):
            target_device = Device_type.qnx.value
        elif (parsed_args.architecture in [a.value for a in Windows_Architectures]):
            target_device = Device_type.wos_remote.value
        else:
            raise ParameterError("Invalid architecture.")
        parsed_args.target_device = target_device

        if parsed_args.executor_type is None:
            # Set executor type since user didn't provide
            if parsed_args.sdk_type in [Engine.QNN.value, Engine.QAIRT.value]:
                parsed_args.executor_type = Engine.QNN.value
            elif parsed_args.sdk_type == Engine.SNPE.value:
                parsed_args.executor_type = Engine.SNPE.value
        else:
            parsed_args.executor_type = parsed_args.executor_type.upper()
            # Validate user provided executor type
            if parsed_args.executor_type not in [Engine.QNN.value, Engine.SNPE.value]:
                raise ParameterError(
                    f"Invalid option {parsed_args.executor_type} provided for --executor_type argument"
                )
            elif parsed_args.executor_type == Engine.SNPE.value \
                and parsed_args.sdk_type not in [Engine.SNPE.value, Engine.QAIRT.value]:
                raise ParameterError(
                    f"Invalid option {parsed_args.executor_type} provided for --executor_type argument"
                )
            elif parsed_args.executor_type == Engine.QNN.value \
                and parsed_args.sdk_type not in [Engine.QAIRT.value]:
                raise ParameterError(
                    f"Invalid option {parsed_args.executor_type} provided for --executor_type argument"
                )

        # set default dsp version if 'dsp' is selected as runtime (NOTE: exclude dspv79 since it is not yet officially announced)
        # TODO Default to dspv79 once it is officially announced
        if parsed_args.runtime == "dsp":
            dspArchs = [
                r.value for r in Runtime if r.value.startswith("dsp")
                and r.value not in ["dsp", "dspv79"]
            ]
            parsed_args.runtime = max(dspArchs)

        valid_runtimes = [r.value for r in Runtime if r.value not in ['htp']]
        if parsed_args.executor_type == Engine.SNPE.value:
            valid_runtimes.remove(Runtime.aic.value)

        if parsed_args.runtime not in valid_runtimes:
            raise ParameterError(
                f"Given {parsed_args.runtime} runtime is not supported. Runtime should be one of the following: {valid_runtimes}"
            )

        if 'dsp' in parsed_args.runtime and parsed_args.architecture == 'x86_64-linux-clang':
            raise ParameterError(
                f"{parsed_args.runtime} runtime is unsupported for {parsed_args.architecture} architecture."
            )

        if parsed_args.architecture == Qnx_Architectures.aarch64_qnx.value and 'dsp' not in parsed_args.runtime:
            raise ParameterError(
                f"{parsed_args.runtime} runtime is unsupported for {parsed_args.architecture} architecture."
            )

        # verify that runtime and architecture align
        if parsed_args.architecture == X86_windows_Architectures.x86_64_windows_msvc.value\
                                                    and parsed_args.runtime not in ["cpu"]:
            raise ParameterError(
                f"{parsed_args.runtime} runtime is unsupported for {X86_windows_Architectures.x86_64_windows_msvc.value} architecture."
            )

        aarch64_windows_supported_runtimes = [
            runtime for runtime in valid_runtimes if runtime == 'cpu' or 'dsp' in runtime
        ]
        if parsed_args.architecture == Aarch64_windows_Architectures.wos.value\
                                                    and parsed_args.runtime not in aarch64_windows_supported_runtimes:
            raise ParameterError(
                f"{parsed_args.runtime} runtime is unsupported for {Aarch64_windows_Architectures.wos.value} architecture."
            )

        if parsed_args.runtime == Runtime.gpu.value and \
                parsed_args.architecture != 'aarch64-android':
            raise ParameterError(
                f"GPU runtime is supported only for aarch64-android architecture but passed {parsed_args.architecture} architecture"
            )

        if parsed_args.runtime == Runtime.aic.value and \
                parsed_args.architecture != 'x86_64-linux-clang':
            raise ParameterError(
                f"AIC runtime is supported only for x86_64-linux-clang architecture but passed {parsed_args.architecture} architecture"
            )

        if parsed_args.executor_type == Engine.SNPE.value and parsed_args.architecture == 'aarch64-qnx':
            raise ParameterError(
                f"{Qnx_Architectures.aarch64_qnx.value} architecture is not supported when using SNPE execution"
            )

        if parsed_args.stage == 'source':
            # Validating Source stage related Arguments
            source_attr = ["input_network"]
            for attr in source_attr:
                if getattr(parsed_args, attr) is None:
                    raise ParameterError(
                        f"Missing --{attr} argument, following arguments are required for source stage {source_attr}"
                    )
        elif parsed_args.stage == 'converted' or parsed_args.stage == 'quantized':
            # Validating Converted/Quantized stage related Arguments
            required_attr = ["input_dlc"]
            for attr in required_attr:
                if getattr(parsed_args, attr) is None:
                    raise ParameterError(
                        f"Missing --{attr} argument, following arguments are required for converted/quantized stages {required_attr}"
                    )
        else:
            raise ParameterError(f"Invalid value '{parsed_args.stage}' passed for --stage argument")

        if parsed_args.desired_input_shape is not None:
            # get proper desired_input_shape format
            tensor_list = []
            for tensor in parsed_args.desired_input_shape:
                if len(tensor) < 2:
                    raise argparse.ArgumentTypeError(
                        "Invalid format for desired_input_shape, format as "
                        "--desired_input_shape \"INPUT_NAME\" INPUT_DIM.")
                # collect first two args and discard rest of the args since they won't be used
                tensor_list.append(tensor[:2])
            parsed_args.desired_input_shape = tensor_list

        # Find framework type
        parsed_args.framework = get_framework_info(parsed_args.input_network)

        # Remove datatype info in input/output names for tf and tflite models
        if parsed_args.framework and parsed_args.framework != 'onnx':
            if parsed_args.desired_input_shape is not None:
                tensor_list = []
                for tensor in parsed_args.desired_input_shape:
                    if ":" in tensor[0]:
                        tensor[0] = tensor[0].split(":")[0]
                    tensor_list.append(tensor)
                parsed_args.desired_input_shape = tensor_list

            if parsed_args.out_tensor_node is not None:
                tensor_list = []
                for tensor in parsed_args.out_tensor_node:
                    if ":" in tensor:
                        tensor = tensor.split(":")[0]
                    tensor_list.append(tensor)
                parsed_args.out_tensor_node = tensor_list

        if parsed_args.extra_converter_args:
            converter_ignore_list = [
                'input_network', 'desired_input_shape', 'out_tensor_node', 'io_config',
                'quantization_overrides'
            ]
            parsed_args.extra_converter_args = format_args(parsed_args.extra_converter_args,
                                                           converter_ignore_list)

        if parsed_args.extra_quantizer_args:
            quantizer_ignore_list = [
                'input_dlc', 'bias_bitwidth', 'act_bitwidth', 'weights_bitwidth',
                'use_native_input_files', 'use_native_output_files', 'act_quantizer_calibration',
                'param_quantizer_calibration', 'act_quantizer_schema', 'param_quantizer_schema',
                'percentile_calibration_value', 'use_per_channel_quantization',
                'use_per_row_quantization', 'float_bitwidth', 'float_fallback'
            ]
            parsed_args.extra_quantizer_args = format_args(parsed_args.extra_quantizer_args,
                                                           quantizer_ignore_list)

        if parsed_args.extra_runtime_args:
            runtime_ignore_list = ['perf_profile', 'profiling_level', 'userlogs', 'log_level']
            parsed_args.extra_runtime_args = format_args(parsed_args.extra_runtime_args,
                                                         runtime_ignore_list)

        if parsed_args.act_quantizer_calibration != 'percentile' and parsed_args.param_quantizer_calibration != 'percentile':
            parsed_args.percentile_calibration_value = None

        if parsed_args.extra_converter_args and '--float_bitwidth' in parsed_args.extra_converter_args:
            raise ParameterError(
                "float_bitwidth is not allowed as part of --extra_converter_args, instead use --float_bitwidth quantizer arg"
            )

        if parsed_args.float_bitwidth:
            float_bitwidth_options = [32, 16]
            if parsed_args.float_bitwidth not in float_bitwidth_options:
                raise ParameterError(
                    f"Allowed options for --float_bitwidth are {float_bitwidth_options}")

        if parsed_args.precision == 'fp16':
            if parsed_args.float_bitwidth is None:
                parsed_args.float_bitwidth = 16
            elif parsed_args.float_bitwidth != 16:
                raise ParameterError(
                    "Please correct your --float_bitwidth, with --precision set as fp16, --float_bitwidth must be 16"
                )

            if parsed_args.calibration_input_list:
                raise ParameterError(
                    f"--calibration_input_list argument cannot be used when --precision is {parsed_args.precision}"
                )
        elif parsed_args.precision == 'int8':
            if parsed_args.calibration_input_list is None and parsed_args.quantization_overrides is None:
                raise ParameterError(
                    f"Either --calibration_input_list or --quantization_overrides argument is required when --precision is {parsed_args.precision}"
                )
            elif parsed_args.calibration_input_list and parsed_args.float_fallback:
                raise ParameterError(
                    "Cannot use --calibration_input_list and --float_fallback arguments together"
                )

            if parsed_args.float_bitwidth:
                raise ParameterError(
                    f"--float_bitwidth argument cannot be used when --precision is {parsed_args.precision}"
                )

        parsed_args.model_path = parsed_args.input_network

        if parsed_args.add_layer_outputs:
            parsed_args.add_layer_outputs = parsed_args.add_layer_outputs.split(',')

        if parsed_args.skip_layer_outputs:
            parsed_args.skip_layer_outputs = parsed_args.skip_layer_outputs.split(',')

        if parsed_args.add_layer_types:
            parsed_args.add_layer_types = parsed_args.add_layer_types.split(',')

        if parsed_args.skip_layer_types:
            parsed_args.skip_layer_types = parsed_args.skip_layer_types.split(',')

        if parsed_args.add_layer_types or parsed_args.skip_layer_types or \
            parsed_args.skip_layer_outputs or parsed_args.start_layer or parsed_args.end_layer:
            # update_layer_options() parses add/skip layer types/outputs
            # along with start and end layer options and returns final list of nodes to be dumped
            parsed_args.add_layer_outputs = update_layer_options(parsed_args)

        # By default offline prepare will be True for dsp runtimes,
        # unless user disables it using --disable_offline_prepare flag
        parsed_args.offline_prepare = True
        if parsed_args.disable_offline_prepare or 'dsp' not in parsed_args.runtime or \
                parsed_args.architecture == 'wos-remote':
            parsed_args.offline_prepare = False

        if parsed_args.extra_contextbin_args and parsed_args.executor_type != Engine.QNN.value:
            raise ParameterError(
                f"--extra_contextbin_args can be used only with --executor_type qnn combination.")
        elif parsed_args.extra_contextbin_args:
            context_binary_ignore_list = [
                'model', 'backend', 'binary_file', 'model_prefix', 'output_dir', 'config_file',
                'enable_intermediate_outputs', 'set_output_tensors'
            ]
            parsed_args.extra_contextbin_args = format_args(parsed_args.extra_contextbin_args,
                                                            context_binary_ignore_list)

        if parsed_args.backend_extension_config or \
                        parsed_args.context_config_params or parsed_args.graph_config_params:
            if ('dsp' not in parsed_args.runtime and parsed_args.runtime not in \
                    [Runtime.gpu.value, Runtime.aic.value]):
                raise ParameterError(
                    "--backend_extension_config/--context_config_params/--graph_config_params \
                        allowed only for dsp, aic and gpu runtimes")

        if parsed_args.backend_extension_config and parsed_args.executor_type != Engine.QNN.value:
            raise ParameterError(
                f"--backend_extension_config can be used only with --executor_type qnn combination."
            )
        elif parsed_args.backend_extension_config:
            parsed_args.backend_extension_config = get_absolute_path(
                parsed_args.backend_extension_config)

        if parsed_args.context_config_params and parsed_args.executor_type != Engine.QNN.value:
            raise ParameterError(
                f"--context_config_params can be used only with --executor_type qnn combination.")
        elif parsed_args.context_config_params:
            parsed_args.context_config_params = format_params(parsed_args.context_config_params)

        if parsed_args.graph_config_params and parsed_args.executor_type != Engine.QNN.value:
            raise ParameterError(
                f"--graph_config_params can be used only with --executor_type qnn combination.")
        elif parsed_args.graph_config_params:
            parsed_args.graph_config_params = format_params(parsed_args.graph_config_params)

        if parsed_args.perf_profile:
            perf_profile_options = [
                'low_balanced', 'balanced', 'default', 'high_performance',
                'sustained_high_performance', 'burst', 'low_power_saver', 'power_saver',
                'high_power_saver', 'extreme_power_saver', 'system_settings'
            ]
            if parsed_args.perf_profile not in perf_profile_options:
                raise ParameterError(
                    f"Allowed options for --perf_profile are {perf_profile_options}")

        if parsed_args.profiling_level:
            qnn_profiling_level_options = ["basic", "detailed", "client"]
            snpe_profiling_level_options = ["off", "basic", "moderate", "detailed", "linting"]
            if parsed_args.executor_type == Engine.QNN.value and parsed_args.profiling_level not in qnn_profiling_level_options:
                raise ParameterError(
                    f"When --executor_type is qnn, allowed options for --profiling_level are {qnn_profiling_level_options}"
                )
            elif parsed_args.executor_type == Engine.SNPE.value and parsed_args.profiling_level not in snpe_profiling_level_options:
                raise ParameterError(
                    f"When --executor_type is snpe, allowed options for --profiling_level are {snpe_profiling_level_options}"
                )

        if parsed_args.log_level and parsed_args.executor_type != Engine.QNN.value:
            raise ParameterError(
                f"--log_level can be used only with --executor_type qnn combination.")

        if parsed_args.userlogs and parsed_args.executor_type != Engine.SNPE.value:
            raise ParameterError(
                f"--userlogs can be used only with --executor_type snpe combination.")

        if parsed_args.host_device:
            host_device_options = ['x86', 'x86_64-windows-msvc', 'wos']
            if parsed_args.host_device not in host_device_options:
                raise ParameterError(f"Allowed options for --host_device are {host_device_options}")

        if parsed_args.calibration_input_list and parsed_args.precision in ['fp16', 'fp32']:
            raise ParameterError("When calibration_input_list is provided, precision can not be {}".format(parsed_args.precision))

        return parsed_args

    def get_all_associated_parsers(self):
        return [self.parser]
