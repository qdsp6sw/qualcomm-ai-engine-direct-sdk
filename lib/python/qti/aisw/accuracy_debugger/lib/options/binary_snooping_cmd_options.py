# =============================================================================
#
#  Copyright (c) 2023-2024 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

from qti.aisw.accuracy_debugger.lib.utils.nd_path_utility import get_absolute_path, format_args
from qti.aisw.accuracy_debugger.lib.options.cmd_options import CmdOptions
from qti.aisw.accuracy_debugger.lib.utils.nd_exceptions import ParameterError
from qti.aisw.accuracy_debugger.lib.utils.nd_constants import Framework, Engine, Runtime
from qti.aisw.accuracy_debugger.lib.utils.nd_exceptions import InferenceEngineError
from qti.aisw.accuracy_debugger.lib.utils.nd_constants import Architecture_Target_Types, Engine, Runtime, \
    Android_Architectures, X86_Architectures, \
    Device_type, Qnx_Architectures, Windows_Architectures, X86_windows_Architectures, Aarch64_windows_Architectures
from qti.aisw.accuracy_debugger.lib.utils.nd_errors import get_message

import argparse
import os
import numpy as np


class BinarySnoopingCmdOptions(CmdOptions):

    def __init__(self, args, engine, validate_args=True):
        super().__init__('binary_snooping', args, engine, validate_args=validate_args)

    def _get_engine(self, engine):
        if engine == Engine.SNPE.value:
            return True, False, False
        elif engine == Engine.QNN.value:
            return False, True, False
        elif engine == Engine.ANN.value:
            return False, False, True
        else:
            raise InferenceEngineError(
                get_message("ERROR_INFERENCE_ENGINE_ENGINE_NOT_FOUND")(engine))

    def initialize(self):
        """
        type: (List[str]) -> argparse.Namespace

        :param args: User inputs, fed in as a list of strings
        :return: Namespace object
        """
        self.parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                              description="Script that runs quant_checker")

        required = self.parser.add_argument_group('required arguments')

        required.add_argument('-m', '--model_path', type=str, required=True,
                              help='Path to the model file(s).')
        required.add_argument('-l', '--input_list', type=str, required=True,
                              help="Path to the input list text.")

        required.add_argument(
            '-i', '--input_tensor', nargs="+", action='append', required=True,
            help='The name, dimensions, raw data, and optionally data type of the '
            'network input tensor(s) specified'
            'in the format "input_name" comma-separated-dimensions '
            'path-to-raw-file, '
            'for example: "data" 1,224,224,3 data.raw float32. Note that the '
            'quotes should always be included in order to handle special '
            'characters, spaces, etc. For multiple inputs specify multiple '
            '--input_tensor on the command line like: --input_tensor "data1" '
            '1,224,224,3 data1.raw --input_tensor "data2" 1,50,100,3 data2.raw '
            'float32.')
        required.add_argument(
            '-f', '--framework', nargs='+', type=str.lower, required=True,
            help='Framework type and version, version is optional. '
            'Currently supported frameworks are [' + ', '.join([f.value for f in Framework]) + ']. '
            'For example, tensorflow 2.3.0')
        required.add_argument('-o', '--output_tensor', type=str, required=True, action='append',
                              help='Name of the graph\'s output tensor(s).')
        required.add_argument('--verifier', type=str.lower, required=False, default='mse',
                              help='Choose verifer among [sqnr, mse] for the comparison')
        required.add_argument(
            '-e', '--engine', nargs='+', type=str, required=True,
            metavar=('ENGINE_NAME', 'ENGINE_VERSION'),
            help='Name of engine that will be running inference, '
            'optionally followed by the engine version.')
        required.add_argument('-qo', '--quantization_overrides', type=str, required=True,
                              help="Path to quantization overrides json file.")

        optional = self.parser.add_argument_group('optional arguments')
        optional.add_argument('-a', '--architecture', type=str.lower, default='x86_64-linux-clang',
                              choices=Architecture_Target_Types.target_types.value,
                              help='Name of the architecture to use for inference engine.')
        optional.add_argument(
            '--host_device', type=str, required=False, default='x86',
            choices=['x86', 'x86_64-windows-msvc', 'wos'],
            help='The device that will be running conversion. Set to x86 by default.')
        optional.add_argument(
            '-r', '--runtime', type=str.lower, default=Runtime.dspv68.value,
            choices=[r.value for r in Runtime], help='Runtime to be used.Please '
            'use htp runtime for emulation on x86 host')
        optional.add_argument(
            '--deviceId', required=False, default=None,
            help='The serial number of the device to use. If not available, '
            'the first in a list of queried devices will be used for validation.')
        optional.add_argument(
            '--golden_output_reference_directory', dest='golden_output_reference_directory',
            type=str, required=False,
            help='Optional parameter to indicate the directory of the golden reference. '
            'When this option provided, framework diagnosis stage is skipped. '
            'Provide the path to directory which contains the fp32 outputs for '
            'all inputs mentioned in the input_list.txt. ')
        optional.add_argument('--bias_bitwidth', type=str, required=False, default=8,
                              help='Bit-width to use for biases. E.g., 8, 32. Default is 8.')
        optional.add_argument('--use_per_channel_quantization', type=bool, required=False,
                              default=False, help='performs per channel quantization')
        optional.add_argument('--weights_bitwidth', type=str, required=False, default=8,
                              help='Bit-width to use for weights. E.g., 8. Default is 8.')
        optional.add_argument(
            '--act_bitwidth', type=int, required=False, default=8, choices=[8, 16],
            help="option to select the bitwidth to use when quantizing the activations. default 8")
        optional.add_argument(
            '-fbw', '--float_bias_bitwidth', type=int, required=False, default=32, choices=[16, 32],
            help="option to select the bitwidth to use when biases are in float. default 32")
        optional.add_argument(
            '-rqs', '--restrict_quantization_steps', type=str, required=False,
            help="ENCODING_MIN, ENCODING_MAX \
                    Specifies the number of steps to use for computing quantization encodings \
                    such that scale = (max - min) / number of quantization steps.\
                    The option should be passed as a space separated pair of hexadecimal string \
                    minimum and maximum values. i.e. --restrict_quantization_steps 'MIN MAX'. \
                    Please note that this is a hexadecimal string literal and not a signed \
                    integer, to supply a negative value an explicit minus sign is required. \
                    E.g.--restrict_quantization_steps '-0x80 0x7F' indicates an example 8 bit \
                    range, --restrict_quantization_steps '-0x8000 0x7F7F' indicates an example 16 \
                    bit range.")
        optional.add_argument('-v', '--verbose', action="store_true", default=False,
                              help="Verbose printing")
        optional.add_argument('-w', '--working_dir', type=str, required=False,
                            default='working_directory',
                            help='Working directory for the {} to store temporary files. '.format(self.component) + \
                                'Creates a new directory if the specified working directory does not exitst.')
        optional.add_argument('--output_dirname', type=str, required=False,
                                default='<curr_date_time>',
                                help='output directory name for the {} to store temporary files under <working_dir>/{}. '.format(self.component, self.component) + \
                                    'Creates a new directory if the specified working directory does not exist')
        optional.add_argument('-p', '--engine_path', type=str, required=False,
                              help="Path to the inference engine.")
        optional.add_argument('--min_graph_size', default=16, type=int, required=False,
                              help='Provide the minimum subgraph size')
        optional.add_argument(
            '--extra_converter_args', type=str, required=False, default=None,
            help="additional converter arguments in a quoted string. \
                 example: --extra_converter_args 'input_dtype=data float;input_layout=data1 NCHW'")
        optional.add_argument(
            '--act_quantizer_calibration', type=str.lower, required=False, default=None,
            choices=['min-max', 'sqnr', 'entropy', 'mse', 'percentile'],
            help="Specify which quantization calibration method to use for activations. \
                                    This option has to be paired with --act_quantizer_schema.")

        optional.add_argument(
            '--param_quantizer_calibration', type=str.lower, required=False, default=None,
            choices=['min-max', 'sqnr', 'entropy', 'mse', 'percentile'],
            help="Specify which quantization calibration method to use for parameters.\
                                    This option has to be paired with --param_quantizer_schema.")

        optional.add_argument(
            '--act_quantizer_schema', type=str.lower, required=False, default='asymmetric',
            choices=['asymmetric', 'symmetric', 'unsignedsymmetric'],
            help="Specify which quantization schema to use for activations. \
                                    Can not be used together with act_quantizer.")

        optional.add_argument(
            '--param_quantizer_schema', type=str.lower, required=False, default='asymmetric',
            choices=['asymmetric', 'symmetric', 'unsignedsymmetric'],
            help="Specify which quantization schema to use for parameters. \
                                    Can not be used together with param_quantizer.")

        optional.add_argument('--percentile_calibration_value', type=float, required=False,
                              default=99.99, help="Value must lie between 90-100")
        optional.add_argument('--param_quantizer', type=str.lower, required=False, default='tf',
                              choices=['tf', 'enhanced', 'adjusted',
                                       'symmetric'], help="Param quantizer algorithm used.")
        optional.add_argument('--act_quantizer', type=str, required=False, default='tf',
                              choices=['tf', 'enhanced', 'adjusted', 'symmetric'],
                              help="Optional parameter to indicate the activation quantizer to use")
        optional.add_argument('--per_channel_quantization', action="store_true", default=False,
                              help="Use per-channel quantization for convolution-based op weights.")
        optional.add_argument(
            '--algorithms', type=str, required=False, default=None,
            help="Use this option to enable new optimization algorithms. Usage is: \
                --algorithms <algo_name1> ... \
                The available optimization algorithms are: 'cle ' - Cross layer equalization \
                includes a number of methods for equalizing weights and biases across layers \
                in order to rectify imbalances that cause quantization errors.")
        optional.add_argument('--verifier_config', type=str, default=None,
                              help='Path to the verifiers\' config file')
        optional.add_argument(
            '--start_layer', type=str, default=None, required=False,
            help="Extracts the given model from mentioned start layer output name")
        optional.add_argument('--end_layer', type=str, default=None, required=False,
                              help="Extracts the given model from mentioned end layer output name")
        optional.add_argument('--precision', choices=['int8', 'fp16'], default='int8',
                              help='select precision')
        optional.add_argument('--compiler_config', type=str, default=None, required=False,
                              help="Path to the compiler config file.")

        optional.add_argument(
            '--ignore_encodings', action="store_true", default=False, help=
            "Use only quantizer generated encodings, ignoring any user or model provided encodings."
        )
        optional.add_argument(
            '--extra_runtime_args', type=str, required=False, default=None,
            help="additional convereter arguments in a quoted string. \
                                        example: --extra_runtime_args profiling_level=basic;log_level=debug"
        )
        optional.add_argument(
            '--add_layer_outputs', default=[], help="Output layers to be dumped. \
                                    example:1579,232")
        optional.add_argument(
            '--add_layer_types', default=[],
            help='outputs of layer types to be dumped. e.g :Resize,Transpose.\
                                All enabled by default.')
        optional.add_argument(
            '--skip_layer_types', default=[],
            help='comma delimited layer types to skip snooping. e.g :Resize, Transpose')
        optional.add_argument(
            '--skip_layer_outputs', default=[],
            help='comma delimited layer output names to skip debugging. e.g :1171, 1174')
        optional.add_argument('--remote_server', type=str, required=False, default=None,
                              help="ip address of remote machine")
        optional.add_argument('--remote_username', type=str, required=False, default=None,
                              help="username of remote machine")
        optional.add_argument('--remote_password', type=str, required=False, default=None,
                              help="password of remote machine")
        optional.add_argument(
            '-nif', '--use_native_input_files', action="store_true", default=False, required=False,
            help="Specifies that the input files will be parsed in the data type native to the graph.\
                                    If not specified, input files will be parsed in floating point."
        )
        optional.add_argument(
            '-nof', '--use_native_output_files', action="store_true", default=False, required=False,
            help="Specifies that the output files will be generated in the data \
                                    type native to the graph. If not specified, output files will \
                                    be generated in floating point.")
        self.initialized = True

    def verify_update_parsed_args(self, parsed_args):
        parsed_args.model_path = get_absolute_path(parsed_args.model_path)
        parsed_args.input_list = get_absolute_path(parsed_args.input_list)
        parsed_args.quantization_overrides = get_absolute_path(parsed_args.quantization_overrides)
        if parsed_args.golden_output_reference_directory:
            parsed_args.golden_output_reference_directory = get_absolute_path(
                parsed_args.golden_output_reference_directory)
        # Parse version since it is an optional argument that is combined with framework
        parsed_args.framework_version = None
        if len(parsed_args.framework) == 2:
            parsed_args.framework_version = parsed_args.framework[1]
        parsed_args.framework = parsed_args.framework[0]

        parsed_args.engine = self.engine

        # verify that target_device and architecture align
        if hasattr(parsed_args, 'architecture'):
            arch = parsed_args.architecture
            if (arch in [a.value for a in Android_Architectures]):
                target_device = Device_type.android.value
            elif (arch in [a.value for a in X86_Architectures]):
                target_device = Device_type.x86.value
            elif (arch in [a.value for a in X86_windows_Architectures]):
                target_device = Device_type.x86_64_windows_msvc.value
            elif (arch in [a.value for a in Aarch64_windows_Architectures]):
                target_device = Device_type.wos.value
            elif (arch in [a.value for a in Qnx_Architectures]):
                target_device = Device_type.qnx.value
            elif (arch in [a.value for a in Windows_Architectures]):
                target_device = Device_type.wos_remote.value
            else:
                raise ParameterError("Invalid architecture.")
            parsed_args.target_device = target_device
        linux_target, android_target, x86_64_windows_msvc_target, wos_target = (parsed_args.target_device == 'x86'
                                        or parsed_args.target_device
                                        == 'linux_embedded'), parsed_args.target_device == 'android',\
            parsed_args.target_device == Device_type.x86_64_windows_msvc.value, parsed_args.target_device == Device_type.wos.value

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
                tensor[:] = tensor[:3]

        #The last data type gets shaved off
        if parsed_args.engine == Engine.QNN.value:
            if parsed_args.input_tensor is not None:
                tensor_list = []
                for tensor in parsed_args.input_tensor:
                    #this : check acts differently on snpe vs qnn on tensorflow models.
                    if ":" in tensor[0]:
                        tensor[0] = tensor[0].split(":")[0]
                    tensor_list.append(tensor)
                parsed_args.input_tensor = tensor_list

            if parsed_args.output_tensor is not None:
                tensor_list = []
                for tensor in parsed_args.output_tensor:
                    if ":" in tensor:
                        tensor = tensor.split(":")[0]
                    tensor_list.append(tensor)
                parsed_args.output_tensor = tensor_list

        # Since underlying QNN tools are not supporting 64 bit inputs but our framework
        # diagnosis works on 64 bit, if model accepts 64 bit inputs
        # we need to reconvert the user provided 64 bit inputs through the input_list
        # into 32 bits. Here, we loop over each line in the input_list.txt
        # and convert the 64 bit input and dump them into the working directory
        # and finally we create new input_list containing 32 bit input paths
        converted_input_file_dump_path = os.path.join(parsed_args.working_dir, "inputs_32")
        os.makedirs(converted_input_file_dump_path, exist_ok=True)
        new_input_list_file_path = os.path.join(converted_input_file_dump_path, 'input_list.txt')
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
                            new_file_name_and_path.append(file_name_and_path[0] + ":=" + file_name)
                        else:
                            new_file_name_and_path.append(file_name)
                    new_input_list_file.write(" ".join(new_file_name_and_path) + "\n")
        new_input_list_file.close()
        parsed_args.qnn_input_list = new_input_list_file_path

        return parsed_args

    def get_all_associated_parsers(self):
        parsers_to_be_validated = [self.parser]
        option_classes = []
        from qti.aisw.accuracy_debugger.lib.options.framework_diagnosis_cmd_options import FrameworkDiagnosisCmdOptions
        option_classes.append(FrameworkDiagnosisCmdOptions(self.args, False))

        from qti.aisw.accuracy_debugger.lib.options.inference_engine_cmd_options import InferenceEngineCmdOptions

        option_classes.append(InferenceEngineCmdOptions(self.engine, self.args, False))

        for option_class in option_classes:
            option_class.initialize()
            parsers_to_be_validated.extend(option_class.get_all_associated_parsers())
        self._validate_quantizer_args()
        return parsers_to_be_validated

    def _validate_quantizer_args(self):
        # Make sure (--activation_quantizer, --param_quantizer) and
        # (--act_quantizer_calibration, --param_quantizer_calibration) not in
        # args passed by the user
        if ("--param_quantizer_calibration" in self.args or "--act_quantizer_calibration"
                in self.args) and ("--act_quantizer" in self.args
                                   or "--param_quantizer" in self.args or "-pq" in self.args):
            raise ParameterError(
                "Invalid combination: legacy quantizer options: --act_quantizer or --param_quantizer cannot be combined with --act_quantizer_calibration or --param_quantizer_calibration"
            )
