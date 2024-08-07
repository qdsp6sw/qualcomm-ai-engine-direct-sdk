# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
import os
from argparse import Namespace
import yaml
import ast
from typing import Union
from qti.aisw.converters.common.utils.converter_utils import log_error, log_warning
from qti.aisw.converters.qnn_backend.custom_ops.op_factory import QnnCustomOpFactory
from qti.aisw.tools.core.modules.converter.constants import *
from qti.aisw.converters.common.model_validator import Validator


def get_framework(extension: str) -> str:
    if OnnxFrameworkInfo.check_framework(extension):
        return OnnxFrameworkInfo.name
    elif PytorchFrameworkInfo.check_framework(extension):
        return PytorchFrameworkInfo.name
    elif TensorflowFrameworkInfo.check_framework(extension):
        return TensorflowFrameworkInfo.name
    elif TFLiteFrameworkInfo.check_framework(extension):
        return TFLiteFrameworkInfo.name
    else:
        raise Exception("Invalid model format specified. Supported types are {}".format(SUPPORTED_EXTENSIONS))


def get_frontend_converter(framework: str, args: dict, validator: Validator):
    if framework == OnnxFrameworkInfo.name:
        if not args.use_onnx_relay:
            from qti.aisw.converters.onnx.onnx_to_ir import OnnxConverterFrontend
            return OnnxConverterFrontend(args, custom_op_factory=QnnCustomOpFactory(), validator=validator)
        else:
            try:
                # use onnx-relay-converter flow
                from qti.aisw.converters.onnx.onnx_to_ir_relay import OnnxRelayConverterFrontend
                return OnnxRelayConverterFrontend(args, custom_op_factory=QnnCustomOpFactory())
            except Exception as e:
                raise Exception(
                    "'use_onnx_relay' is not available. Please remove use_onnx_relay in converter input args.")
    elif framework == TensorflowFrameworkInfo.name:
        from qti.aisw.converters.tensorflow.tf_to_ir import TFConverterFrontend
        if not args.input_dim or not args.out_names:
            raise Exception("'desired_input_shape' and 'out_tensor_node' are required for TensorFlow conversion")
        return TFConverterFrontend(args, custom_op_factory=QnnCustomOpFactory(), validator=validator)
    elif framework == TFLiteFrameworkInfo.name:
        from qti.aisw.converters.tflite.tflite_to_ir import TFLiteConverterFrontend
        return TFLiteConverterFrontend(args, custom_op_factory=QnnCustomOpFactory())
    else:
        from qti.aisw.converters.pytorch.pytorch_to_ir import PyTorchConverterFrontend
        from qti.aisw.converters.relay.custom_ops.utils.pytorch_helpers import PytorchCustomOpFactory
        if not args.input_dim:
            raise Exception("'desired_input_shape' is required for PyTorch conversion")
        return PyTorchConverterFrontend(args, custom_op_factory=PytorchCustomOpFactory())


def infer_framework(input_network: str) -> str:

    model_path, model_ext = os.path.splitext(input_network)

    # tensorflow2 takes as input a folder which would have the ".pb" file

    if model_ext not in SUPPORTED_EXTENSIONS:
        model_files = os.listdir(model_path)
        for file in model_files:
            file_ext = os.path.splitext(file)[1]
            if file_ext == '.pb':
                model_ext = '.pb'

    if model_ext not in SUPPORTED_EXTENSIONS:
        raise Exception("Invalid model format specified. Supported types are {}".format(SUPPORTED_EXTENSIONS))
    framework = get_framework(model_ext)
    return framework


def get_num_tensor_configs(tensor_configs: Union[int, str]) -> int:
    # for when there is only one tensor config e.g. tensor_configs == (1,2,3)
    if isinstance(tensor_configs[0], int):
        return 1

    # for when input_dims is passed individually via CLI
    elif isinstance(tensor_configs, str):
        return 1

    # for when there is multiple tensor configs e.g. tensor_configs == ((1,2,3), (4,5,6))
    else:
        return len(tensor_configs)


def get_graph_configs(args: dict) -> list:
    def convert_dimensions_to_string(dims):
        return ",".join([str(dim) for dim in dims])

    num_configurations = get_num_graph_configs(args)
    configurations = []

    for i in range(num_configurations):
        configuration = []
        for tensor_name, tensor_configs in args.input_dim:
            if get_num_tensor_configs(tensor_configs) > 1:
                tensor_dims = convert_dimensions_to_string(tensor_configs[i])
            else:
                tensor_dims = convert_dimensions_to_string(tensor_configs)
            configuration.append([tensor_name, tensor_dims])
        configurations.append(configuration)
    return configurations


def get_num_graph_configs(args: Namespace) -> int:
    def validate_num_configs_is_1_or_n(num_tensor_configs_seen):
        error_message = "Error: Number of tensor configurations can either be 1 or N. \
                       You specified the following number of tensor configurations: {}" \
            .format(num_tensor_configs_seen)
        if len(num_tensor_configs_seen) > 2:
            log_error(error_message)
        elif len(num_tensor_configs_seen) == 2:
            if 1 not in num_tensor_configs_seen:
                log_error(error_message)

    if args.input_dim is None:
        return 1

    num_tensor_configs_seen = set()
    for tensor_name, tensor_configs in args.input_dim:
        num_tensor_configs = get_num_tensor_configs(tensor_configs)
        num_tensor_configs_seen.add(num_tensor_configs)
    validate_num_configs_is_1_or_n(num_tensor_configs_seen)
    return max(num_tensor_configs_seen)


def get_validator(framework: str, args: dict) -> Validator:
    validator = None
    if ((framework == OnnxFrameworkInfo.name or framework == TensorflowFrameworkInfo.name)
            and args.validate_models):
        if args.converter_op_package_lib:
            log_warning("Model is having custom ops skipping validation.")
            args.validate_models = False
        else:
            validator = Validator()
    return validator


def set_graph_configs(args, config):
    args.input_dim = config


def set_optimization_args(args: Namespace, framework: str) -> None:
    # TODO: Align optimizations for all frameworks
    if framework == OnnxFrameworkInfo.name:
        args.expand_gru_op_structure = True
        args.unroll_gru_time_steps = True
        args.expand_sparse_op_structure = True

    if framework == OnnxFrameworkInfo.name or framework == PytorchFrameworkInfo.name:
        args.perform_axes_to_spatial_first_order = True
        args.preprocess_roi_pool_inputs = True

    if framework == OnnxFrameworkInfo.name or framework == TensorflowFrameworkInfo.name:
        args.unroll_lstm_time_steps = True
        args.align_matmul_ranks = True
        args.handle_gather_negative_indices = True

    if framework == TensorflowFrameworkInfo.name or framework == PytorchFrameworkInfo.name:
        args.match_caffe_ssd_to_tf = True

    # Enable/Disable following optimizations for onnx, tf, pytorch
    if framework != TFLiteFrameworkInfo.name:
        args.squash_box_decoder = True
        args.adjust_nms_features_dims = True
        args.extract_color_transform = True
        args.inject_cast_for_gather = True
        args.force_prune_cast_ops = False


def convert_args_v2_to_v1(args: Namespace) -> Namespace:
    args_dict = vars(args)

    # input_dims is parsed as [['ip1', 'a,b,c,d'], ['ip1', 'd,e,f,g']]
    input_dims = None
    input_encoding = []
    input_layout = []
    input_dtype = []
    output_names = []
    user_custom_io = []
    # in case user provides multiple dimensions for an input, network specialization will be enabled (supported only
    # in onnx) and input_dims will be populated as [['ip1', ((a,b,c), (d,e,f))], ['ip2', ((a',b',c'), (d',e',f'))]]
    network_specialization = False

    if args.io_config:
        f = open(args.io_config)
        io_config_dict = yaml.safe_load(f)
        print("io con = {}\n".format(io_config_dict))
        input_layout_dict = {}
        output_layout_dict = {}

        if 'Input Tensor Configuration' in io_config_dict:
            print("INPUT TENS")
            for i in range(len(io_config_dict['Input Tensor Configuration'])):
                for key, val in io_config_dict['Input Tensor Configuration'][i].items():
                    print("key = {}, val = {}\n".format(key, val))
                    if key == 'Name':
                        if val:
                            name = str(val)
                    elif key == 'Src Model Parameters':
                        if 'DataType' in val and val['DataType']:
                            input_dtype.append([name, val['DataType']])
                        if 'Layout' in val and val['Layout']:
                            input_layout.append([name, val['Layout']])
                            input_layout_dict[name] = val['Layout']
                    elif key == 'Desired Model Parameters':
                        if 'Shape' in val and val['Shape']:
                            print("shape")
                            if input_dims is None:
                                input_dims = []

                            # for cases when user passes a shape with one dimension
                            # e.g. Shape: 1
                            if isinstance(val["Shape"], int):
                                val["Shape"] = "(" + str(val['Shape']) + ",)"

                            dim = ast.literal_eval(val['Shape'])

                            # for cases when user passes a shape with one dimension
                            # e.g. Shape: (1)
                            if isinstance(dim, int):
                                dim = (dim,)

                            if type(dim[0]) is tuple:
                                network_specialization = True
                            input_dims.append([name, dim])

                        custom_io_options = dict()
                        custom_io_options['IOName'] = name
                        if 'DataType' in val and val['DataType']:
                            custom_io_options['Datatype'] = val['DataType']
                        if 'Layout' in val and val['Layout']:
                            custom_io_options['Layout'] = dict()
                            custom_io_options['Layout']['Custom'] = val['Layout']
                            # Get the model layout corresponding to the custom layout for current input
                            if name in input_layout_dict:
                                custom_io_options['Layout']['Model'] = input_layout_dict[name]
                        # if any of the quant params are provided
                        if 'QuantParams' in val and (val['QuantParams']['Scale'] or val['QuantParams']['Offset']):
                            custom_io_options['QuantParam'] = val['QuantParams']
                            custom_io_options['QuantParam']['Type'] = 'QNN_DEFINITION_DEFINED'
                        if len(custom_io_options) > 1:
                            user_custom_io.append(custom_io_options)
                        if 'Color Conversion' in val and val['Color Conversion']:
                            input_encoding.append([name, val['Color Conversion']])

        if 'Output Tensor Configuration' in io_config_dict:
            for i in range(len(io_config_dict['Output Tensor Configuration'])):
                for key, val in io_config_dict['Output Tensor Configuration'][i].items():
                    if key == 'Name':
                        if val:
                            output_names.append(str(val))
                            name = str(val)
                    elif key == 'Src Model Parameters':
                        if 'Layout' in val and val['Layout']:
                            output_layout_dict[name] = val['Layout']
                    elif key == 'Desired Model Parameters':
                        custom_io_options = dict()
                        custom_io_options['IOName'] = name
                        if 'Layout' in val and val['Layout']:
                            custom_io_options['Layout'] = dict()
                            custom_io_options['Layout']['Custom'] = val['Layout']
                            # Get the model layout corresponding to the custom layout for current output
                            if name in output_layout_dict:
                                custom_io_options['Layout']['Model'] = output_layout_dict[name]
                        if 'DataType' in val and val['DataType']:
                            custom_io_options['Datatype'] = val['DataType']
                        # if any of the quant params are provided
                        if 'QuantParams' in val and (val['QuantParams']['Scale'] or val['QuantParams']['Offset']):
                            custom_io_options['QuantParam'] = val['QuantParams']
                            custom_io_options['QuantParam']['Type'] = 'QNN_DEFINITION_DEFINED'
                        if len(custom_io_options) > 1:
                            user_custom_io.append(custom_io_options)

    # update following args only if they were not provided on the commandline
    if not args_dict['input_dim']:
        # convert name:str, dim:tuple to name:str, dim:str if network specialization is disabled
        if input_dims and not network_specialization:
            for i in range(len(input_dims)):
                # convert tuple of dimension to comma separated string
                if type(input_dims[i][1]) is tuple:
                    input_dims[i][1] = ','.join(map(str, input_dims[i][1]))
                # remove whitespaces if any from string of dimension
                elif isinstance(input_dims[i][1], str):
                    input_dims[i][1] = input_dims[i][1].replace(" ", "")

        args_dict["input_dim"] = input_dims

    if not args_dict['input_layout']:
        args_dict['input_layout'] = input_layout
    if not args_dict['input_dtype']:
        args_dict['input_dtype'] = input_dtype
    if not args_dict['input_encoding']:
        args_dict['input_encoding'] = input_encoding

    # following arguments will be unused
    args_dict['input_type'] = []
    args_dict['dump_custom_io_config_template'] = ""

    if not args_dict['out_names']:
        args_dict['out_names'] = output_names

    args_dict['user_custom_io'] = user_custom_io

    # populate preserve_io_arg with [['layout']] to apply it to all inputs/outputs
    args_dict['preserve_io'] = [['layout']]
    if args.disable_preserve_io:
        args_dict['preserve_io'] = []

    return Namespace(**args_dict)
