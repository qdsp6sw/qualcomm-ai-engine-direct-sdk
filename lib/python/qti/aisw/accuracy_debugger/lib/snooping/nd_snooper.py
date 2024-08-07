# =============================================================================
#
#  Copyright (c) 2021 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
import logging
import os
import re
import sys

import numpy as np
import shutil
import signal
from collections import OrderedDict

from qti.aisw.accuracy_debugger.lib.utils.nd_logger import setup_logger
from qti.aisw.accuracy_debugger.lib.utils.nd_exceptions import SnoopingError
from qti.aisw.accuracy_debugger.lib.inference_engine.nd_execute_inference_engine import exec_inference_engine
from qti.aisw.accuracy_debugger.lib.utils.nd_exceptions import ConfigError, SnoopingError
from qti.aisw.accuracy_debugger.lib.utils.nd_namespace import Namespace
from qti.aisw.accuracy_debugger.lib.utils.nd_path_utility import santize_node_name
from qti.aisw.accuracy_debugger.lib.utils.nd_framework_utility import read_json, transpose_to_nhwc
from qti.aisw.accuracy_debugger.lib.snooping.snooper_utils import files_to_compare, LayerStatus
from qti.aisw.accuracy_debugger.lib.inference_engine.nd_get_tensor_mapping import TensorMapping
from qti.aisw.accuracy_debugger.lib.snooping.snooper_utils import SnooperUtils as su


def signal_handler(sig, _):
    Snooper.logger.info('Stopping snooping on user request.')
    if Snooper.stop:
        Snooper.logger.info('Waiting for current layer to complete.')
        Snooper.stop = True
    else:
        sys.exit(1)


signal.signal(signal.SIGINT, signal_handler)


class Snooper:
    stop = False
    logger = None

    def __init__(self, snooping_type, args, logger, verbose="info"):

        self.snooping_type = snooping_type
        self.args = args
        self.logger = logger if logger else setup_logger(verbose, args.output_dir)
        Snooper.logger = self.logger
        self.input_list_file = args.input_list
        self.model = args.model_path
        self.engine_path = args.engine_path
        self.deviceId = args.deviceId
        self.engine = args.engine
        self.framework = args.framework
        self.framework_version = None
        self.runtime = args.runtime
        self.framework_results = args.golden_output_reference_directory
        self.work_dir = args.working_dir
        self.output_dir = args.output_dir
        self.model_traverser = None
        self.model_handler = None
        self.target_device = args.target_device
        self.host_device = args.host_device
        self.architecture = args.architecture
        self.precision = args.precision
        self.compiler_config = args.compiler_config
        self.profile_info = None
        self.extra_converter_args = args.extra_converter_args
        self.extra_runtime_args = args.extra_runtime_args
        self.remote_server = args.remote_server
        self.remote_username = args.remote_username
        self.remote_password = args.remote_password
        self.act_quantizer = args.act_quantizer
        self.param_quantizer = args.param_quantizer
        self.bias_bitwidth = args.bias_bitwidth
        self.weights_bitwidth = args.weights_bitwidth
        self.act_bitwidth = args.act_bitwidth
        self.quantization_overrides = args.quantization_overrides
        self.algorithms = args.algorithms
        self.ignore_encodings = args.ignore_encodings
        self.per_channel_quantization = args.per_channel_quantization
        self.add_layer_outputs = args.add_layer_outputs
        self.add_layer_types = args.add_layer_types
        self.use_native_input_files = args.use_native_input_files
        self.use_native_output_files = args.use_native_output_files
        self.symbol_data_map = self.define_symbols()
        self.is_transpose_needed_dict = {}

    def define_symbols(self):
        """Populate symbol_data_map with mappings of onnx model symbols with
        corresponding data which is provided by user in extra_converter_args.
        Symbols are defined in the extra_converter_args as following examples:

        'define_symbol batch_size = 1'
        'define_symbol seq_len 128'

        """
        symbol_data_map = {}
        if self.extra_converter_args:
            converter_args = self.extra_converter_args.split(';')
            for arg in converter_args:
                arg = " ".join(arg.split())
                split_arg = re.split(' |=', arg)
                if split_arg[0] == "define_symbol":
                    if len(split_arg) == 3:
                        symbol, data = split_arg[1], split_arg[2]
                        symbol_data_map[symbol] = data
                    else:
                        raise ConfigError(
                            'Symbols are not defined correctly in extra_converter_args.')

        return symbol_data_map

    def get_input_tensors(self, list_file, model=None):
        input_tensors = []
        with open(list_file, 'r') as f:
            input_paths = f.readline().rstrip().split(' ')

        # get input_layers of extracted model by creating a new handler for it
        from qti.aisw.accuracy_debugger.lib.framework_diagnosis.nd_framework_runner import FrameworkRunner
        framework_args = Namespace(None, framework=self.framework, version=self.framework_version,
                                   model_path=model, output_dir=self.output_dir, engine=self.engine)
        model_handler = FrameworkRunner(self.logger, framework_args)
        model_handler.load_framework()
        input_layers = model_handler.framework_instance.get_input_layers()

        for i, item in enumerate(input_layers):
            if i >= len(input_paths):
                break
            dim_str = str(item[2])
            for symbol, data in self.symbol_data_map.items():
                dim_str = dim_str.replace(symbol, data)
            dim_str = dim_str.replace(' ', '')[1:-1]
            if ":=" in input_paths[i]:
                input_paths[i] = input_paths[i].split(":=")[1]
            input_data = (item[0], dim_str, input_paths[i])
            input_tensors.append(input_data)
        return input_tensors

    def update_list_file(self, graph_inputs):
        """Create a new input list file (temp-list.txt) based on the given
        input names."""
        updated_input_list = []
        handleInputNames = False
        # check is needed for caffe
        if isinstance(graph_inputs, dict):
            input_map = graph_inputs.copy()
            handleInputNames = True
            graph_inputs = list(graph_inputs.values())
            os.makedirs(self.work_dir + '/temp_inp/', exist_ok=True)

        for ip in graph_inputs:
            if ip in self.original_input_names_raw_map:
                updated_input_list.append(self.original_input_names_raw_map[ip].strip())
            else:
                s_ip = santize_node_name(ip)
                inp_path = os.path.join(self.framework_results, s_ip + '.raw')
                inp_shape = self.profile_info[s_ip][1]
                inp_dtype = self.profile_info[s_ip][0]
                inp_raw = np.fromfile(inp_path, dtype=inp_dtype)
                if ip in self.is_transpose_needed_dict:
                    inp_raw = transpose_to_nhwc(inp_raw, inp_shape)
                inp_path = os.path.join(self.framework_results, s_ip + '_nhwc.raw')
                inp_raw.tofile(inp_path)

                if handleInputNames:
                    # move req input files to temp folder
                    dst_path = self.work_dir + '/temp_inp/' + list(input_map.keys())[list(
                        input_map.values()).index(ip)] + '.raw'
                    try:
                        shutil.copy(inp_path, dst_path)
                        self.logger.debug('copied file {} to {}'.format(inp_path, dst_path))
                        inp_path = dst_path
                    except:
                        inp_path = self.work_dir + '/temp_inp/' + list(input_map.keys())[list(
                            input_map.values()).index(ip)] + '.raw'
                updated_input_list.append(ip + ':=' + inp_path)

        # creating new input-list-file for extracted model
        if len(updated_input_list) > 0:
            with open(self.output_dir + '/temp-list.txt', "w") as f:
                f.write(' '.join(updated_input_list))

        list_file = self.output_dir + '/temp-list.txt'
        return list_file

    def initiate_model_extraction(self, model, start_layer=None, end_layer=None, set_model=True):
        """
        This method partitions the model at start layer output till end layer and generates
        updated input list file
        Args:
            model : path to the model which needs to be partitioned
        Returns:
            status          : True if success
            model           : path to partitioned model
            list_file       : input list file for partitioned model
            new_g_inputs    : list of new inputs of partitioned model
        """
        s_utility = su.getInstance()
        self.model_handler = s_utility.getFrameworkInstance()

        # populate original_input_names_raw_map needed for end layer extraction.
        if set_model:
            start_layer = s_utility.getStartLayer()
            end_layer = s_utility.getEndLayer()
        valid_layers = [item[1] for item in self.model_traverser._layerlist]
        # check if valid layers are provided as start/end layers
        if start_layer and start_layer not in valid_layers:
            raise ConfigError('{} is not present in {}. Please provide valid start_layer'.format(
                start_layer, model))
        if end_layer and end_layer not in valid_layers:
            raise ConfigError('{} is not present in {}. Please provide valid end_layer'.format(
                end_layer, model))

        list_file = self.input_list_file
        original_input_names = self.model_traverser.framework_instance.get_input_layers(
            names_only=True)

        with open(list_file, 'r') as F:
            file_items = F.readline().strip().split(' ')
            file_paths = [f_path.split(':=')[-1] for f_path in file_items]
            self.original_input_names_raw_map = dict(zip(original_input_names, file_paths))
        (ret_status, model,
         new_g_inputs) = self.model_handler.extract_sub_graph(start_layer, end_layer,
                                                              self.output_dir)

        if not ret_status:
            return False, None, None, None
        # create input list file for partitioned model
        list_file = self.update_list_file(new_g_inputs)

        return True, model, list_file, new_g_inputs

    def handle_qnn_run_failure(self, std_out, cur_layer_out_name, layer_status_map, conv_fail_nodes,
                               lib_fail_nodes, cntx_fail_nodes, exec_fail_nodes):
        """
        This method handles the compilation and execution failures of qnn run
        Args:
            std_out             : output of qnn inference engine
            cur_layer_out_name  : output name of layer
            layer_status_map    : dict that gives status of each layer
            conv_fail_nodes     : list of qnn converter fail layers
            lib_fail_nodes      : list of qnn lib-generator fail layers
            cntx_fail_nodes     : list of qnn context binary generator fail layers
            exec_fail_nodes     : list of qnn net-run fail layers
        Returns:
            conv_fail_nodes     : updated list of qnn converter fail layers
            lib_fail_nodes      : updated list of qnn lib-generator fail layers
            cntx_fail_nodes     : updated list of qnn context binary generator fail layers
            exec_fail_nodes     : updated list of qnn net-run fail layers
            layer_status_map    : updated dict that gives status of each layer
        """
        s_cur_layer_out_name = santize_node_name(cur_layer_out_name)
        if 'ERROR_INFERENCE_ENGINE_BASE_CONVERSION_FAILED' in std_out:
            # handles qnn_converter failure
            conv_fail_nodes.append(cur_layer_out_name)
            self.logger.info(
                'Skipping current_node : {}, proceeding to next node'.format(cur_layer_out_name))
            layer_status_map[s_cur_layer_out_name] = LayerStatus.LAYER_STATUS_CON_ERROR

        elif 'ERROR_INFERENCE_ENGINE_LIB_GENERATOR_FAILED' in std_out:
            # handles qnn_lib_generator failure
            lib_fail_nodes.append(cur_layer_out_name)
            self.logger.info(
                'Skipping current_node : {}, proceeding to next node'.format(cur_layer_out_name))
            layer_status_map[s_cur_layer_out_name] = LayerStatus.LAYER_STATUS_LIB_ERROR

        elif 'ERROR_INFERENCE_ENGINE_CONTEXT_BINARY_GENERATE_FAILED' in std_out:
            # handles qnn_context_bin_gen failure
            cntx_fail_nodes.append(cur_layer_out_name)
            self.logger.info(
                'Skipping current_node : {}, proceeding to next node'.format(cur_layer_out_name))
            layer_status_map[s_cur_layer_out_name] = LayerStatus.LAYER_STATUS_CNTX_ERROR

        elif 'ERROR_INFERENCE_ENGINE_INFERENCE_FAILED' in std_out:
            # handles qnn_net_run failure
            exec_fail_nodes.append(cur_layer_out_name)
            self.logger.info(
                'Skipping current_node : {}, proceeding to next node'.format(cur_layer_out_name))
            layer_status_map[s_cur_layer_out_name] = LayerStatus.LAYER_STATUS_EXEC_ERROR
        return conv_fail_nodes, lib_fail_nodes, cntx_fail_nodes, exec_fail_nodes, layer_status_map

    def execute_on_qnn(self, model=None, list_file=None, output_tensors=None,
                       orig_model_outputs=None, out_dir=None, capture_intermediate_outputs=False,
                       float_fallback=False):
        """This method executes the given model on qnn platform.

        Args:
            model                           : path of the model
            list_file                       : file containing input paths to model
            output_tensors                  : output node names of model
            out_dir                         : output folder name inside work directory
            capture_intermediate_outputs    : boolean flag to save intermediate outputs of model
        Returns:
            ret_status                      : status of qnn execution
            std_out                         : console output of qnn inference engine
        """
        model = model if model else self.model
        list_file = list_file if list_file else self.input_list_file
        input_tensors = self.get_input_tensors(list_file, model)
        extra_converter_list = []
        extra_netrun_list = []
        args = {
            'framework':
            '{} {}'.format(self.framework,
                           (self.framework_version if self.framework_version else '')),
            'engine_path':
            self.engine_path,
            'runtime':
            self.runtime,
            'working_dir':
            os.path.join(self.output_dir),
            'output_dirname':
            out_dir,
            'input_list':
            list_file,
            'deviceId':
            self.deviceId,
            'host_device':
            self.host_device,
            'target_device':
            self.target_device,
            'model_path':
            model,
            'model_inputs':
            ''.join([
                ' --input_tensor {} {} {}'.format(item[0], item[1], item[2])
                for item in input_tensors
            ]),
            'model_outputs':
            ''.join([' --output_tensor {}'.format(name) for name in orig_model_outputs]),
            'target_architecture':
            self.architecture,
            'precision':
            self.precision,
            'extra_converter_args':
            self.extra_converter_args,
            'extra_runtime_args':
            self.extra_runtime_args,
            'verbose': (' -v' if self.logger.level == logging.DEBUG else ''),
            'act_quantizer':
            self.act_quantizer,
            'param_quantizer':
            self.param_quantizer,
            'bias_bitwidth':
            self.bias_bitwidth,
            'weights_bitwidth':
            self.weights_bitwidth,
            'act_bitwidth':
            self.act_bitwidth,
            'quantization_overrides':
            self.quantization_overrides,
            'algorithms':
            self.algorithms,
            'ignore_encodings':
            self.ignore_encodings,
            'per_channel_quantization':
            self.per_channel_quantization,
            'remote_server':
            self.remote_server,
            'remote_username':
            self.remote_username,
            'remote_password':
            self.remote_password,
        }

        inference_args = (' --framework {args[framework]}'
                          ' --engine_path {args[engine_path]}'
                          ' --runtime {args[runtime]}'
                          ' --working_dir {args[working_dir]}'
                          ' --output_dirname {args[output_dirname]}'
                          ' --input_list {args[input_list]}'
                          ' --deviceId {args[deviceId]}'
                          ' --host_device {args[host_device]}'
                          ' --model_path {args[model_path]}'
                          ' --architecture {args[target_architecture]}'
                          ' --precision {args[precision]}'
                          ' --remote_server {args[remote_server]}'
                          ' --remote_username {args[remote_username]}'
                          '{args[model_inputs]}'
                          '{args[model_outputs]}'
                          '{args[verbose]}').format(args=args)

        quantization_args = (' --act_quantizer {args[act_quantizer]}'
                             ' --param_quantizer {args[param_quantizer]}'
                             ' --bias_bitwidth {args[bias_bitwidth]}'
                             ' --weights_bitwidth {args[weights_bitwidth]}'
                             ' --act_bitwidth {args[act_bitwidth]}').format(args=args)

        if self.remote_password:
            inference_args += ' --remote_password ' + self.remote_password

        if self.use_native_input_files:
            inference_args += ' --use_native_input_files'
        if self.use_native_output_files:
            inference_args += ' --use_native_output_files'

        if self.runtime in [
                'aic', 'dsp', 'dspv68', 'dspv69', 'dspv73', 'dspv75', 'htp'
        ]:
            inference_args += ' --offline_prepare'

        if not capture_intermediate_outputs:
            # Used only for cumulative layerwise because it adds output node at current layer also
            if output_tensors is not None:
                args['add_layer_outputs'] = ','.join(['{}'.format(name) for name in output_tensors])
                inference_args += ' --add_layer_outputs {args[add_layer_outputs]}'.format(args=args)
            inference_args += ' --debug_mode_off'

        if self.precision in ['int8', 'fp16'] and self.compiler_config:
            inference_args += ' --compiler_config ' + self.compiler_config

        if self.quantization_overrides:
            quantization_args += ' --quantization_overrides ' + self.quantization_overrides
        if self.algorithms: quantization_args += ' --algorithms ' + self.algorithms
        if self.ignore_encodings:
            quantization_args += ' --ignore_encodings {args[ignore_encodings]}'.format(args=args)
        if self.per_channel_quantization:
            quantization_args += ' --per_channel_quantization {args[per_channel_quantization]}'.format(
                args=args)
        if float_fallback:
            quantization_args += " --float_fallback"
        inference_args += quantization_args
        if self.extra_converter_args:
            extra_converter_list = ['--extra_converter_args', self.extra_converter_args]
        if self.extra_runtime_args:
            extra_netrun_list = ['--extra_runtime_args', self.extra_runtime_args]

        # Execute model on QNN
        self.logger.info("Running exec_inference_engine with parameters: {}".format(
            inference_args + ' ' + ' '.join(extra_converter_list + extra_netrun_list)))
        try:
            #TODO: Need to use Python API call once available
            exec_inference_engine(self.engine,
                                  inference_args.split() + extra_converter_list + extra_netrun_list,
                                  self.logger)
        except Exception as e:
            self.logger.info(str(e))
            return 1, str(e)
        return 0, ''

    def partition_initial_model(self, model):
        s_utility = su.getInstance(self.args)

        if s_utility.getStartLayer() or s_utility.getEndLayer():
            self.set_profile_info(model)
            status, model, list_file, _ = self.initiate_model_extraction(model)
            if status is False:
                return status, None, None
            # keep a copy of extracted model as there is chance of replacement due to partitions
            if os.path.exists(os.path.join(self.work_dir, 'cleaned')) and os.path.isdir(
                    os.path.join(self.work_dir, 'cleaned')):
                if os.path.exists(model):
                    model_dir = os.path.dirname(model)
                    if not os.path.exists(
                            os.path.join(
                                self.output_dir, 'transformed' +
                                self.model_traverser.framework_instance.FRAMEWORK_SUFFIX)):
                        os.makedirs(
                            os.path.join(
                                self.output_dir, 'transformed' +
                                self.model_traverser.framework_instance.FRAMEWORK_SUFFIX))
                    for path in os.listdir(model_dir):
                        shutil.copy(
                            os.path.join(model_dir, path),
                            os.path.join('cleaned', 'cleanmodel' + os.path.splitext(path)[1]))
            else:
                if os.path.exists(model):
                    shutil.copy(
                        model,
                        os.path.join(
                            self.output_dir, 'cleanmodel' +
                            self.model_traverser.framework_instance.FRAMEWORK_SUFFIX))
            model = os.path.join(
                self.output_dir,
                'cleanmodel' + self.model_traverser.framework_instance.FRAMEWORK_SUFFIX)
        else:
            list_file = self.input_list_file

        return True, model, list_file

    def set_profile_info(self, model, list_file=None):
        """
        Create and set profile info of model.
        """
        s_utility = su.getInstance(self.args)
        self.model_handler = s_utility.setFrameworkInstance(self.logger, self.args, model)
        self.model_traverser = s_utility.setModelTraverserInstance(self.logger, self.args, model,
                                                                   self.add_layer_outputs,
                                                                   self.add_layer_types)

        original_output_names = self.model_handler.framework_instance.get_output_layers(
            names_only=True)
        original_input_names = self.model_handler.framework_instance.get_input_layers(
            names_only=True)

        if not list_file:
            list_file = self.input_list_file

        with open(list_file, 'r') as F:
            file_items = F.readline().strip().split(' ')
            file_paths = [f_path.split(':=')[-1] for f_path in file_items]
            self.original_input_names_raw_map = dict(zip(original_input_names, file_paths))

        # get profile info like tensor dimensions, dtype, min, max and median values
        profile_path = os.path.join(self.framework_results, 'profile_info.json')
        temp_profile_path = os.path.join(self.work_dir, 'temp', 'profile_info.json')
        if not os.path.exists(profile_path) and not os.path.exists(temp_profile_path):
            inputs = self.model_handler.framework_instance.get_input_layers()
            for idx, ip in enumerate(inputs):
                input_dim_str = ','.join(str(d) for d in ip[2])
                inputs[idx] = (ip[0], input_dim_str,
                               self.original_input_names_raw_map[inputs[idx][0]], ip[1])
            self.model_handler.generate_intermediate_outputs(os.path.join(self.work_dir,
                                                                          'temp'), input=inputs,
                                                             output=original_output_names)
            profile_path = os.path.join(self.work_dir, 'temp', 'profile_info.json')
        profile_info = read_json(profile_path)
        self.profile_info = profile_info

    def initial_run(self, model):
        """This method checks mismatch in original outputs of model with
        reference framwork outputs.

        Also does tensormapping to map qnn and reference tensor names
        """
        self.logger.debug('Initial run to check final output mismatch is started')
        self.logger.info('Started initial run to compare original outputs of model')
        s_utility = su.getInstance()
        percent_match_origin = OrderedDict()
        overall_comp_list = []
        comparators_list = s_utility.getComparator()
        layer_output_comp_map = {}
        final_outputs_mismatched = False
        output_tensors = self.model_handler.framework_instance.get_output_layers(names_only=True)
        ret_inference_engine, std_out = self.execute_on_qnn(model, self.input_list_file,
                                                            output_tensors, output_tensors,
                                                            out_dir='initial_run')

        if ret_inference_engine != 0:
            raise SnoopingError(std_out)

        temp_model_outputs = self.model_handler.framework_instance.get_output_layers()
        original_output_names = self.model_handler.framework_instance.get_output_layers(
            names_only=True)

        # comparision of qnn and reference outputs.
        for elem in temp_model_outputs:
            out_name = elem[0]
            out_dtype = elem[1]
            out_optype = elem[2]
            org_inf_raw, org_rt_raw = files_to_compare(self.framework_results, self.output_dir,
                                                       out_name, out_dtype, self.logger,
                                                       out_folder='initial_run')

            if out_name in layer_output_comp_map:
                comp_list = layer_output_comp_map[out_name]
            else:
                comp_list = comparators_list.copy()
            for _, comp in enumerate(comp_list):
                try:
                    match_origin, percent_origin = comp.verify(out_optype, None, [org_rt_raw],
                                                               [org_inf_raw], False)
                except Exception as e:
                    match_origin, percent_origin, _ = False, 0.0, ''
                    self.logger.info('Skipping comparision for node : {}, and marking 0.0% '
                                     'match'.format(out_name))
                # store percentage match for each user supplied comparator
                comp_name = comp.V_NAME
                if isinstance(percent_origin, str) and percent_origin == 'SAME':
                    percent_origin = 100.0
                percent_match_origin[comp_name] = round(percent_origin, 4)

                if comp_name not in overall_comp_list:
                    overall_comp_list.append(comp_name)

            if out_name in original_output_names:
                if not match_origin:
                    final_outputs_mismatched = True

        if final_outputs_mismatched:

            ret_inference_engine = self.execute_on_qnn(model, self.input_list_file, output_tensors,
                                                       output_tensors, out_dir='mapping_run',
                                                       capture_intermediate_outputs=True)
            get_mapping_arg = Namespace(None, framework=self.framework,
                                        version=self.framework_version, model_path=model,
                                        output_dir=self.work_dir, engine=self.engine,
                                        golden_dir_for_mapping=self.framework_results)
            self.logger.info('Creating tensor mapping as final outputs mismatched')
            tensor_mapping = TensorMapping(get_mapping_arg, self.logger)
            self.logger.debug('Completed initial run to check final output mismatch')
            return tensor_mapping

        else:
            self.logger.info('No mismatches seen in final outputs of model. Stopping debugger')
            exit(1)
