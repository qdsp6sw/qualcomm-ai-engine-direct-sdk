# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc.
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
from abc import abstractmethod

from qti.aisw.accuracy_debugger.lib.utils.nd_logger import setup_logger
from qti.aisw.accuracy_debugger.lib.utils.nd_exceptions import SnoopingError
from qti.aisw.accuracy_debugger.lib.runner.component_runner import exec_inference_engine, exec_framework_diagnosis
from qti.aisw.accuracy_debugger.lib.utils.nd_exceptions import ConfigError, SnoopingError
from qti.aisw.accuracy_debugger.lib.utils.nd_namespace import Namespace
from qti.aisw.accuracy_debugger.lib.utils.nd_path_utility import santize_node_name
from qti.aisw.accuracy_debugger.lib.utils.nd_framework_utility import read_json, transpose_to_nhwc
from qti.aisw.accuracy_debugger.lib.snooping.snooper_utils import files_to_compare, LayerStatus
from qti.aisw.accuracy_debugger.lib.inference_engine.nd_get_tensor_mapping import TensorMapping
from qti.aisw.accuracy_debugger.lib.snooping.snooper_utils import SnooperUtils as su


def signal_handler(sig, _):
    QAIRTSnooper.logger.info('Stopping snooping on user request.')
    if QAIRTSnooper.stop:
        QAIRTSnooper.logger.info('Waiting for current layer to complete.')
        QAIRTSnooper.stop = True
    else:
        sys.exit(1)


signal.signal(signal.SIGINT, signal_handler)


class QAIRTSnooper:
    stop = False
    logger = None

    def __init__(self, snooping_type, args, logger, verbose="info"):

        self.snooping_type = snooping_type
        self.args = args
        self._logger = logger if logger else setup_logger(verbose, args.output_dir)
        QAIRTSnooper.logger = self._logger
        self._args = args
        self.symbol_data_map = self.define_symbols()
        self.is_transpose_needed_dict = {}

    @abstractmethod
    def run(self):
        pass

    def define_symbols(self):
        """Populate symbol_data_map with mappings of onnx model symbols with
        corresponding data which is provided by user in extra_converter_args.
        Symbols are defined in the extra_converter_args as following examples:

        'define_symbol batch_size = 1'
        'define_symbol seq_len 128'

        """
        symbol_data_map = {}
        if self._args.extra_converter_args:
            converter_args = self._args.extra_converter_args.split(';')
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

    def _handle_qairt_run_failure(self, std_out, cur_layer_out_name, layer_status_map,
                                  conv_fail_nodes, lib_fail_nodes, cntx_fail_nodes,
                                  exec_fail_nodes):
        """
        This method handles the compilation and execution failures of qairt run
        """
        s_cur_layer_out_name = santize_node_name(cur_layer_out_name)
        if 'ERROR_INFERENCE_ENGINE_BASE_CONVERSION_FAILED' in std_out:
            # handles qairt_converter failure
            conv_fail_nodes.append(cur_layer_out_name)
            self.logger.info(
                'Skipping current_node : {}, proceeding to next node'.format(cur_layer_out_name))
            layer_status_map[s_cur_layer_out_name] = LayerStatus.LAYER_STATUS_CON_ERROR

        elif 'ERROR_INFERENCE_ENGINE_LIB_GENERATOR_FAILED' in std_out:
            # handles qairt_lib_generator failure
            lib_fail_nodes.append(cur_layer_out_name)
            self.logger.info(
                'Skipping current_node : {}, proceeding to next node'.format(cur_layer_out_name))
            layer_status_map[s_cur_layer_out_name] = LayerStatus.LAYER_STATUS_LIB_ERROR

        elif 'ERROR_INFERENCE_ENGINE_CONTEXT_BINARY_GENERATE_FAILED' in std_out:
            # handles qairt_context_bin_gen failure
            cntx_fail_nodes.append(cur_layer_out_name)
            self.logger.info(
                'Skipping current_node : {}, proceeding to next node'.format(cur_layer_out_name))
            layer_status_map[s_cur_layer_out_name] = LayerStatus.LAYER_STATUS_CNTX_ERROR

        elif 'ERROR_INFERENCE_ENGINE_INFERENCE_FAILED' in std_out:
            # handles qairt_net_run failure
            exec_fail_nodes.append(cur_layer_out_name)
            self.logger.info(
                'Skipping current_node : {}, proceeding to next node'.format(cur_layer_out_name))
            layer_status_map[s_cur_layer_out_name] = LayerStatus.LAYER_STATUS_EXEC_ERROR
        return conv_fail_nodes, lib_fail_nodes, cntx_fail_nodes, exec_fail_nodes, layer_status_map

    def get_input_tensors(self, list_file, model=None):
        input_tensors = []
        with open(list_file, 'r') as f:
            input_paths = f.readline().rstrip().split(' ')

        # get input_layers of extracted model by creating a new handler for it
        from qti.aisw.accuracy_debugger.lib.framework_diagnosis.nd_framework_runner import FrameworkRunner
        framework_args = Namespace(None, framework=self._args.framework,
                                   version=self._args.framework_version, model_path=model,
                                   output_dir=self._args.output_dir, engine='QAIRT')
        model_handler = FrameworkRunner(self._logger, framework_args)
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

    def set_profile_info(self, model, list_file=None):
        """
        Create and set profile info of model.
        """
        s_utility = su.getInstance(self._args)
        self.model_handler = s_utility.setFrameworkInstance(self._logger, self._args, model)
        self.model_traverser = s_utility.setModelTraverserInstance(self._logger, self.args, model,
                                                                   self._args.add_layer_outputs,
                                                                   self._args.add_layer_types)

        original_output_names = self.model_handler.framework_instance.get_output_layers(
            names_only=True)
        original_input_names = self.model_handler.framework_instance.get_input_layers(
            names_only=True)

        if not list_file:
            list_file = self._args.input_list

        with open(list_file, 'r') as F:
            file_items = F.readline().strip().split(' ')
            file_paths = [f_path.split(':=')[-1] for f_path in file_items]
            self.original_input_names_raw_map = dict(zip(original_input_names, file_paths))

        # get profile info like tensor dimensions, dtype, min, max and median values
        profile_path = os.path.join(self._args.golden_output_reference_directory,
                                    'profile_info.json')
        temp_profile_path = os.path.join(self._args.working_dir, 'temp', 'profile_info.json')
        if not os.path.exists(profile_path) and not os.path.exists(temp_profile_path):
            inputs = self.model_handler.framework_instance.get_input_layers()
            for idx, ip in enumerate(inputs):
                input_dim_str = ','.join(str(d) for d in ip[2])
                inputs[idx] = (ip[0], input_dim_str,
                               self.original_input_names_raw_map[inputs[idx][0]], ip[1])
            self.model_handler.generate_intermediate_outputs(
                os.path.join(self._args.working_dir, 'temp'), input=inputs,
                output=original_output_names)
            profile_path = os.path.join(self._args.working_dir, 'temp', 'profile_info.json')
        profile_info = read_json(profile_path)
        self.profile_info = profile_info

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
                inp_path = os.path.join(self._args.golden_output_reference_directory, s_ip + '.raw')
                inp_shape = self.profile_info[s_ip][1]
                inp_dtype = self.profile_info[s_ip][0]
                inp_raw = np.fromfile(inp_path, dtype=inp_dtype)
                if ip in self.is_transpose_needed_dict:
                    inp_raw = transpose_to_nhwc(inp_raw, inp_shape)
                inp_path = os.path.join(self._args.golden_output_reference_directory,
                                        s_ip + '_nhwc.raw')
                inp_raw.tofile(inp_path)

                if handleInputNames:
                    # move req input files to temp folder
                    dst_path = self._args.working_dir + '/temp_inp/' + list(input_map.keys())[list(
                        input_map.values()).index(ip)] + '.raw'
                    try:
                        shutil.copy(inp_path, dst_path)
                        self._logger.debug('copied file {} to {}'.format(inp_path, dst_path))
                        inp_path = dst_path
                    except:
                        inp_path = self._args.working_dir + '/temp_inp/' + list(
                            input_map.keys())[list(input_map.values()).index(ip)] + '.raw'
                updated_input_list.append(ip + ':=' + inp_path)

        # creating new input-list-file for extracted model
        if len(updated_input_list) > 0:
            with open(self._args.output_dir + '/temp-list.txt', "w") as f:
                f.write(' '.join(updated_input_list))

        list_file = self._args.output_dir + '/temp-list.txt'
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

        list_file = self._args.input_list
        original_input_names = self.model_traverser.framework_instance.get_input_layers(
            names_only=True)

        with open(list_file, 'r') as F:
            file_items = F.readline().strip().split(' ')
            file_paths = [f_path.split(':=')[-1] for f_path in file_items]
            self.original_input_names_raw_map = dict(zip(original_input_names, file_paths))
        (ret_status, model,
         new_g_inputs) = self.model_handler.extract_sub_graph(start_layer, end_layer,
                                                              self._args.output_dir)

        if not ret_status:
            return False, None, None, None
        # create input list file for partitioned model
        list_file = self.update_list_file(new_g_inputs)

        return True, model, list_file, new_g_inputs

    def partition_initial_model(self, model):
        s_utility = su.getInstance(self._args)

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
            list_file = self._args.input_list

        return True, model, list_file

    def execute_on_qairt(self, model_path=None, input_list=None, calib_input_list=None,
                         output_tensors=None, orig_model_outputs=[], input_tensors=None,
                         output_dirname=None, intermediate_outputs=False,
                         float_fallback=False, io_config=None, quantization_overrides=None,
                         no_graph_optimization=False, runtime=None, architecture=None,
                         working_dir=None, precision=None):
        """This method executes the given model on qairt platform.

        Args:
            model                           : path of the model
            list_file                       : file containing input paths to model
            output_tensors                  : output node names of model
            out_dir                         : output folder name inside work directory
            intermediate_outputs    : boolean flag to save intermediate outputs of model
        Returns:
            ret_status                      : status of qairt execution
            std_out                         : console output of qairt inference engine
        """
        input_tensors = input_tensors if input_tensors else self.get_input_tensors(
            input_list, model_path)
        runtime = runtime if runtime else self._args.runtime
        architecture = architecture if architecture else self._args.architecture
        working_dir = working_dir if working_dir else self._args.output_dir
        precision = precision if precision else self._args.precision

        required_args = [
            '--inference_engine', '--runtime', runtime, '--architecture', architecture,
            '--input_list', input_list
        ]

        converter_args = ['--input_network', model_path]
        for item in input_tensors:
            converter_args += ['--input_tensor', item[0], item[1]]
        for name in orig_model_outputs:
            converter_args += ['--output_tensor', name]
        if io_config:
            converter_args += ['--io_config', io_config]
        if quantization_overrides:
            converter_args += ['--quantization_overrides', quantization_overrides]
        if self._args.extra_converter_args:
            converter_args += ['--extra_converter_args', self._args.extra_converter_args]
        if calib_input_list:
            converter_args += ['--calibration_input_list', calib_input_list]

        quantizer_args = [
            '--bias_bitwidth',
            str(self._args.bias_bitwidth), '--act_bitwidth',
            str(self._args.act_bitwidth), '--weights_bitwidth',
            str(self._args.weights_bitwidth), '--act_quantizer_calibration',
            self._args.act_quantizer_calibration, '--param_quantizer_calibration',
            self._args.param_quantizer_calibration, '--act_quantizer_schema',
            self._args.act_quantizer_schema, '--param_quantizer_schema',
            self._args.param_quantizer_schema, '--percentile_calibration_value',
            str(self._args.percentile_calibration_value)
        ]
        if self._args.float_bitwidth:
            quantizer_args += ['--float_bitwidth', str(self._args.float_bitwidth)]
        if self._args.use_per_channel_quantization:
            quantizer_args += ['--use_per_channel_quantization']
        if self._args.use_per_row_quantization:
            quantizer_args += ['--use_per_row_quantization']
        if float_fallback or self._args.float_fallback:
            quantizer_args += ['--float_fallback']
        if self._args.extra_quantizer_args:
            quantizer_args += ['--extra_quantizer_args', self._args.extra_quantizer_args]
        if self._args.float_bitwidth:
            quantizer_args += ['--float_bitwidth', self._args.float_bitwidth]

        net_run_args = ['--perf_profile', self._args.perf_profile]
        if self._args.userlogs:
            net_run_args += ['--userlogs', self._args.userlogs]
        elif self._args.log_level:
            net_run_args += ['--log_level', self._args.log_level]
        if self._args.profiling_level:
            net_run_args += ['--profiling_level', self._args.profiling_level]
        if self._args.extra_runtime_args:
            net_run_args += ['--extra_runtime_args', self._args.extra_runtime_args]

        optional_args = [
            '--precision',
            precision,
            '--engine_path',
            self._args.engine_path,
            '--verbose',
            '--host_device',
            self._args.host_device,
            '--working_dir',
            working_dir,
            '--output_dirname',
            output_dirname,
        ]
        if self._args.executor_type:
            optional_args += ['--executor_type', self._args.executor_type]
        if self._args.deviceId:
            optional_args += ['--deviceId', self._args.deviceId]
        if intermediate_outputs is False:
            optional_args += ['--debug_mode_off']
            if output_tensors is not None:
                optional_args += ['--add_layer_outputs', ','.join(output_tensors)]
        if self._args.runtime not in ['aic', 'dsp', 'dspv68', 'dspv69', 'dspv73', 'dspv75']:
            optional_args += [' --disable_offline_prepare']

        # Execute model on qairt
        all_args = required_args + converter_args + quantizer_args + net_run_args + optional_args
        self._logger.info("Running exec_inference_engine with parameters: {}".format(all_args))
        try:
            #TODO: Need to use Python API call once available
            exec_inference_engine(args=all_args, engine_type="QAIRT", logger=self._logger,
                                  validate_args=True)
        except Exception as e:
            self._logger.error(str(e))
            return 1, str(e)
        return 0, ''

    def initial_run(self, model):
        """This method checks mismatch in original outputs of model with
        reference framwork outputs.

        Also does tensormapping to map qairt and reference tensor names
        """
        self._logger.debug('Initial run to check final output mismatch is started')
        self._logger.info('Started initial run to compare original outputs of model')
        s_utility = su.getInstance()
        percent_match_origin = OrderedDict()
        overall_comp_list = []
        comparators_list = s_utility.getComparator()
        layer_output_comp_map = {}
        final_outputs_mismatched = False
        output_tensors = self.model_handler.framework_instance.get_output_layers(names_only=True)
        ret_inference_engine, std_out = self.execute_on_qairt(
            model_path=model, input_list=self._args.input_list,
            calib_input_list=self._args.calibration_input_list,
            orig_model_outputs=output_tensors, output_dirname="initial_run",
            quantization_overrides=self._args.quantization_overrides)

        # Set the quantization_overrides if not provided by the user
        if not self._args.quantization_overrides:
            self._args.quantization_overrides = os.path.join(self._args.output_dir,
                                                             "inference_engine", "initial_run",
                                                             "base_quantized_encoding.json")

        if ret_inference_engine != 0:
            raise SnoopingError(std_out)

        temp_model_outputs = self.model_handler.framework_instance.get_output_layers()
        original_output_names = self.model_handler.framework_instance.get_output_layers(
            names_only=True)

        # comparision of qairt and reference outputs.
        for elem in temp_model_outputs:
            out_name = elem[0]
            out_dtype = elem[1]
            out_optype = elem[2]
            org_inf_raw, org_rt_raw = files_to_compare(self._args.golden_output_reference_directory,
                                                       self._args.output_dir, out_name, out_dtype,
                                                       self._logger, out_folder='initial_run')

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
                    self._logger.info('Skipping comparision for node : {}, and marking 0.0% '
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

            ret_inference_engine = self.execute_on_qairt(
                model_path=model, input_list=self._args.input_list,
                calib_input_list=self._args.calibration_input_list,
                orig_model_outputs=output_tensors, output_dirname="mapping_run",
                intermediate_outputs=True, quantization_overrides=self._args.quantization_overrides)
            tensor_mapping_dir = os.path.join(self._args.output_dir, "inference_engine",
                                              "mapping_run")
            tensor_mapping = os.path.join(tensor_mapping_dir, 'tensor_mapping.json')
            return tensor_mapping

        else:
            self._logger.info('No mismatches seen in final outputs of model. Stopping debugger')
            exit(1)

    def trigger_framework_runner(self, output_dirname=None, add_layer_types=[], add_layer_outputs=[],
                              skip_layer_types=[], skip_layer_outputs=[]):
        framework_args = []
        framework_args += ["--model_path", self._args.model_path]
        framework_args += ['--framework', self._args.framework]
        for item in self._args.input_tensor:
            framework_args += ['--input_tensor', *item]
        for item in self._args.output_tensor:
            framework_args += ['--output_tensor', item]
        framework_args += ['--working_dir', self._args.working_dir]
        if output_dirname:
            framework_args += ['--output_dirname', output_dirname]
        if self._args.verbose:
            framework_args += ['--verbose']
        if self._args.disable_graph_optimization:
            framework_args += ['--disable_graph_optimization']
        if self._args.onnx_custom_op_lib:
            framework_args += ['--onnx_custom_op_lib', self._args.onnx_custom_op_lib]
        if add_layer_types:
            framework_args += ['--add_layer_types', ','.join(add_layer_types)]
        if add_layer_outputs:
            framework_args += ['--add_layer_outputs', ','.join(add_layer_outputs)]
        if skip_layer_types:
            framework_args += ['--skip_layer_types', ','.join(skip_layer_types)]
        if skip_layer_outputs:
            framework_args += ['--skip_layer_outputs', ','.join(skip_layer_outputs)]
        if self._args.start_layer:
            framework_args += ['--start_layer', self._args.start_layer]
        if self._args.end_layer:
            framework_args += ['--end_layer', self._args.end_layer]

        exec_framework_diagnosis(args=framework_args, logger=self._logger, validate_args=True)
        framework_results = os.path.join(self._args.working_dir, 'framework_diagnosis', 'latest')

        if self._args.disable_graph_optimization and self._args.framework == "onnx":
            optimized_model_path = os.path.join(framework_results, "optimized_model.onnx")
            if os.path.exists(optimized_model_path):
                self._args.model_path = optimized_model_path
        self._args.golden_output_reference_directory = framework_results
