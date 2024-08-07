# =============================================================================
#
#  Copyright (c) 2021-2024 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import os
import json

import numpy as np

from qti.aisw.accuracy_debugger.lib.snooping.snooper_utils import SnooperUtils as su
from qti.aisw.accuracy_debugger.lib.snooping.nd_snooper import Snooper


class BinarySnooping(Snooper):
    """Class that runs layer wise snooping."""

    def __init__(self, args, logger):
        super(BinarySnooping, self).__init__("binary", args, logger)
        self._logger = logger
        self._args = args
        self.graph_result = {}
        self.subgraph_scores = {}
        self.sqnr_scores = {}
        self.fully_quantized_model_score = None
        self._node_activation_node_input_map = None
        self._valid_ir_activations = None
        self._activations_in_framework_model = None
        self.fp32_outputs = {}

        self._all_layers = None

    def get_valid_middle_layer(self, start_idx: int, end_idx: int):
        middle_index = (start_idx + end_idx) // 2
        for i in range((end_idx - start_idx) // 2):
            if middle_index - i >= start_idx and self._activations_in_framework_model[
                    middle_index - i] in self._valid_ir_activations:
                return middle_index - i + 1
            if middle_index + i < end_idx and self._activations_in_framework_model[
                    middle_index + i] in self._valid_ir_activations:
                return middle_index + i + 1
        return None

    def create_custom_node_precision_info(self, start_idx: int, end_idx: int):
        '''
        Creates the custom node precsion file by keeping a particular sets of node in lower precision
        for e.g. for node "conv.1", it's param encoding and output_encodings will be kept in lower precision.
        '''
        encodings = {}
        encodings['activation_encodings'] = {}
        encodings['param_encodings'] = {}

        with open(self._args.quantization_overrides, 'r') as file:
            npi_file = json.load(file)
        for output_name in self._activations_in_framework_model[start_idx:end_idx]:
            if output_name in self._valid_ir_activations:
                encodings['activation_encodings'][output_name] = npi_file["activation_encodings"][
                    output_name]
                for idx in range(len(encodings['activation_encodings'][output_name])):
                    encodings['activation_encodings'][output_name][idx]["is_symmetric"] = str(
                        encodings['activation_encodings'][output_name][idx]["is_symmetric"])

            if output_name in self._node_activation_node_input_map:
                for input_tensor_name in self._node_activation_node_input_map[output_name]:
                    if input_tensor_name in npi_file["param_encodings"]:
                        encodings["param_encodings"][input_tensor_name] = npi_file[
                            "param_encodings"][input_tensor_name]
                        for idx in range(len(encodings["param_encodings"][input_tensor_name])):
                            encodings["param_encodings"][input_tensor_name][idx][
                                "is_symmetric"] = str(encodings["param_encodings"]
                                                      [input_tensor_name][idx]["is_symmetric"])
        custom_node_precision_dump_dir = os.path.join(self._args.output_dir,
                                                      'sub_graph_node_precision_files')
        os.makedirs(custom_node_precision_dump_dir, exist_ok=True)
        file_name = "sub_graph_" + str(self._activations_in_framework_model[start_idx]).replace(
            '/', '#') + "_" + str(self._activations_in_framework_model[end_idx - 1].replace(
                '/', '#')) + ".json"
        custom_node_precision_dump_path = os.path.join(custom_node_precision_dump_dir, file_name)
        with open(custom_node_precision_dump_path, 'w') as file:
            json.dump(encodings, file, indent=4)
        return custom_node_precision_dump_path

    def execute_graph(self, start_idx: int, end_idx: int, float_fallback: bool = True,
                      create_subgraph: bool = True):
        '''
        Pseudo Code:
            1. Convert the qnn_net.json to a custom aimet profile file for nodes in between start_idx and end_idx
            2. call execute on qnn
        '''
        if create_subgraph:
            custom_node_precision_path = self.create_custom_node_precision_info(start_idx, end_idx)
            self.quantization_overrides = custom_node_precision_path
        self.execute_on_qnn(
            out_dir="{}_{}".format(
                self._activations_in_framework_model[start_idx].replace('/', '#'),
                self._activations_in_framework_model[end_idx - 1].replace('/', '#')),
            float_fallback=float_fallback, orig_model_outputs=self._args.output_tensor,
            list_file=self._args.qnn_input_list)

    def _load_fp32_results(self):
        reference_directory = os.path.join(self._args.working_dir, 'framework_diagnosis')
        for fp32_output_dir in os.scandir(reference_directory):
            fp32_output_dir_name = fp32_output_dir.name
            if fp32_output_dir_name != "latest":
                _fp32_outputs = {}
                for file_entry in os.scandir(fp32_output_dir.path):
                    file_entry_name = file_entry.name
                    if file_entry_name.endswith('.raw'):
                        _fp32_outputs[file_entry_name] = np.fromfile(file_entry.path,
                                                                     dtype=np.float32)
                self.fp32_outputs[fp32_output_dir_name] = _fp32_outputs

    def _calculate_verifer_score(self, target_directory):
        sqnr = 0.0
        mse = 0.0
        for out_node in self._args.output_tensor:
            activation_name = "_" + out_node + ".raw" if out_node[0].isdigit(
            ) else out_node + ".raw"
            for quantized_output_directory in os.scandir(os.path.join(target_directory, 'output')):
                if os.path.isdir(quantized_output_directory):
                    quantized_output_directory_name = quantized_output_directory.name
                    target_tensor_path = os.path.join(quantized_output_directory.path, activation_name)
                    reference_tensor = self.fp32_outputs[quantized_output_directory_name][
                        activation_name]
                    target_tensor = np.fromfile(target_tensor_path, dtype=np.float32)
                    delta = reference_tensor - target_tensor
                    _mse = np.mean(delta**2)
                    mse += _mse
                    mss = np.mean(np.array(reference_tensor)**2)
                    sqnr += 10 * np.log10(mss / _mse)
        sqnr /= len(self._args.output_tensor)
        mse /= len(self._args.output_tensor)
        if self._args.verifier.lower() == 'sqnr':
            return sqnr
        self.sqnr_scores[os.path.basename(target_directory)] = sqnr
        return mse

    def _is_search_candidate(self, tree1_score, tree2_score):
        # add check for sqnr
        # add the full_quantized_score check
        total_tree_score = tree1_score + tree2_score
        if tree1_score > 0.4 * total_tree_score:
            return True

    def binary_search(self, start_idx: int, end_idx: int):
        '''
        Psuedo Code:
            1. Get the mid layer
            2. Find the comaprator loss for first subgraph
            3. Find the comparator loss for second subgraph
            4. call the binary based on condition
        
        :param start_idx: index pointing to the node in the graph from where subgraph will start"
        :param end_idx: index pointing to the node in the graph to which subgraph will end"
        '''
        # Base Condition :Remove
        if (end_idx - start_idx) <= self.args.min_graph_size:
            # Do logging and return
            return None
        mid_index = self.get_valid_middle_layer(start_idx, end_idx)
        if mid_index is None:
            # Do logging and return
            return None
        # # Execute the graph
        try:
            self.execute_graph(start_idx, mid_index)
            left_output_path = os.path.join(
                self._args.output_dir, 'inference_engine',
                "{}_{}".format(self._activations_in_framework_model[start_idx],
                               self._activations_in_framework_model[mid_index - 1]))
            left_sub_tree_score = self._calculate_verifer_score(left_output_path)
        except Exception as e:
            self.logger.info("GOT LEFT SUBTREE EXCEPTION:\n" + str(e))
            left_sub_tree_score = None
        try:
            if mid_index == end_idx:
                raise IndexError
            self.execute_graph(mid_index, end_idx)
            right_output_path = os.path.join(
                self._args.output_dir, 'inference_engine',
                "{}_{}".format(self._activations_in_framework_model[mid_index],
                               self._activations_in_framework_model[end_idx - 1]))
            right_sub_tree_score = self._calculate_verifer_score(right_output_path)
        except Exception as e:
            self.logger.info("GOT RIGHT SUBTREE EXCEPTION:\n" + str(e))
            right_sub_tree_score = None

        master_tree = "{}_{}".format(
            self._activations_in_framework_model[start_idx].replace('/', '#'),
            self._activations_in_framework_model[end_idx - 1].replace('/', '#'))
        self.graph_result[master_tree] = {}
        left_tree = "{}_{}".format(
            self._activations_in_framework_model[start_idx].replace('/', '#'),
            self._activations_in_framework_model[mid_index - 1].replace('/', '#'))
        right_tree = "{}_{}".format(
            self._activations_in_framework_model[mid_index].replace('/', '#'),
            self._activations_in_framework_model[end_idx - 1].replace('/', '#'))
        self.graph_result[master_tree][left_tree] = str(left_sub_tree_score)
        self.graph_result[master_tree][right_tree] = str(right_sub_tree_score)
        self.subgraph_scores[left_tree] = str(left_sub_tree_score)
        self.subgraph_scores[right_tree] = str(right_sub_tree_score)

        self.logger.info(
            "Score for master tree {} with left child {} is {} and right child {} is {}".format(
                master_tree, left_tree, left_sub_tree_score, right_tree, right_sub_tree_score))

        if mid_index != end_idx:
            if right_sub_tree_score is None or self._is_search_candidate(
                    left_sub_tree_score, right_sub_tree_score):
                self.binary_search(start_idx, mid_index)
            if left_sub_tree_score is None or self._is_search_candidate(
                    right_sub_tree_score, left_sub_tree_score):
                self.binary_search(mid_index, end_idx)

    def run(self):
        """This method contains the sequence of debugger for
        BinarySnooping."""
        self._logger.info('Starting Binary Snooping')
        s_utility = su.getInstance(self._args)
        self.model_traverser = s_utility.setModelTraverserInstance(self._logger, self._args)
        framework_instance = self.model_traverser.framework_instance
        self._node_activation_node_input_map = framework_instance.get_node_activation_to_node_inputs_map(
        )

        global snoop_stop
        snoop_stop = False

        # get the valid_ir_nodes
        self._get_valid_ir_activations()
        self._activations_in_framework_model = self.model_traverser.get_all_layers()
        start_idx = 0
        end_idx = len(self._activations_in_framework_model)
        self._logger.info('Started Binary Snooping')
        self._load_fp32_results()
        self.execute_graph(start_idx, end_idx, False, False)
        fully_quantized_model_output_dir = os.path.join(
            self._args.output_dir, 'inference_engine',
            "{}_{}".format(self._activations_in_framework_model[0].replace('/', '#'),
                           self._activations_in_framework_model[-1].replace('/', '#')))
        self.fully_quantized_model_score = self._calculate_verifer_score(
            fully_quantized_model_output_dir)
        self.logger.info("Verifier score for fully quantized model is {}".format(
            self.fully_quantized_model_score))
        self.graph_result["full_model_quantized"] = str(self.fully_quantized_model_score)
        self.binary_search(start_idx, end_idx)
        self.subgraph_scores = {
            k: v
            for k, v in sorted(self.subgraph_scores.items(), key=lambda item: item[1], reverse=True)
        }
        graph_result_json_file = os.path.join(self._args.output_dir, 'graph_result.json')
        subgraph_result_json_file = os.path.join(self._args.output_dir, 'subgraph_result.json')
        sqnr_result_json_file = os.path.join(self._args.output_dir, 'sqnr_result.json')
        with open(graph_result_json_file, 'w') as file:
            json.dump(self.graph_result, file, indent=4)
        with open(subgraph_result_json_file, 'w') as file:
            json.dump(self.subgraph_scores, file, indent=4)
        with open(sqnr_result_json_file, 'w') as file:
            json.dump(self.sqnr_scores, file, indent=4)

    def _get_valid_ir_activations(self):
        npi_file_path = self._args.quantization_overrides
        with open(npi_file_path, 'r') as file:
            npi_file = json.load(file)
        self._valid_ir_activations = set(list(npi_file["activation_encodings"].keys()))
