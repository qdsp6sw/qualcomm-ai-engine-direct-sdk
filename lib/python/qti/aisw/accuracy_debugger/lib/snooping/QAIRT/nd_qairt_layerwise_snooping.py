# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import logging
import os
import traceback
import pandas as pd

from qti.aisw.accuracy_debugger.lib.snooping.snooper_utils import SnooperUtils as su
from qti.aisw.accuracy_debugger.lib.snooping.snooper_utils import show_progress, LayerStatus, files_to_compare
from qti.aisw.accuracy_debugger.lib.utils.nd_framework_utility import read_json
from qti.aisw.accuracy_debugger.lib.utils.nd_path_utility import santize_node_name
from qti.aisw.accuracy_debugger.lib.utils.nd_verifier_utility import permute_tensor_data_axis_order, get_irgraph_axis_data
from qti.aisw.accuracy_debugger.lib.snooping.QAIRT.nd_qairt_snooper import QAIRTSnooper


class QAIRTLayerwiseSnooping(QAIRTSnooper):
    """Class that runs layer wise snooping."""

    def __init__(self, args, logger, verbose="info"):
        super().__init__(snooping_type="layerwise", args=args, logger=logger, verbose=verbose)

    def run(self):
        """This method contains the sequence of debugger for
        LayerwiseSnooping."""
        # Execute the framework diagnosis
        if self._args.golden_output_reference_directory is None:
            self.trigger_framework_runner()
        model = self._args.model_path
        ret_status = True
        s_utility = su.getInstance(self._args)

        QAIRTSnooper.stop = False

        layer_status_map = {}
        layer_perc_map = {}
        layer_compare_info_map = {}
        layer_type_map = {}
        layer_shape_map = {}
        layer_dtype_map = {}
        layer_profile_map = {}
        conv_fail_nodes = []
        lib_fail_nodes = []
        cntx_fail_nodes = []
        exec_fail_nodes = []
        extract_fail_nodes = []
        compare_skip_nodes = []
        overall_comp_list = []
        comparators_list = s_utility.getComparator()
        layer_output_comp_map = {}
        layerwise_report_path = os.path.join(self._args.output_dir, 'layerwise.csv')

        # partition the model from user supplied -start-from-layer-output
        # the input list file is updated accordingly.
        status, model, list_file = self.partition_initial_model(model)
        self._args.input_list = list_file
        if status is False:
            return status

        self.set_profile_info(model)
        original_output_names = self.model_handler.framework_instance.get_output_layers(
            names_only=True)

        profile_info = self.profile_info
        total_layers = self.model_traverser.get_layer_count()
        skip_count = 0

        tensor_mapping = self.initial_run(model)
        # get list of nodes from qnn inference tensor mapping
        valid_nodes = list(read_json(tensor_mapping).values())
        self.logger.info('Started layerwise snooping')

        # Snooping loop
        count = 0
        prev_layer_out_name = None
        while (True):
            skip_compare = False
            # Get next layer
            (status, layer_name, cur_layer_out_name,
             layer_type) = self.model_traverser.get_next_layer()
            if status == 1 or QAIRTSnooper.stop:
                # Reached end.
                break
            # check if cur_layer_out_name is in qnn valid nodes
            if not cur_layer_out_name:
                continue
            s_cur_layer_out_name = santize_node_name(cur_layer_out_name)
            if s_cur_layer_out_name not in valid_nodes:
                continue

            if prev_layer_out_name is not None:
                # Populate layer details with default and known values.
                layer_perc_map[s_cur_layer_out_name] = '-'
                layer_compare_info_map[s_cur_layer_out_name] = '-'

                if cur_layer_out_name in original_output_names:
                    layer_type_map[s_cur_layer_out_name] = '(*)' + layer_type
                else:
                    layer_type_map[s_cur_layer_out_name] = layer_type

                if profile_info and s_cur_layer_out_name in profile_info:
                    layer_profile_map[s_cur_layer_out_name] = profile_info[s_cur_layer_out_name][2:]
                    layer_shape_map[s_cur_layer_out_name] = profile_info[s_cur_layer_out_name][1]
                    layer_dtype_map[s_cur_layer_out_name] = profile_info[s_cur_layer_out_name][0]
                else:
                    layer_profile_map[s_cur_layer_out_name] = '-'
                    layer_shape_map[s_cur_layer_out_name] = '-'
                    layer_dtype_map[s_cur_layer_out_name] = '-'

                layer_status_map[s_cur_layer_out_name] = LayerStatus.LAYER_STATUS_SUCCESS

                if status == 2:  # Skipped layer
                    count += 1
                    skip_count += 1
                    prog_info = str(count) + '/' + str(total_layers) + ', skipped:' + str(
                        skip_count)
                    show_progress(total_layers, count, prog_info)
                    layer_status_map[s_cur_layer_out_name] = LayerStatus.LAYER_STATUS_SKIPPED
                    continue

                # Check if snooping needs to be stopped
                if QAIRTSnooper.stop:
                    break

                count += 1
                # Show progress
                prog_info = str(count) + '/' + str(total_layers) + ', skipped:' + str(skip_count)
                show_progress(total_layers, count, prog_info)

                self.logger.debug('Debugging layer ' + layer_name)

                # Extract sub model. Continue to next layer if extraction fails.
                try:
                    ret_status, extracted_model_path, list_file, _ = \
                        self.initiate_model_extraction(model, prev_layer_out_name, cur_layer_out_name, set_model = False)
                    if ret_status:
                        # update status as partition success if no error
                        layer_status_map[s_cur_layer_out_name] += LayerStatus.LAYER_STATUS_PARTITION
                except Exception as e:
                    ret_status = False
                    traceback.print_exc()
                    self.logger.error('Extraction error {}'.format(e))
                if not ret_status:
                    extract_fail_nodes.append(cur_layer_out_name)
                    self.logger.error('Extraction failed at node {}'.format(cur_layer_out_name))
                    if cur_layer_out_name in layer_status_map:
                        layer_status_map[
                            s_cur_layer_out_name] += ',' + LayerStatus.LAYER_STATUS_PARTITION_ERR
                    else:
                        layer_status_map[
                            s_cur_layer_out_name] = LayerStatus.LAYER_STATUS_PARTITION_ERR
                    continue

                # Use this extracted model for debugging.
                temp_model = extracted_model_path
                # self.model_handler = s_utility.setFrameworkInstance(self.logger, self.args, temp_model)

                output_tensors = None

                # Execute model on QAIRT
                ret_inference_engine, std_out = self.execute_on_qairt(
                    model_path=temp_model, input_list=list_file,
                    orig_model_outputs=[cur_layer_out_name], output_dirname=s_cur_layer_out_name,
                    float_fallback=True, quantization_overrides=self._args.quantization_overrides)

                if ret_inference_engine != 0:
                    self.logger.warning(
                        f'inference-engine failed for the layer -> {s_cur_layer_out_name}, skipping verification'
                    )
                    skip_compare = True
                    percent_match = 0.0
                    conv_fail_nodes, lib_fail_nodes, cntx_fail_nodes, exec_fail_nodes, layer_status_map = self._handle_qairt_run_failure(
                        std_out, cur_layer_out_name, layer_status_map, conv_fail_nodes,
                        lib_fail_nodes, cntx_fail_nodes, exec_fail_nodes)

                # Compare current layer outputs
                if not skip_compare:
                    d_type = [layer_dtype_map[s_cur_layer_out_name]]
                    inf_raw, rt_raw = files_to_compare(self._args.golden_output_reference_directory,
                                                       self._args.output_dir, cur_layer_out_name,
                                                       d_type[0], self.logger)
                    quantized_dlc_path = os.path.join(self._args.output_dir, "inference_engine",
                                                      s_cur_layer_out_name, "base_quantized.dlc")
                    axis_data = get_irgraph_axis_data(dlc_path=quantized_dlc_path, output_dir=self._args.output_dir)
                    sanitized_cur_layer_out_name = santize_node_name(cur_layer_out_name)

                    if axis_data is not None and sanitized_cur_layer_out_name in axis_data:
                        tensor_dict = axis_data[sanitized_cur_layer_out_name]
                        src_axis_format = tensor_dict['src_axis_format']
                        axis_format = tensor_dict['axis_format']
                        tensor_dims = tensor_dict['dims']
                        rt_raw, is_permuted = permute_tensor_data_axis_order(
                            src_axis_format, axis_format, tensor_dims, rt_raw)
                        self.is_transpose_needed_dict[cur_layer_out_name] = is_permuted

                    info_origin = {}
                    if (inf_raw is not None) and (rt_raw is not None):
                        percent_match = {}
                        if cur_layer_out_name in layer_output_comp_map:
                            comp_list = layer_output_comp_map[s_cur_layer_out_name]
                        else:
                            comp_list = comparators_list.copy()

                        for idx, comp in enumerate(comp_list):
                            try:
                                match_info = '-'
                                _, percent = comp.verify(layer_type, None, [rt_raw], [inf_raw],
                                                         False)
                            except Exception:
                                percent, match_info = 0.0, ''
                                compare_skip_nodes.append(cur_layer_out_name)
                                self.logger.debug(
                                    'Skipping comparision for node : {}, and marking 0.0% match'.
                                    format(cur_layer_out_name))
                                layer_status_map[
                                    cur_layer_out_name] = LayerStatus.LAYER_STATUS_COMPARE_ERROR
                            # store percentage match for each user supplied comparator
                            comp_name = comp.V_NAME
                            if isinstance(percent, str) and percent == 'SAME':
                                percent = 100.0
                            percent_match[comp_name] = round(percent, 4)
                            if match_info:
                                info_origin[comp_name] = comp_name + ": " + match_info
                            # maintain a list of over all comparators used in snooping
                            if comp_name not in overall_comp_list:
                                overall_comp_list.append(comp_name)
                    else:
                        percent_match = 0.0

                    self.logger.info('Debug Layer {}, output {} match percent {}'.format(
                        layer_name, cur_layer_out_name, percent_match))

                    layer_perc_map[s_cur_layer_out_name] = percent_match
                    layer_compare_info_map[s_cur_layer_out_name] = "\n".join(
                        list(info_origin.values()))

            prev_layer_out_name = cur_layer_out_name

            #  Exit if end layer is provided
            if s_utility.getEndLayer() == cur_layer_out_name:
                skip_count += (total_layers - count)
                count = total_layers
                prog_info = str(count) + '/' + str(total_layers) + ', skipped:' + str(skip_count)
                show_progress(total_layers, count, prog_info)
                break

        print("============== Layerwise Debug Results ==============")
        pd.set_option('display.max_rows', None, 'display.max_colwidth', 30, 'expand_frame_repr',
                      False)

        # to split the layer_perc_map into multiple dicts comparator wise
        overall_comp_list.sort()
        perc_compwise_map = {}
        for idx, elem in enumerate(overall_comp_list):
            _updated = {}
            for k, v in layer_perc_map.items():
                try:
                    if overall_comp_list[idx] in v:
                        _updated[k] = v[overall_comp_list[idx]]
                    else:
                        _updated[k] = '-'
                except:
                    _updated[k] = '-'
            perc_compwise_map[elem] = _updated

        #Check if info column is populated for all the keys:
        for op in layer_perc_map.keys():
            if op not in layer_compare_info_map or layer_compare_info_map[op] == '':
                layer_compare_info_map[op] = '-'

        perc_compwise_list = [perc_compwise_map[elem] for elem in overall_comp_list]
        results_dicts = ([layer_status_map, layer_type_map, layer_shape_map, layer_profile_map] +
                         perc_compwise_list + [layer_compare_info_map])
        results_dict = {}
        for k in layer_perc_map.keys():
            results_dict[k] = tuple(d[k] for d in results_dicts)
        if len(results_dict) == 0:
            logging.info('No layers has been debugged.')
            return ret_status

        df = pd.DataFrame.from_dict(results_dict, orient='index')
        labels = ['Status', 'Layer Type', 'Shape', 'Activations (Min,Max,Median)'
                  ] + overall_comp_list + ['Info']
        df.columns = labels
        df.index.name = 'O/P Name'
        print('\n' + str(df))
        df.to_csv(layerwise_report_path)
        print('Results saved at {}'.format(layerwise_report_path))
        print("\n============== Error details ==============")
        print(
            'Converter Failures at nodes : {} \nLibgenerator Failures at nodes : {} \nContext Binary Genrator Failures at nodes : {} \nExtraction Failures at nodes : {} \nExecution '
            'Failures at nodes : {} \nComparition Failures at nodes : {}'.format(
                str(conv_fail_nodes), str(lib_fail_nodes), str(cntx_fail_nodes),
                str(extract_fail_nodes), str(exec_fail_nodes), str(compare_skip_nodes)))

        # Layer Snooping completed.
        self.logger.debug('Layerwise snooping completed successfully')
        return ret_status
