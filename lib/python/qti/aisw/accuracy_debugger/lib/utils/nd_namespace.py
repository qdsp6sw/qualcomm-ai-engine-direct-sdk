# =============================================================================
#
#  Copyright (c) 2019 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import copy
from argparse import Namespace
from qti.aisw.accuracy_debugger.lib.snooping.snooper_utils import SnooperUtils
from qti.aisw.accuracy_debugger.lib.utils.nd_logger import setup_logger
from qti.aisw.accuracy_debugger.lib.utils.nd_exceptions import FrameworkError


class Namespace(Namespace):

    def __init__(self, data=None, **kwargs):
        if data is not None:
            kwargs.update(data)
        super(Namespace, self).__init__(**kwargs)


def update_layer_options(parsed_args):
    """
    Returns string of comma separated layer output name for provided
    layer types.
    """
    filtered_layer_list = []
    s_utility = SnooperUtils.getInstance(parsed_args)
    logger = setup_logger(False, parsed_args.output_dir, disable_console_logging=True)
    args_copy = copy.deepcopy(parsed_args)
    add_layer_types = args_copy.__dict__.pop('add_layer_types')
    add_layer_outputs = args_copy.__dict__.pop('add_layer_outputs')
    skip_layer_types = args_copy.__dict__.pop('skip_layer_types')
    skip_layer_outputs = args_copy.__dict__.pop('skip_layer_outputs')
    try:
        start_layer = args_copy.__dict__.pop('start_layer')
    except :
        start_layer = None
    try:
        end_layer = args_copy.__dict__.pop('end_layer')
    except:
        end_layer = None
    model_traverser = s_utility.setModelTraverserInstance(logger, args_copy,
                                                            add_layer_outputs=add_layer_outputs,
                                                            add_layer_types=add_layer_types,
                                                            skip_layer_outputs=skip_layer_outputs,
                                                            skip_layer_types=skip_layer_types)
    add_layer_outputs = model_traverser.get_all_layers()

    if start_layer or end_layer:
        filtered_layer_list = [layer_item[1] for layer_item in model_traverser._layerlist]

        if start_layer:
            if start_layer in filtered_layer_list:
                filtered_layer_list =  filtered_layer_list[filtered_layer_list.index(start_layer):]
            else:
                logger.error(f'Invalid layer {start_layer} is provided as --start_layer. Please refer valid layers : {filtered_layer_list}')
                raise FrameworkError(f'Invalid layer {start_layer} is provided as --start_layer. Please refer log file to check valid layers')

        if end_layer:
            if end_layer in filtered_layer_list:
                filtered_layer_list =  filtered_layer_list[:filtered_layer_list.index(end_layer)+1]
            else:
                logger.error(f'Invalid layer {end_layer} is provided as --end_layer. Please refer valid layers : {filtered_layer_list}')
                raise FrameworkError(f'Invalid layer {end_layer} is provided as --end_layer.Please refer log file to check valid layers')

    final_node_list = list(set(filtered_layer_list).intersection(set(add_layer_outputs)))

    if len(final_node_list)==0:
        final_node_list = filtered_layer_list if len(filtered_layer_list)>0 else add_layer_outputs
    # clear snooperUtils instance
    SnooperUtils.clear()
    return ','.join(final_node_list)

def remove_layer_options(args_list):
    """
    Returns filtered args_list by removing all layer related options
    """
    filter_list = ['--start_layer', '--end_layer', '--add_layer_outputs', '--add_layer_types', '--skip_layer_outputs', '--skip_layer_types' ]
    for opt in filter_list:
        if opt in args_list:
            idx = args_list.index(opt)
            del args_list[idx:idx+2]

    return args_list