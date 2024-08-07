# =============================================================================
#
#  Copyright (c) 2019-2022 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

from pathlib import Path
import numpy as np
import pandas as pd
import os

from qti.aisw.accuracy_debugger.lib.utils.nd_constants import AxisFormat
from qti.aisw.accuracy_debugger.lib.utils.nd_path_utility import santize_node_name
from qti.aisw.accuracy_debugger.lib.utils.nd_framework_utility import read_json, dump_json


def get_tensors_axis_from_dlc(dlc_path):
    """
    Returns axis of each tensor in the given dlc
    """

    from qti.aisw.dlc_utils import modeltools

    model_reader = modeltools.IrDlcReader()
    model_reader.open(dlc_path)
    ir_graph = model_reader.get_ir_graph()

    irgraph_axis_data = {}
    for name, tensor in ir_graph.get_tensor_map().items():
        sanitized_name = santize_node_name(name)
        irgraph_axis_data[sanitized_name] = {
            'src_axis_format': tensor.src_axis_format().name,
            'axis_format': tensor.axis_format().name,
            'dims': tensor.dims()
        }

    model_reader.close()
    return irgraph_axis_data


def get_irgraph_axis_data(qnn_model_json_path=None, dlc_path=None, output_dir=None):
    """
    Returns axis of each tensor in the given qnn_model_json_path/dlc
    """
    irgraph_axis_data = {}
    if qnn_model_json_path is not None:
        qnn_model_json = read_json(qnn_model_json_path)
        irgraph_axis_data = qnn_model_json['graph']['tensors']
    elif dlc_path is not None:
        dlc_name = os.path.basename(dlc_path)
        axis_info_json_path = os.path.join(output_dir, dlc_name.replace('.dlc', '.json'))
        if os.path.exists(axis_info_json_path):
            irgraph_axis_data = read_json(axis_info_json_path)
        else:
            irgraph_axis_data = get_tensors_axis_from_dlc(dlc_path)
            dump_json(irgraph_axis_data, axis_info_json_path)

    return irgraph_axis_data


def permute_tensor_data_axis_order(src_axis_format, axis_format, tensor_dims, golden_tensor_data):
    """Permutes intermediate tensors goldens to spatial-first axis order for
    verification :param src_axis_format: axis format of source framework tensor
    :param axis_format: axis format of QNN tensor :param tensor_dims: current
    dimensions of QNN tensor :param golden_tensor_data: golden tensor data to
    be permuted :return: np.array of permuted golden tensor data."""

    # base case for same axis format or other invalid cases
    invalid_axis = ['NONTRIVIAL', 'NOT_YET_DEFINED', 'ANY']
    if src_axis_format == axis_format or \
        src_axis_format in invalid_axis or axis_format in invalid_axis:
        return golden_tensor_data, False
    # reshape golden data to spatial-last axis format
    golden_tensor_data = np.reshape(
        golden_tensor_data,
        tuple([
            tensor_dims[i]
            for i in AxisFormat.axis_format_mappings.value[(src_axis_format, axis_format)][0]
        ]))
    # transpose golden data to spatial-first axis format
    golden_tensor_data = np.transpose(
        golden_tensor_data,
        AxisFormat.axis_format_mappings.value[(src_axis_format, axis_format)][1])
    # return flatten golden data
    return golden_tensor_data.flatten(), True


def to_csv(data, file_path):
    if isinstance(data, pd.DataFrame):
        data.to_csv(file_path, encoding='utf-8', index=False)


def to_html(data, file_path):
    if isinstance(data, pd.DataFrame):
        data.to_html(file_path, classes='table', index=False)


def to_json(data, file_path):
    if isinstance(data, pd.DataFrame):
        data.to_json(file_path, orient='records', indent=4)


def save_to_file(data, filename) -> None:
    """Save data to file in CSV, HTML and JSON formats :param data: Data to be
    saved to file :param filename: Name of the file."""
    filename = Path(filename)
    to_csv(data, filename.with_suffix(".csv"))
    to_html(data, filename.with_suffix(".html"))
    to_json(data, filename.with_suffix(".json"))
