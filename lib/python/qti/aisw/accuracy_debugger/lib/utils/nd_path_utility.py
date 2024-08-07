# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import os
import re
import shutil
from pathlib import Path

from qti.aisw.accuracy_debugger.lib.utils.nd_errors import get_message
from qti.aisw.accuracy_debugger.lib.utils.nd_exceptions import ParameterError, InferenceEngineError
from qti.aisw.accuracy_debugger.lib.utils.nd_constants import Engine


def get_absolute_path(dir, checkExist=True, pathPrepend=None):
    """
    Returns a absolute path
    :param dir: the relate path or absolute path
           checkExist: whether to check whether the path exists
    :return: absolute path
    """
    if not dir:
        return dir

    absdir = os.path.expandvars(dir)
    if not os.path.isabs(absdir):
        if not pathPrepend:
            absdir = os.path.abspath(absdir)
        else:
            absdir = os.path.join(pathPrepend, dir)

    if not checkExist:
        return absdir

    if os.path.exists(absdir):
        return absdir
    else:
        raise ParameterError(dir + "(relpath) and " + absdir + "(abspath) are not existing")


def get_tensor_paths(tensors_path):
    """Returns a dictionary indexed by k, of tensor paths :param tensors_path:
    path to output directory with all the tensor raw data :return:
    Dictionary."""
    tensors = {}
    for dir_path, sub_dirs, files in os.walk(tensors_path):
        for file in files:
            if file.endswith(".raw"):
                tensor_path = os.path.join(dir_path, file)
                # tensor name is part of it's path
                path = os.path.relpath(tensor_path, tensors_path)

                # remove .raw extension
                tensor_name = str(Path(path).with_suffix(''))
                tensors[tensor_name] = tensor_path
    return tensors


def format_args(additional_args, ignore_args=[]):
    """Returns a formatted string(after removing args mentioned in ignore_args) to append to qnn/snpe core tool's commands
    :param additional_args: extra options to be addded to append to qnn/snpe core tool's commands
    commands :param ignore_args: list of args to be ignored :return: String."""
    extra_options = additional_args.split(';')
    extra_cmd = ''
    for item in extra_options:
        arg = item.strip(' ').split('=')
        if arg[0].rstrip(' ') in ignore_args:
            continue
        if len(arg) == 1:
            extra_cmd += '--' + arg[0].rstrip(' ') + ' '
        else:
            extra_cmd += '--' + arg[0].rstrip(' ') + ' ' + arg[1].lstrip(' ') + ' '
    return extra_cmd


def format_params(config_params):
    """Returns a dict of key-value pairs to be used in context/graph config
    :param config_params: extra config params to be added to context config / netrun config file
    :return: Dictionary."""
    extra_options = config_params.split(';')
    config_dict = {}
    for item in extra_options:
        arg = item.strip(' ').split('=')
        config_dict[arg[0]] = arg[1]
    return config_dict


def retrieveQnnSdkDir(filePath=__file__):
    filePath = Path(filePath).resolve()
    try:
        # expected path to this file in the SDK: <QNN root>/lib/python/qti/aisw/accuracy_debugger/lib/utils/nd_path_utility.py
        qnn_sdk_dir = filePath.parents[7]  # raises IndexError if out of bounds
        if (qnn_sdk_dir.match('qnn-*') or qnn_sdk_dir.match('qaisw-*')):
            return str(qnn_sdk_dir)
        else:
            qnn_path = filePath
            for _ in range(len(filePath.parts)):
                qnn_path = qnn_path.parent
                if (qnn_path.match('qnn-*') or qnn_path.match('qaisw-*')):
                    return str(qnn_path)

            raise InferenceEngineError(get_message('ERROR_INFERENCE_ENGINE_SDK_NOT_FOUND')('QNN'))
    except IndexError:
        raise InferenceEngineError(get_message('ERROR_INFERENCE_ENGINE_SDK_NOT_FOUND')('QNN'))


def retrieveSnpeSdkDir(filePath=__file__):
    filePath = Path(filePath).resolve()
    try:
        # expected path to this file in the SDK: <SNPE root>/lib/python/qti/aisw/accuracy_debugger/lib/utils/nd_path_utility.py
        snpe_sdk_dir = filePath.parents[7]  # raises IndexError if out of bounds
        if (snpe_sdk_dir.match('snpe-*')) or snpe_sdk_dir.match('qaisw-*'):
            return str(snpe_sdk_dir)
        else:
            snpe_path = filePath
            for _ in range(len(filePath.parts)):
                snpe_path = snpe_path.parent
                if (snpe_path.match('snpe-*') or snpe_path.match('qaisw-*')):
                    return str(snpe_path)

            raise InferenceEngineError(get_message('ERROR_INFERENCE_ENGINE_SDK_NOT_FOUND')('SNPE'))
    except IndexError:
        raise InferenceEngineError(get_message('ERROR_INFERENCE_ENGINE_SDK_NOT_FOUND')('SNPE'))


def retrieveQairtSdkDir(filePath=__file__):
    filePath = Path(filePath).resolve()
    try:
        # expected path to this file in the SDK: <QAIRT root>/lib/python/qti/aisw/accuracy_debugger/lib/utils/nd_path_utility.py
        qairt_sdk_dir = filePath.parents[7]  # raises IndexError if out of bounds
        if (qairt_sdk_dir.match('qairt-*')) or qairt_sdk_dir.match('qaisw-*'):
            return str(qairt_sdk_dir)
        else:
            qairt_path = filePath
            for _ in range(len(filePath.parts)):
                qairt_path = qairt_path.parent
                if (qairt_path.match('qairt-*') or qairt_path.match('qaisw-*')):
                    return str(qairt_path)

            raise InferenceEngineError(get_message('ERROR_INFERENCE_ENGINE_SDK_NOT_FOUND')('QAIRT'))
    except IndexError:
        raise InferenceEngineError(get_message('ERROR_INFERENCE_ENGINE_SDK_NOT_FOUND')('QAIRT'))


def retrieveSdkDir(filePath=__file__):
    """Returns SDK path
    :param filePath: Engine path(default is current file path)
    :return: String"""
    filePath = Path(filePath).resolve()
    try:
        # expected path to this file in the SDK: <SDK_ROOT>/lib/python/qti/aisw/accuracy_debugger/lib/utils/nd_path_utility.py
        sdk_dir = filePath.parents[7]  # raises IndexError if out of bounds
        # QAIRT converters/quantizers are present in QNN, SNPE and QAIRT SDKs
        if sdk_dir.match('qaisw-*'):
            return str(sdk_dir)
        else:
            return None
    except IndexError:
        return None


def get_sdk_type(engine_path):
    """Returns SDK type(QNN/SNPE/QAIRT) from given engine_path
    :param engine_path: SDK path
    :return: String"""
    sdk_share_path = os.path.join(engine_path, 'share')
    try:
        share_folders = os.listdir(sdk_share_path)
        if Engine.QNN.value in share_folders and Engine.SNPE.value in share_folders:
            return Engine.QAIRT.value
        elif Engine.QNN.value in share_folders:
            return Engine.QNN.value
        elif Engine.SNPE.value in share_folders:
            return Engine.SNPE.value
        else:
            raise InferenceEngineError(
                f"Failed while fetching SDK type, expected {sdk_share_path} path to have QNN or SNPE folders but found {share_folders}"
            )
    except:
        raise InferenceEngineError(
            f"Failed while fetching SDK type, expected {sdk_share_path} path to have QNN or SNPE folders but found None"
        )


def santize_node_name(node_name):
    """Santize the Node Names to follow Converter Node Naming Conventions
    All Special Characters will be replaced by '_' and numeric node names
    will be modified to have preceding '_'."""
    if not isinstance(node_name, str):
        node_name = str(node_name)
    sanitized_name = re.sub(pattern='\\W+', repl='_', string=node_name)
    if sanitized_name and not sanitized_name[0].isalpha() and sanitized_name[0] != '_':
        sanitized_name = "_" + sanitized_name
    return sanitized_name


def sanitize_output_tensor_files(output_directory):
    """
    Moves output raw files from sub folders to 'output_directory' and sanitizes raw file names
    Example: output_directory/conv/tensor_0.raw will become output_directory/_conv_tensor_0.raw
    param output_directory: path to output raw file directory.
    return: status code 0 for success and -1 for failure.
    """

    if not output_directory or not os.path.isdir(output_directory):
        return -1
    for root, folders, files in os.walk(output_directory):

        if not files:
            continue
        rel_path = os.path.relpath(root, output_directory)
        if rel_path == ".":
            rel_path = ""

        for file in files:
            if not file.endswith(".raw"):
                continue
            new_file = os.path.join(rel_path, file)
            new_file, file_extension = os.path.splitext(new_file)
            new_file = santize_node_name(new_file) + file_extension
            shutil.move(os.path.join(root, file), os.path.join(output_directory, new_file))

    for item in os.listdir(output_directory):
        full_path = os.path.join(output_directory, item)
        if os.path.isdir(full_path):
            shutil.rmtree(full_path)
    return 0
