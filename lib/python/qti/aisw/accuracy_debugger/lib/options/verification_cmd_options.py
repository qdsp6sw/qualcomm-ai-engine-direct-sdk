# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import argparse

from qti.aisw.accuracy_debugger.lib.options.cmd_options import CmdOptions
from qti.aisw.accuracy_debugger.lib.utils.nd_exceptions import ParameterError
from qti.aisw.accuracy_debugger.lib.utils.nd_path_utility import get_absolute_path
from qti.aisw.accuracy_debugger.lib.utils.nd_framework_utility import get_framework_info
from qti.aisw.accuracy_debugger.lib.utils.nd_constants import Framework
from qti.aisw.accuracy_debugger.lib.utils.nd_constants import Engine


class VerificationCmdOptions(CmdOptions):

    def __init__(self, args, validate_args=True):
        super().__init__('verification', args, validate_args=validate_args)

    def initialize(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                              description="Script to run verification.")

        # Workaround to list required arguments before optional arguments
        self.parser._action_groups.pop()

        required = self.parser.add_argument_group('required arguments')

        # default_verifier will verify in nd_verifier_factory.py
        required.add_argument(
            '--default_verifier', type=str.lower, required=True, nargs='+', action="append",
            help='Default verifier used for verification. The options '
            '"RtolAtol", "AdjustedRtolAtol", "TopK", "L1Error", "CosineSimilarity", "MSE", "MAE", "SQNR", "ScaledDiff" are supported. '
            'An optional list of hyperparameters can be appended. For example: --default_verifier rtolatol,rtolmargin,0.01,atolmargin,0.01 '
            'An optional list of placeholders can be appended. For example: --default_verifier CosineSimilarity param1 1 param2 2. '
            'to use multiple verifiers, add additional --default_verifier CosineSimilarity')
        required.add_argument(
            '--golden_output_reference_directory', '--framework_results',
            dest='golden_output_reference_directory', type=str, required=True,
            help="Path to root directory generated from . "
            "Paths may be absolute, or relative to the working directory.")
        required.add_argument(
            '--inference_results', type=str, required=True,
            help="Path to root directory generated from inference engine diagnosis. "
            "Paths may be absolute, or relative to the working directory.")

        optional = self.parser.add_argument_group('optional arguments')

        optional.add_argument(
            '--tensor_mapping', type=str, required=False, default=None,
            help='Path to the file describing the tensor name mapping '
            'between inference and golden tensors.'
            'can be generated with nd_run_{engine}_inference_engine')
        optional.add_argument(
            '--qnn_model_json_path', type=str, required=False, help=
            "Path to the qnn model net json, used for transforming axis of golden outputs w.r.t to qnn outputs.\
                Note: Applicable only for QNN"
        )
        optional.add_argument(
            '--dlc_path', type=str, required=False, default=None,
            help= "Path to the dlc file, used for transforming axis of golden outputs w.r.t to target outputs.\
                Note: Applicable for QAIRT/SNPE"
        )
        optional.add_argument('--verifier_config', type=str, default=None,
                              help='Path to the verifiers\' config file')
        optional.add_argument(
            '--graph_struct', type=str, default=None, help=
            'Path to the inference graph structure .json file. This file aids in providing structure related information of the converted model graph during this stage.'
        )
        optional.add_argument('-v', '--verbose', action="store_true", default=False,
                              help="Verbose printing")
        optional.add_argument('-w', '--working_dir', type=str, required=False,
                                default='working_directory',
                                help='Working directory for the {} to store temporary files. '.format(self.component) + \
                                    'Creates a new directory if the specified working directory does not exist')
        optional.add_argument('--output_dirname', type=str, required=False,
                                default='<curr_date_time>',
                                help='output directory name for the {} to store temporary files under <working_dir>/{}. '.format(self.component, self.component) + \
                                    'Creates a new directory if the specified working directory does not exist')
        optional.add_argument(
            '--args_config', type=str, required=False,
            help="Path to a config file with arguments.  This can be used to feed arguments to "
            "the AccuracyDebugger as an alternative to supplying them on the command line.")
        optional.add_argument('--target_encodings', type=str, required=False, default=None,
                              help="Path to target encodings json file.")

        tensor_mapping = self.parser.add_argument_group(
            'arguments for generating a new tensor_mapping.json')

        tensor_mapping.add_argument('-m', '--model_path', type=str, required=False,
                                    help='path to original model for tensor_mapping uses here.')
        tensor_mapping.add_argument(
            '-e', '--engine', nargs='+', type=str, required=False, default=None,
            metavar=('ENGINE_NAME', 'ENGINE_VERSION'),
            help='Name of engine(qnn/snpe) that is used for running inference, '
            'optionally followed by the engine version. Used here for tensor_mapping.')
        tensor_mapping.add_argument(
            '-f', '--framework', nargs='+', type=str.lower, default=None, required=False,
            help='Framework type and version, version is optional. '
            'Currently supported frameworks are [' + ', '.join([f.value for f in Framework]) + ']. '
            'For example, tensorflow 2.10.1 ')
        self.initialized = True

    def verify_update_parsed_args(self, parsed_args):
        supported_verifiers = [
            "rtolatol", "adjustedrtolatol", "topk", "l1error", "cosinesimilarity", "mse", "mae",
            "sqnr", "scaleddiff"
        ]
        for verifier in parsed_args.default_verifier:
            verifier_name = verifier[0].split(',')[0]
            if verifier_name not in supported_verifiers:
                raise ParameterError(
                    f"--default_verifier '{verifier_name}' is not a supported verifier.")

        parsed_args.verify_types = parsed_args.default_verifier
        parsed_args.golden_output_reference_directory = get_absolute_path(
            parsed_args.golden_output_reference_directory)
        parsed_args.inference_results = get_absolute_path(parsed_args.inference_results)
        parsed_args.tensor_mapping = get_absolute_path(parsed_args.tensor_mapping)
        parsed_args.verifier_config = get_absolute_path(parsed_args.verifier_config)
        parsed_args.graph_struct = get_absolute_path(parsed_args.graph_struct)
        parsed_args.qnn_model_json_path = get_absolute_path(parsed_args.qnn_model_json_path)
        parsed_args.dlc_path = get_absolute_path(parsed_args.dlc_path)

        if parsed_args.framework is None:
            parsed_args.framework = [get_framework_info(parsed_args.model_path)]

        #get framework and framework version
        parsed_args.framework_version = None
        if parsed_args.framework is not None:
            if len(parsed_args.framework) > 2:
                raise ParameterError("Maximum two arguments required for framework.")
            elif len(parsed_args.framework) == 2:
                parsed_args.framework_version = parsed_args.framework[1]
            parsed_args.framework = parsed_args.framework[0]

        #get engine and engine version
        parsed_args.engine_version = None
        if parsed_args.engine is not None:
            if len(parsed_args.engine) > 2:
                raise ParameterError("Maximum two arguments required for inference engine.")
            elif len(parsed_args.engine) == 2:
                parsed_args.engine_version = parsed_args.engine[1]
            parsed_args.engine = parsed_args.engine[0]

        if parsed_args.qnn_model_json_path and parsed_args.dlc_path:
            raise ParameterError("Cannot pass both --qnn_model_json_path and --dlc_path.")

        return parsed_args

    def get_all_associated_parsers(self):
        return [self.parser]
