# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

from qti.aisw.accuracy_debugger.lib.utils.nd_path_utility import get_absolute_path, format_args
from qti.aisw.accuracy_debugger.lib.options.snooping.qairt_snooping_cmd_options import QAIRTSnoopingCmdOptions
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


class QAIRTBinarySnoopingCmdOptions(QAIRTSnoopingCmdOptions):

    def __init__(self, args, validate_args=True):
        super().__init__(args=args, type="binary", validate_args=validate_args)

    def initialize(self):
        """
        type: (List[str]) -> argparse.Namespace

        :param args: User inputs, fed in as a list of strings
        :return: Namespace object
        """
        self.parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                              description="Script to run snooping.")

        optional_args = self.parser.add_argument_group('Required Arguments')
        optional_args.add_argument('--min_graph_size', default=16, type=int, required=False,
                                   help='Provide the minimum subgraph size')
        optional_args.add_argument(
            '--subgraph_relative_weight', default=0.4, type=float, required=False,
            help='Helps in deciding whether a sub graph is further debugged or not. '
            'If a subgraph scores > 40 percent of the aggreagte score of two subgraphs, we investage '
            'the subgraph further.')
        optional_args.add_argument('--verifier', type=str.lower, required=False, default='mse',
                                   help='Choose verifer among [sqnr, mse] for the comparison')
        self._base_initialize()

        self.initialized = True

    def verify_update_parsed_args(self, parsed_args):
        parsed_args = self._verify_update_base_parsed_args(parsed_args)
        return parsed_args
