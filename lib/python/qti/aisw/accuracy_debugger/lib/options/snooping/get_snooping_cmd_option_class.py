# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

from qti.aisw.accuracy_debugger.lib.options.snooping.qairt_binary_snooping_cmd_options import QAIRTBinarySnoopingCmdOptions
from qti.aisw.accuracy_debugger.lib.options.snooping.qairt_layerwise_cmd_options import QAIRTLayerwiseSnoopingCmdOptions
from qti.aisw.accuracy_debugger.lib.options.snooping.qairt_cumulative_layerwise_cmd_options import QAIRTCumulativeLayerwiseSnoopingCmdOptions
from qti.aisw.accuracy_debugger.lib.options.snooping.qairt_oneshot_layerwise_cmd_options import QAIRTOneshotLayerwiseSnoopingCmdOptions


def get_snooping_cmd_option_class(snooper: str):
    if snooper == "binary":
        return QAIRTBinarySnoopingCmdOptions
    elif snooper == "layerwise":
        return QAIRTLayerwiseSnoopingCmdOptions
    elif snooper == "cumulative-layerwise":
        return QAIRTCumulativeLayerwiseSnoopingCmdOptions
    elif snooper == "oneshot-layerwise":
        return QAIRTOneshotLayerwiseSnoopingCmdOptions
