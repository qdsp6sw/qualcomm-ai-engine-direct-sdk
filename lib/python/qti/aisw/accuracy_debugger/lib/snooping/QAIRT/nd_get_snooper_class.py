# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

from qti.aisw.accuracy_debugger.lib.snooping.QAIRT.nd_qairt_binary_snooping import QAIRTBinarySnooping
from qti.aisw.accuracy_debugger.lib.snooping.QAIRT.nd_qairt_layerwise_snooping import QAIRTLayerwiseSnooping
from qti.aisw.accuracy_debugger.lib.snooping.QAIRT.nd_qairt_cumulative_layerwise_snooping import QAIRTCumulativeLayerwiseSnooping
from qti.aisw.accuracy_debugger.lib.snooping.QAIRT.nd_qairt_oneshot_layerwise_snooping import QAIRTOneshotLayerwiseSnooping


def get_snooper_class(type: str):
    if type == "binary":
        return QAIRTBinarySnooping
    elif type == "layerwise":
        return QAIRTLayerwiseSnooping
    elif type == "cumulative-layerwise":
        return QAIRTCumulativeLayerwiseSnooping
    elif type == "oneshot-layerwise":
        return QAIRTOneshotLayerwiseSnooping
