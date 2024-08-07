# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

from qti.aisw.accuracy_debugger.lib.utils.nd_constants import Engine
from qti.aisw.accuracy_debugger.lib.inference_engine.environment_manager.nd_qnn_environment import QNNEnvironment
from qti.aisw.accuracy_debugger.lib.inference_engine.environment_manager.nd_snpe_environment import SNPEEnvironment
from qti.aisw.accuracy_debugger.lib.inference_engine.environment_manager.nd_qairt_environment import QAIRTEnvironment


def get_environment_cls(engine: Engine):
    if engine.value == Engine.QNN.value:
        return QNNEnvironment
    elif engine.value == Engine.SNPE.value:
        return SNPEEnvironment
    elif engine.value == Engine.QAIRT.value:
        return QAIRTEnvironment
