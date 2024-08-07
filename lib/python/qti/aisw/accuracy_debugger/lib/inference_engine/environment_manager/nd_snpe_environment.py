# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import sys
import os

from qti.aisw.accuracy_debugger.lib.inference_engine.environment_manager.nd_environment import Environment
from qti.aisw.accuracy_debugger.lib.utils.nd_path_utility import retrieveSnpeSdkDir


class SNPEEnvironment(Environment):

    def __init__(self, engine, host_device, output_directory, engine_path, logger) -> None:
        super().__init__(engine, host_device, output_directory, logger)
        if engine_path is None:
            self.engine_path = retrieveSnpeSdkDir(self._logger)
        else:
            self.engine_path = engine_path
