# =============================================================================
#
#  Copyright (c) 2019-2021 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../'))
from lib.utils.nd_constants import Engine
from nd_execute_inference_engine import *


if __name__ == '__main__':
    exec_inference_engine(Engine.QNN.value, sys.argv[1:])
