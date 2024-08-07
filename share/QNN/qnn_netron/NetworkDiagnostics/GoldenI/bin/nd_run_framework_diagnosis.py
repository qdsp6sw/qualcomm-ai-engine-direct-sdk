# =============================================================================
#
#  Copyright (c) 2019-2021 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
import sys
import traceback
from typing import List
import os

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../'))
from lib.framework_diagnosis.nd_framework_runner import FrameworkRunner
from lib.utils.nd_exceptions import FrameworkError
from lib.utils.nd_logger import setup_logger
from lib.utils.nd_symlink import symlink
from lib.options.framework_diagnosis_cmd_options import FrameworkDiagnosisCmdOptions

def exec_framework_runner():
    args = FrameworkDiagnosisCmdOptions(sys.argv[1:]).parse()
    logger = setup_logger(args.verbose, args.output_dir)

    symlink('latest', args.output_dir, logger)

    try:
        framework_runner = FrameworkRunner(logger, args)
        framework_runner.run()
    except FrameworkError as e:
        raise FrameworkError("Conversion failed: {}".format(str(e)))
    except Exception as e:
        traceback.print_exc()
        raise Exception("Encountered Error: {}".format(str(e)))


if __name__ == '__main__':
    exec_framework_runner()
