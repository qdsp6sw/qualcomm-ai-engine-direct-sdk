# =============================================================================
#
#  Copyright (c) 2021 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
import sys
import traceback
import os

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../'))
from lib.utils.nd_exceptions import DeepAnalyzerError
from lib.utils.nd_logger import setup_logger
from lib.utils.nd_symlink import symlink
from lib.deep_analyzer.nd_deep_analyzer import DeepAnalyzer
from lib.options.accuracy_deep_analyzer_cmd_options import AccuracyDeepAnalyzerCmdOptions
from lib.inference_engine.nd_get_tensor_mapping import TensorMapping
from lib.utils.nd_namespace import Namespace

def exec_deep_analyzer():
    args = AccuracyDeepAnalyzerCmdOptions(sys.argv[1:]).parse()
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
    logger = setup_logger(args.verbose, args.output_dir)

    symlink('latest', args.output_dir, logger)

    try:
        if not args.tensor_mapping:
            logger.warn("--tensor_mapping is not set, a tensor_mapping will be generated based on user input.")
            get_mapping_arg = Namespace(None, framework=args.framework,
                                        version=args.framework_version, model_path=args.model_path,
                                        output_dir=args.inference_results, engine=args.engine,
                                        golden_dir_for_mapping=args.framework_results)
            args.tensor_mapping = TensorMapping(get_mapping_arg, logger)
        deep_analyzer = DeepAnalyzer(args, logger)
        deep_analyzer.analyze()
        logger.info("Successfully ran deep_analyzer!")
    except DeepAnalyzerError as excinfo:
        raise DeepAnalyzerError("deep analyzer failed: {}".format(str(excinfo)))
    except Exception as excinfo:
        traceback.print_exc()
        raise Exception("Encountered error: {}".format(str(excinfo)))


if __name__ == '__main__':
    exec_deep_analyzer()
