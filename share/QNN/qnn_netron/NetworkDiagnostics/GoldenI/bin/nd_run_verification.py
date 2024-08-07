# =============================================================================
#
#  Copyright (c) 2019-2021 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
import os
import sys
import traceback
import pandas as pd
from datetime import datetime
from typing import List

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../'))
from lib.utils.nd_errors import get_message
from lib.utils.nd_exceptions import VerifierError
from lib.utils.nd_logger import setup_logger
from lib.utils.nd_symlink import symlink
from lib.verifier.nd_verification import Verification
from lib.options.verification_cmd_options import VerificationCmdOptions
from lib.inference_engine.nd_get_tensor_mapping import TensorMapping
from lib.utils.nd_namespace import Namespace

def exec_verification():
    args = VerificationCmdOptions(sys.argv[1:]).parse()
    logger = setup_logger(args.verbose, args.output_dir)

    try:
        if not args.tensor_mapping:
            logger.warn("--tensor_mapping is not set, a tensor_mapping will be generated based on user input.")
            get_mapping_arg = Namespace(None, framework=args.framework,
                                        version=args.framework_version, model_path=args.model_path,
                                        output_dir=args.inference_results, engine=args.engine,
                                        golden_dir_for_mapping=args.framework_results)
            args.tensor_mapping = TensorMapping(get_mapping_arg, logger)

        verify_results = []
        for verifier in args.verify_types:
            verify_type = verifier[0]
            verifier_configs = verifier[1:]
            verification = Verification(verify_type, logger, args, verifier_configs)
            if verification.has_specific_verifier() and len(args.verify_types) > 1:
                raise VerifierError(get_message('ERROR_VERIFIER_USE_MULTI_VERIFY_AND_CONFIG'))
            verify_result = verification.verify_tensors()
            verify_result = verify_result.drop(columns=['Units', 'Verifier'])
            verify_result = verify_result.rename(columns={'Metric':verify_type})
            verify_results.append(verify_result)

        # if args.verifier_config is None, all tensors use the same verifer. So we can export Summary
        if args.verifier_config == None:
            summary_df = verify_results[0]
            for verify_result in verify_results[1:]:
                summary_df = pd.merge(summary_df, verify_result, on=['Name', 'LayerType', 'Size', 'Tensor_dims'])
            summary_df.to_csv(os.path.join(args.output_dir, Verification.SUMMARY_NAME + ".csv"), index=False, encoding="utf-8")
            summary_df.to_html(os.path.join(args.output_dir, Verification.SUMMARY_NAME + ".html"), index=False, classes='table')
        symlink('latest', args.output_dir, logger)
        logger.info("Successfully ran verification!")
    except VerifierError as excinfo:
        raise Exception("Verification failed: {}".format(str(excinfo)))
    except Exception as excinfo:
        traceback.print_exc()
        raise Exception("Encountered error: {}".format(str(excinfo)))


if __name__ == '__main__':
    exec_verification()
