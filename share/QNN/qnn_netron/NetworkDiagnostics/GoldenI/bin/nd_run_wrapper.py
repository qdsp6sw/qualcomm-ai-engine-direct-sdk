# =============================================================================
#
#  Copyright (c) 2019-2021 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../'))
from lib.wrapper.nd_tool_setup import ToolConfig
from lib.utils.nd_constants import Engine
from lib.options.wrapper_cmd_options import WrapperCmdOptions

def main():
    # since we spawn a copy subprocess with a copy of our path, we have to manually modify the
    # env value
    os.environ["PYTHONPATH"] = os.pathsep.join([os.path.dirname(__file__)]
                                               + os.environ.get("PYTHONPATH", "").split(os.pathsep))

    system_args = sys.argv[1:]
    args = WrapperCmdOptions(system_args).parse()

    config = ToolConfig()

    # configs and spawns run_framework_diagnosis sub-process
    ret_framework_diagnosis = config.run_framework_diagnosis(list(system_args))
    if ret_framework_diagnosis != 0:
        exit(ret_framework_diagnosis)

    inference_args = list(system_args)
    model_name = os.path.basename(os.path.splitext(args.model_path)[0])
    inference_args.extend(['--model_name', model_name])
    #replace --engine args to avoid ambiguity error
    if '--engine' in inference_args: inference_args[inference_args.index('--engine')] = '-e'

    # configs and spawns run_inference_engine sub-process
    if args.engine == Engine.QNN.value:
        ret_inference_engine = config.run_qnn_inference_engine(inference_args)
    else:
        ret_inference_engine = config.run_snpe_inference_engine(inference_args)
    if ret_inference_engine != 0:
        exit(ret_inference_engine)

    # configs and spawns run_verification sub-process
    verification_args = list(system_args)
    graph_structure = model_name + '_graph_struct.json'
    graph_structure_path = os.path.join(args.working_dir, 'inference_engine', 'latest',
                                        graph_structure)
    verification_args.extend(['--graph_struct', graph_structure_path])

    verification_args.extend(['--inference_results', os.path.join(args.working_dir,
                                                'inference_engine', 'latest','output/Result_0')])

    verification_args.extend(['--framework_results', os.path.join(args.working_dir,
                                                    'framework_diagnosis', 'latest'),
                              '--tensor_mapping', os.path.join(args.working_dir,
                                                    'inference_engine', 'latest', 'tensor_mapping.json')])

    ret_verifier = config.run_verifier(verification_args)
    if ret_verifier != 0:
        exit(ret_verifier)

    if args.engine == Engine.QNN.value and args.deep_analyzer:
        # configs and spawns run_acuracy_deep_analyzer sub-process
        da_param_index = system_args.index('--deep_analyzer')
        deep_analyzers = system_args[da_param_index+1].split(',')
        del system_args[da_param_index:da_param_index+2]
        deep_analyzer_args = list(system_args)
        deep_analyzer_args.extend(['--tensor_mapping', os.path.join(args.working_dir,
                                                        'inference_engine', 'latest', 'tensor_mapping.json'),
                                    '--inference_results', os.path.join(args.working_dir,
                                                        'inference_engine', 'latest','output/Result_0'),
                                    '--graph_struct', graph_structure_path,
                                    '--framework_results', os.path.join(args.working_dir,
                                                                'framework_diagnosis', 'latest'),
                                    '--result_csv', os.path.join(args.working_dir,
                                                        'verification', 'latest','summary.csv')
                                                        ])
        for d_analyzer in deep_analyzers:
            ret_deep_analyzer = config.run_accuracy_deep_analyzer(deep_analyzer_args + ['--deep_analyzer',d_analyzer])
            if ret_deep_analyzer != 0:
                exit(ret_deep_analyzer)


if __name__ == '__main__':
    main()
