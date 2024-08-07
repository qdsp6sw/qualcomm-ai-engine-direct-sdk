# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import logging
import os
import traceback
import pandas as pd

from qti.aisw.accuracy_debugger.lib.snooping.snooper_utils import SnooperUtils as su
from qti.aisw.accuracy_debugger.lib.snooping.snooper_utils import show_progress, LayerStatus, files_to_compare
from qti.aisw.accuracy_debugger.lib.utils.nd_framework_utility import read_json
from qti.aisw.accuracy_debugger.lib.utils.nd_path_utility import santize_node_name
from qti.aisw.accuracy_debugger.lib.snooping.QAIRT.nd_qairt_snooper import QAIRTSnooper
from qti.aisw.accuracy_debugger.lib.runner.component_runner import exec_verification


class QAIRTOneshotLayerwiseSnooping(QAIRTSnooper):
    """Class that runs layer wise snooping."""

    def __init__(self, args, logger, verbose="info"):
        super().__init__(snooping_type="layerwise", args=args, logger=logger, verbose=verbose)

    def _exec_verification(self, graph_structure_path: str, inference_result_path: str,
                           golden_output_reference_directory: str, tensor_mapping_file_path: str,
                           dlc_file_path: str):
        verification_args = []
        verification_args += ['--graph_struct', graph_structure_path]
        verification_args += ['--inference_results', inference_result_path]
        verification_args += [
            '--golden_output_reference_directory', golden_output_reference_directory
        ]
        verification_args += ['--tensor_mapping', tensor_mapping_file_path]
        # DLC file is required for axis tracking between golden and target outputs
        verification_args += ['--dlc_path', dlc_file_path]
        for verifier in self._args.default_verifier:
            verification_args += ['--default_verifier', *verifier]
        if self._args.verbose:
            verification_args += ['--verbose']
        verification_args += ['--working_dir', self._args.working_dir]
        verification_args += ['--model_path', self._args.model_path]
        verification_args += ['--framework', self._args.framework]
        if self._args.verifier_config:
            verification_args += ['--verifier_config', self._args.verifier_config]

        exec_verification(args=verification_args, logger=self._logger,
                          run_tensor_inspection=self._args.run_tensor_inspection,
                          validate_args=True)

    def run(self):
        """This method contains the sequence of debugger for
        QAIRTOneshotLayerwiseSnooping."""
        self._logger.info("PROGRESS_ONESHOT_LAYERWISE_STARTED")
        # Execute the framework diagnosis
        if self._args.golden_output_reference_directory is None:
            if self._args.framework == 'pytorch':
                self.execute_on_qairt(
                    model_path=self._args.model_path, input_list=self._args.input_list,
                    orig_model_outputs=self._args.output_tensor,
                    input_tensors=self._args.input_tensor, output_dirname="reference_run",
                    intermediate_outputs=True, runtime="cpu",
                    architecture="x86_64-linux-clang", precision="fp32")
                self._args.golden_output_reference_directory = os.path.join(
                    self._args.output_dir, 'inference_engine', 'reference_run', 'output',
                    'Result_0')
            else:
                self.trigger_framework_runner()

        # Now execute inference engine
        self.execute_on_qairt(model_path=self._args.model_path, input_list=self._args.input_list,
                              calib_input_list=self._args.calibration_input_list,
                              orig_model_outputs=self._args.output_tensor,
                              input_tensors=self._args.input_tensor, output_dirname="target_run",
                              intermediate_outputs=True,
                              quantization_overrides=self._args.quantization_overrides)

        # Now execute verification
        graph_structure_path = os.path.join(self._args.output_dir, "inference_engine", "target_run",
                                            "model_graph_struct.json")
        inference_result_path = os.path.join(self._args.output_dir, "inference_engine",
                                             "target_run", 'output', 'Result_0')
        tensor_mapping_file_path = os.path.join(self._args.output_dir, "inference_engine",
                                                "target_run", "tensor_mapping.json")
        dlc_file_path = os.path.join(self._args.output_dir, "inference_engine",
                                                "target_run", "base.dlc")
        self._exec_verification(
            graph_structure_path=graph_structure_path, inference_result_path=inference_result_path,
            golden_output_reference_directory=self._args.golden_output_reference_directory,
            tensor_mapping_file_path=tensor_mapping_file_path, dlc_file_path=dlc_file_path)
        self._logger.info("PROGRESS_ONESHOT_LAYERWISE_FINISHED")
