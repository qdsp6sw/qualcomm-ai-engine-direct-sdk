# =============================================================================
#
#  Copyright (c) 2021 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
import argparse
import os
import sys
import traceback
from datetime import datetime
import numpy as np
import csv
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../'))
from lib.utils.nd_logger import setup_logger
from lib.utils.nd_symlink import symlink
from lib.visualizer.nd_histogram_visualizer import HistogramVisualizer
from lib.visualizer.nd_diff_visualizer import DiffVisualizer
from lib.utils.nd_path_utility import get_absolute_path

dtype_map = {
    "int8": np.int8,
    "uint8": np.uint8,
    "int16": np.int16,
    "uint16": np.uint16,
    "float32": np.float32,
}

def _parse_args(args):  # type:(List[str]) -> argparse.Namespace
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Script to inspection tensor."
    )

    # Workaround to list required arguments before optional arguments
    parser._action_groups.pop()

    required = parser.add_argument_group('required arguments')

    required.add_argument('--framework_result', type=str, required=True,
                          help="Path of framework engine tensor(golden tensor). "
                               "Paths may be absolute, or relative to the working directory.")
    required.add_argument('--inference_result', type=str, required=True,
                          help="Path of inference engine tensor. "
                               "Paths may be absolute, or relative to the working directory.")


    optional = parser.add_argument_group('optional arguments')

    optional.add_argument('-w', '--working_dir', type=str, required=False,
                          default='working_directory',
                          help='Working directory for the inference engine to '
                               'store temporary files.'
                               'Creates a new directory if the specified working directory does '
                               'not exist')
    optional.add_argument('--data_type', type=str, default="float32",
                          choices=['int8', 'uint8', 'int16', 'uint16', 'float32'],
                          help="DataType of the output tensor.")
    optional.add_argument('--tensor_dims', type=str,
                          help="The dimension of tensor like 1,256,256,32."
                               "The rank of dimension must be 4.")
    optional.add_argument('-v', '--verbose', action="store_true", default=False,
                          help="Verbose printing")

    parsed_args, _ = parser.parse_known_args(args)

    parsed_args.working_dir = os.path.join(os.getcwd(), parsed_args.working_dir)
    parsed_args.output_dir = os.path.join(parsed_args.working_dir, "inspection/",
                                          datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

    parsed_args.framework_result = get_absolute_path(parsed_args.framework_result)
    parsed_args.inference_result = get_absolute_path(parsed_args.inference_result)

    return parsed_args


def inspection_tensor():
    args = _parse_args(sys.argv[1:])
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
    logger = setup_logger(args.verbose, args.output_dir)

    try:
        framework_result = np.fromfile(args.framework_result, dtype=dtype_map[args.data_type])
        inference_result = np.fromfile(args.inference_result, dtype=dtype_map[args.data_type])
        if framework_result.shape != inference_result.shape:
            logger.error("The length of framework result and inference result are different.")
            return
        if args.tensor_dims:
            shape = np.array(args.tensor_dims.split(",")).astype(int)
            if len(shape) != 4:
                logger.error("The tensor dimension rank must be 4.")
                return
            framework_result = framework_result.reshape(shape)
            inference_result = inference_result.reshape(shape)
        else:
            framework_result = framework_result.reshape([1,1,1,-1])
            inference_result = inference_result.reshape([1,1,1,-1])

        logger.debug('The shape of tensor {}'.format(inference_result.shape))
        for batch in range(framework_result.shape[0]):
            logger.debug('Try to inspection tensor in batch {}'.format(batch))
            np.savetxt(os.path.join(args.output_dir, "golden_data_batch" + str(batch) + ".csv"), framework_result[batch,:].flatten())
            np.savetxt(os.path.join(args.output_dir, "inference_data_batch" + str(batch) + ".csv"), inference_result[batch,:].flatten())

            dump_data = pd.DataFrame([])
            save_path = os.path.join(args.output_dir, "statistics_batch" + str(batch))
            HistogramVisualizer.visualize(framework_result[batch,:].flatten(), \
                                          inference_result[batch,:].flatten(), \
                                          save_path + ".png")
            DiffVisualizer.visualize(framework_result[batch,:].flatten(), \
                                     inference_result[batch,:].flatten(), \
                                     os.path.join(args.output_dir, "diff_between_two_batch" + str(batch) + ".png"))

            if not args.tensor_dims:
                continue

            for depth in range(framework_result.shape[3]):
                logger.debug('Try to inspection tensor depth {} in batch {}'.format(depth, batch))
                golden_data = framework_result[batch,:,:,depth].flatten()
                inference_data = inference_result[batch,:,:,depth].flatten()
                dump_data.insert(dump_data.shape[1], 'golden_depth_' + str(depth), golden_data)
                dump_data.insert(dump_data.shape[1], 'inf_depth_' + str(depth), inference_data)
                HistogramVisualizer.visualize(golden_data,                                            \
                                              inference_data,                                         \
                                              save_path + "_depth" + str(depth) +".png")
                DiffVisualizer.visualize(golden_data, \
                                         inference_data, \
                                         os.path.join(args.output_dir, "diff_between_two_batch" + str(batch) + "_depth" + str(depth)+".png"))

            dump_data.to_csv(os.path.join(args.output_dir, "inspect_batch" + str(batch) + ".csv"))
        symlink('latest', args.output_dir, logger)
        logger.info("Successfully ran inspection tensor!")
    except Exception as excinfo:
        traceback.print_exc()
        raise Exception("Encountered error: {}".format(str(excinfo)))


if __name__ == '__main__':
    inspection_tensor()
