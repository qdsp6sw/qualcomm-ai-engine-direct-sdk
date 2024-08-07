# ==============================================================================
#
#  Copyright (c) 2020-2024 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
# ==============================================================================
import json
import logging
import os
import time
import re
import sys
import tempfile
import traceback
import itertools
import math
from typing import Callable, List, Tuple, Union
from tqdm import tqdm

from qti.aisw.converters.common import ir_graph as ir_graph_lib

IrGraph = ir_graph_lib.IrGraph
logger = logging.getLogger('AIMET Quantizer')

try:
    import numpy as np
    import torch
    from aimet_common.defs import QuantizationDataType, QuantScheme
    from aimet_common.quantsim_config.config_utils import \
        get_path_for_target_config
    from aimet_common.pro.defs import CallbackFunc
    from aimet_torch import utils
    from aimet_torch.cross_layer_equalization import equalize_model
    from aimet_torch.adaround.adaround_weight import Adaround, AdaroundParameters
    from aimet_torch.auto_quant import _QuantSchemePair
    from aimet_torch.pro.model_preparer import prepare_model_from_ir_graph
    from aimet_torch.pro.quantsim import QuantizationSimModel
    from aimet_torch.pro.quant_sim.dlc_quantsim_exporter import DlcQuantSimExporter
    from aimet_torch.amp.mixed_precision_algo import EvalCallbackFactory
    from aimet_torch.mixed_precision import choose_mixed_precision
    from torch.utils.data import DataLoader, Dataset
    from qti.aisw.converters.aimet.config_argparse import AimetConfigParser
    from aimet_torch.pro.auto_quant import AutoQuant
except ModuleNotFoundError as e:
    traceback.print_exc()
    logger.error("Unable to import required modules to run AIMET Quantizer. "
                 "'--use_aimet_quantizer' option can't be used")
    sys.exit()


# Mapping from QNN calibration to AIMET QuantScheme
QNN_TO_AIMET_QUANT_SCHEME_MAP = {
    "tf": QuantScheme.post_training_tf,
    "min-max": QuantScheme.post_training_tf,
    "enhanced": QuantScheme.post_training_tf_enhanced,
    "sqnr": QuantScheme.post_training_tf_enhanced,
    "percentile": QuantScheme.post_training_percentile,
    "symmetric": QuantScheme.post_training_tf,
}


def _replace_invalid_chars_for_variable(name: str) -> str:
    """
    Replace invalid chars such as dot, slash, ...
    :param name: name to replace invalid chars
    :return: cleansed string can be used Python variable name
    """
    if not name.isidentifier():
        found_list = re.findall(r"\w+", name)
        if found_list:
            name = "_".join(found_list)
        if name[0].isdigit():
            name = '_' + name
        if not name.isidentifier():
            error_str = f"Unable to produce a valid model name name from {name}"
            logger.error(error_str)
            raise RuntimeError(error_str)
    return name


class InputListDataset(Dataset):

    def __init__(self, input_list_path: str, input_shapes: dict, input_dtypes: dict, use_native_input_files: bool) -> None:
        super().__init__()
        self._input_list = open(input_list_path).readlines()
        self._input_list = [x for x in self._input_list if x != '\n']
        self._input_shapes = input_shapes
        self._is_input_list_formatted = True
        self._use_native_input_files = use_native_input_files
        # Creates a list of ordered input file list for each input based on input ordering in Ir graph
        # This ordering is inferred from the input_shapes dict as dictionaries are ordered in python>=3.7
        self._ordered_input_list = list(map(self._order_input_files, self._input_list))

        self._input_dtypes = input_dtypes

    def __len__(self):
        return len(self._input_list)

    def _read_raw_data(self, file_name: str, dtype: str) -> np.ndarray:
        """ Read data from the .raw files into a numpy array """
        with open(file_name, "rb") as file:
            raw_data = file.read()
            if self._use_native_input_files:
                numpy_from_raw_data = np.frombuffer(raw_data, dtype=dtype)
            else:
                numpy_from_raw_data = np.frombuffer(raw_data, dtype=np.float32)
                numpy_from_raw_data = numpy_from_raw_data.astype(dtype, copy=False)
            return numpy_from_raw_data

    def _order_input_files(self, input_files):
        """ Order input files based on IR graph input name(s) """
        input_files = input_files.split() # Inputs separated by space
        is_formatted = [':=' in x for x in input_files]
        assert all(is_formatted) or not any(is_formatted), ("Input list is not well formatted")
        if all(is_formatted):
            input_files_dict = {y[0]:y[1] for y in [x.split(':=') for x in input_files]}
            input_files = [input_files_dict[input_name] for input_name in self._input_shapes.keys()]
        else:
            # Print warning message only once
            if len(input_files) > 1 and self._is_input_list_formatted:
                self._is_input_list_formatted = False
                logger.warning("Input list is not properly formatted, may result in errors. "
                               "Write input list with input_name appended before file path "
                               "for each input, input_name:=<filepath> ..")
        return input_files

    def __getitem__(self, index) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Get the input tensor(s) at a given index of the input_list file
        Input files can be specified with the below format for three sets of inputs for two input layers

        Input_1:=Placeholder_1/real_input_inputs_1/0-0#67c965.rawtensor Input_2:=Placeholder_1/real_input_inputs_1/1-1#54f1ff.rawtensor
        Input_1:=Placeholder_1/real_input_inputs_1/1-0#b42dc6.rawtensor Input_2:=Placeholder_1/real_input_inputs_1/2-1#346a0e.rawtensor
        Input_1:=Placeholder_1/real_input_inputs_1/2-0#e6fb51.rawtensor Input_2:=Placeholder_1/real_input_inputs_1/0-1#8a171b.rawtensor
        """
        ordered_input_files = self._ordered_input_list[index]

        tensors: List[torch.Tensor] = []
        for n, file_name in enumerate(ordered_input_files):
            tensor_name, tensor_dim = list(self._input_shapes.items())[n]
            tensor_dtype=self._input_dtypes[tensor_name]
            raw_data_numpy_array = self._read_raw_data(file_name, dtype=tensor_dtype)
            assert raw_data_numpy_array.shape[0] == np.prod(tensor_dim), (f'Could not reshape input tensor "{tensor_name}" '
                                                                          f'as required. Raw Numpy Data Shape: {raw_data_numpy_array.shape}; '
                                                                          f'Required tensor shape: {tensor_dim}')
            reshaped_numpy_array = raw_data_numpy_array.reshape(tensor_dim)
            tensor = torch.tensor(reshaped_numpy_array)
            tensors.append(tensor)

        if len(tensors) == 1:
            return tensors[0]
        return tuple(tensors)


class QnnToAimetAdapter:

    def __init__(self, ir_graph, opts, datadir: str = "", use_cuda: bool = True):
        self.opts = opts
        # Get activation and param quantizer schemes
        self.param_quant_scheme, self.act_quant_scheme = self._get_act_param_quant_schemes()
        self._valid_opts = self._validate_opts()
        if not self._valid_opts:
            logger.warning("Invalid argument provided. '--use_aimet_quantizer' option can't be used")
        else:
            self._device = self._set_device(use_cuda)

            # Take input shapes from IR Graph
            self._input_shapes = {input_tensor.name():input_tensor.dims() for input_tensor in ir_graph.get_input_tensors_to_graph()}

            if not self._input_shapes:
                logger.error("Could not infer model input shapes from the model. Please specify --input_dim")
                self._valid_opts = False
            else:
                self._input_dtypes = self._infer_input_dtypes(ir_graph)
                if datadir:
                    self._datadir = datadir
                else:
                    self._temp_datadir = tempfile.TemporaryDirectory()
                    self._datadir = self._temp_datadir.name

                self.aimet_argparse = AimetConfigParser(self.opts.config, self.opts.algorithms) if self.opts.config \
                    else None

                self._input_list = None
                self._input_list_dataset = None
                self._input_list_dataloader = None
                self._config_dataloader = None
                if self.aimet_argparse is not None:
                    self._set_config_dataloader()
                elif self.opts.input_list:
                    self._input_list = self._get_input_list_abs()
                    self._input_list_dataset = InputListDataset(self._input_list, self._input_shapes, self._input_dtypes, self.opts.use_native_input_files)
                    # While iterating through the dataloader, the tensors have an additional pseudo dimensions (added by Torch DataLoader class).
                    # So, while using it, we squeeze the additional pseudo-dimension. For example, if the input_tensor has dimensions [1, 224, 224, 3],
                    # while iterating through the dataloader, it will be [1, 1, 224, 224, 3]. So, we squeeze the pseudo-dimension by using input_tensor[0].
                    self._input_list_dataloader = DataLoader(self._input_list_dataset, batch_size=1, shuffle=False)

                self._filename = 'input_model'

                if os.path.isfile(self.opts.input_network):
                    self._filename, _ = os.path.splitext(os.path.basename(self.opts.input_network))

                self._model_name = _replace_invalid_chars_for_variable(self._filename + "_model") # Same as filename +'_model' for simplicity
                if self.opts.output_path is not None:
                    self._output_name, _ = os.path.splitext(os.path.basename(self.opts.output_path))
                    self.output_dir = os.path.dirname(os.path.realpath(self.opts.output_path))
                else:
                    self._output_name = self._filename
                    self.output_dir = os.path.dirname(os.path.realpath(self.opts.input_network))

                keep_linear_without_bias = False
                ir_graph_output_names = [output_tensor.name() for output_tensor in ir_graph.get_output_tensors_of_graph()]
                # Prepare model
                self._prepared_model = prepare_model_from_ir_graph(ir_graph,
                                                                   self._datadir,
                                                                   self._filename,
                                                                   self._model_name,
                                                                   keep_linear_without_bias,
                                                                   ir_graph_output_names).to(self._device)
                self._converted_model_info_file = os.path.join(self._datadir, f"{self._filename}_prepared_model_info.pkl")
                if self.should_run_cle():
                    input_shapes = list(self._input_shapes.values())
                    equalize_model(self._prepared_model, input_shapes)

    def _set_device(self, use_cuda):
        """ Set device for Quantsim """
        device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
        if use_cuda and not torch.cuda.is_available():
            logger.warning("Using 'cpu' for Quantization by AIMET Quantizer, as torch is not compiled with CUDA")
        else:
            logger.info(f"Using '{device}' for Quantization by AIMET Quantizer")
        return device

    def is_valid_opts(self):
        """ Returns whether aimet quantizer can be used with the given arguments """
        return self._valid_opts

    def _get_input_list_abs(self):
        """ Get the absolute path for modified input list """
        with open(self.opts.input_list, "r") as file:
            input_0 = file.readline()

        path_example = input_0.split()[0]
        if ":=" in path_example:
            path_example = path_example.split(":=")[1]

        if os.path.exists(path_example):
            return self.opts.input_list

        else:
            # Re-write input paths
            new_lines = []
            input_file_dir = os.path.dirname(self.opts.input_list)
            with open(self.opts.input_list, "r") as file:
                lines = file.readlines()
                if lines[0][0] == '#' or lines[0][0] == '%':
                    self._output_names = lines[0][1:].split(' ')
                    lines = lines[1:]
                for line in lines:
                    new_line = []
                    for input_path in line.split():
                        if ":=" in input_path:
                            input_name, path = input_path.split(":=")
                            new_path = f"{input_name}:={os.path.join(input_file_dir, path)}"
                        else:
                            new_path = os.path.join(input_file_dir, line)
                        new_line.append(new_path)
                    new_line = " ".join(new_line)
                    new_lines.append(new_line)

            temp_input_list = os.path.join(self._datadir, 'temp_input_list.txt')
            with open(temp_input_list, 'w') as f:
                for line in new_lines:
                    f.write(line)

            return temp_input_list

    def _infer_input_dtypes(self, ir_graph: IrGraph):
        """ Infer the input dtypes of the model from IR graph """
        input_dtypes = {tensor.name(): tensor.data_type_as_numpy_type()
                        for tensor in ir_graph.get_input_tensors_to_graph()}

        return input_dtypes

    def _require_config(self):
        return self.should_run_amp() or self.should_run_autoquant()

    def _validate_opts(self) -> bool:
        """ Validate the command-line opts to check if aimet quantizer can be used """
        valid_opts = True

        if self._require_config():
            if self.opts.config is None:
                logger.error(f"Please provide config file using --config to run the algorithms: {self.opts.algorithms}!")
                valid_opts = False
            elif self.opts.input_list:
                logger.warning(f'--input_list is not required for {self.opts.algorithms}. '
                               f'Using dataloader, provided as callback, through config')
        elif self.should_run_adaround():
            if self.opts.config is None:
                if not self.opts.input_list:
                    logger.error("Either '--input_list' or '--config' must be specified for Adaround!")
                    valid_opts = False
            if self._is_unsigned_symmetric_param():
                logger.error(f"Unsigned symmetric quantizer schema is not supported for params during Adaround!")
                valid_opts = False
        else:
            if not self.opts.input_list and not self.opts.float_fallback:
                logger.error("'--input_list' or '--float_fallback' needs to be specified")
                valid_opts = False

        if self.param_quant_scheme not in QNN_TO_AIMET_QUANT_SCHEME_MAP.keys():
            param_quantizer_arg_key = "--param_quantizer" if not self.opts.disable_legacy_quant_scheme_opts else "--param_quantizer_calibration"
            logger.error(f"invalid value '{self.param_quant_scheme}' for {param_quantizer_arg_key}")
            valid_opts = False
        if self.act_quant_scheme not in QNN_TO_AIMET_QUANT_SCHEME_MAP.keys():
            act_quantizer_arg_key = "--act_quantizer" if not self.opts.disable_legacy_quant_scheme_opts else "--act_quantizer_calibration"
            logger.error(f"invalid value '{self.act_quant_scheme}' for {act_quantizer_arg_key}")
            valid_opts = False

        return valid_opts

    def _get_act_param_quant_schemes(self):
        if self.opts.disable_legacy_quant_scheme_opts:
            # weights_bitwidth 16 is supported only with 'symmetric' on HTP
            # TODO: Sync with backend awareness
            if self.opts.weights_bitwidth == 16:
                logger.warning("Forcing param_quantizer_schema to 'symmetric', as --weights_bitwidth is 16")
                self.opts.quant_schemes['param_quant']['schema'] = 'symmetric'
            param_quant_scheme = self.opts.quant_schemes['param_quant']["calibration"]
            act_quant_scheme = self.opts.quant_schemes['act_quant']["calibration"]
        else:
            # weights_bitwidth 16 is supported only with 'symmetric' on HTP
            # TODO: Sync with backend awareness
            if self.opts.weights_bitwidth  == 16:
                logger.warning("Forcing param_quantizer to 'symmetric', as --weights_bitwidth is 16")
                self.opts.quant_schemes['param_quant'] = 'symmetric'
            param_quant_scheme = self.opts.quant_schemes['param_quant']
            act_quant_scheme = self.opts.quant_schemes['act_quant']
        return param_quant_scheme, act_quant_scheme

    def _force_param_quantizer_schema_to_symmetric(self):
        if self.opts.disable_legacy_quant_scheme_opts and self.opts.quant_schemes['param_quant']['schema'] == 'asymmetric':
            self.opts.quant_schemes['param_quant']['schema'] = 'symmetric'
            logger.warning("Can't use 'asymmetric' param_quantizer_schema with --use_per_channel_quantization "
                           "or --use_per_row_quantization. Using 'symmetric' param_quantizer_schema!")

    def _get_config_file(self) -> str:
        """ Get path to config file """
        # TODO: Add backend awareness
        config_file = get_path_for_target_config("htp_quantsim_config_v75")
        config_dict = json.load(open(config_file))

        force_symmetric_param_quant = False
        if self.opts.use_per_channel_quantization:
            logger.info(f'Per Channel Quantization is enabled')
            force_symmetric_param_quant = True
        if self.opts.use_per_row_quantization:
            logger.info(f'Per Row Quantizaton is enabled')
            force_symmetric_param_quant = True

        if force_symmetric_param_quant:
            self._force_param_quantizer_schema_to_symmetric()

        def add_to_config(op_type, flag):
            if op_type in config_dict["op_type"].keys():
                config_dict["op_type"][op_type]["per_channel_quantization"] = flag
            else:
                config_dict["op_type"][op_type] = {"per_channel_quantization": flag}

        config_dict["defaults"]["per_channel_quantization"] = str(self.opts.use_per_channel_quantization)
        # per_row_ops = ["Gemm", "MatMul"]
        add_to_config("Gemm", str(self.opts.use_per_row_quantization))
        add_to_config("MatMul", str(self.opts.use_per_row_quantization))

        quantizer_type_to_config_key = {
            'act_quant': "ops",
            'param_quant': "params"
        }

        if self.opts.disable_legacy_quant_scheme_opts:
            for quantizer in ['act_quant', 'param_quant']:
                if self.opts.disable_legacy_quant_scheme_opts:
                    if self.opts.quant_schemes[quantizer]['schema'] in ["asymmetric"]:
                        config_dict["defaults"][quantizer_type_to_config_key[quantizer]]["is_symmetric"] = "False"
                    elif self.opts.quant_schemes[quantizer]['schema'] in ["symmetric"]:
                        config_dict["defaults"][quantizer_type_to_config_key[quantizer]]["is_symmetric"] = "True"
                else:
                    if self.opts.quant_schemes[quantizer] == 'symmetric':
                        config_dict["defaults"][quantizer_type_to_config_key[quantizer]]["is_symmetric"] = "True"

        temp_config_file = tempfile.NamedTemporaryFile(delete=False)

        with open(temp_config_file.name, "w") as file:
            json.dump(config_dict, file)

        return temp_config_file.name

    def _get_data_dir(self) -> str:
        """ Returns the path to the directory storing converter artifacts """
        return self._datadir

    def get_sample_data(self) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Get n random samples from the dataset
        :return: return random samples or  entire dataset if input list provided
        """
        use_config = self._config_dataloader is not None
        use_input_list = self._input_list_dataloader is not None
        if use_config or use_input_list:
            # Ensure only one of the dataloaders is alive
            assert (use_config and not use_input_list) or (not use_config and use_input_list)
            if self.should_run_amp():
                # For AMP, self._config_dataloader is labelled
                sample, _ = next(iter(self._config_dataloader))
            else:
                # This covers adaround, autoquant (unlabelled for quantsim instantiation)
                # and normal quantsim flows where dataloaders are unlabelled
                sample = next(iter(self._config_dataloader)) if use_config else next(iter(self._input_list_dataloader))

            if isinstance(sample, (tuple, list)):
                sample = tuple([(tensor[0] if use_input_list else tensor).to(self._device) for tensor in sample])
                logger.info(f'Sample input data shape, {[x.shape for x in sample]}')
            else:
                sample = sample[0] if use_input_list else sample
                sample = sample.to(self._device)
                logger.info(f'Sample input data shape, {sample.shape}')
            return sample

        else:
            dummy_input = tuple(
                torch.randn(shape).to(self._device)
                for shape in self._input_shapes.values()
            )
            logger.info(f'Sample dummy input data shape, {[x.shape for x in dummy_input]}')
            if len(dummy_input) == 1:
                dummy_input = dummy_input[0]

            return dummy_input

    def get_input_shape(self) -> dict:
        """
        Get the shape of the input tensor to the model
        """
        return self._input_shapes

    def should_run_cle(self) -> bool:
        """
        Returns true if cle should be run on the model before quantization
        """
        return "cle" in self.opts.algorithms

    def should_run_adaround(self) -> bool:
        """
        Returns true if adaround should be run on the model
        """
        return "adaround" in self.opts.algorithms

    def should_run_amp(self) -> bool:
        """
        Returns true if amp should be run on the sim model
        """
        return "amp" in self.opts.algorithms

    def should_run_autoquant(self) -> bool:
        """
        Returns true if autoquant should be run on the sim model
        """
        return "autoquant" in self.opts.algorithms

    def _get_quant_scheme(self) -> QuantScheme:
        """
        Get the quantization scheme from qnn arguments
        """
        # Set default Quantization scheme to param_quant_schema, and modify after QuantSim instantiation
        quant_scheme = self.param_quant_scheme
        if QNN_TO_AIMET_QUANT_SCHEME_MAP[self.param_quant_scheme] != QNN_TO_AIMET_QUANT_SCHEME_MAP[self.act_quant_scheme]:
            logger.info("Quantization schemes for parameter quantizers and activations quantizers are different")
        return QNN_TO_AIMET_QUANT_SCHEME_MAP[quant_scheme]

    def get_prepare_model(self):
        return self._prepared_model

    def _get_quantsim_args(self) -> dict:
        """
        Get the arguments to quantsim as kwargs dict
        """
        quantsim_args_dict = {
            "model": self._prepared_model,
            "dummy_input": self.get_sample_data(),
            "quant_scheme": self._get_quant_scheme(),
            "rounding_mode": "nearest",
            "default_output_bw": self.opts.act_bitwidth,
            "default_param_bw": self.opts.weights_bitwidth,
            "in_place": True,
            "config_file": self._get_config_file(),
            "default_data_type": QuantizationDataType.float if self.opts.float_fallback else QuantizationDataType.int,
        }
        return quantsim_args_dict

    def labeled_data_calibration_cb(self) -> Callable:
        """Get the calibration callback needed for computing encodings"""

        def pass_calibration_data(model, *args):

            model.eval()

            with torch.no_grad():
                for input_data,_ in tqdm(self._config_dataloader, desc='compute_encodings'):
                    if isinstance(input_data, torch.Tensor):
                        input_batch = input_data.to(self._device)
                        model(input_batch)
                    else:
                        input_batch = tuple(map(lambda d: d.to(self._device), input_data))
                        model(*input_batch)

        return pass_calibration_data

    def input_list_data_calibration_cb(self) -> Callable:
        """Get the calibration callback needed for computing encodings"""

        def pass_calibration_data(model, *args):

            model.eval()

            with torch.no_grad():
                for input_data in tqdm(self._input_list_dataloader, desc='compute_encodings'):
                    if isinstance(input_data, torch.Tensor):
                        input_batch = input_data[0].to(self._device)
                        model(input_batch)
                    else:
                        input_batch = tuple(map(lambda d: d[0].to(self._device), input_data))
                        model(*input_batch)

        return pass_calibration_data

    def unlabeled_data_calibration_cb(self) -> Callable:
        """Get the calibration callback needed for computing encodings"""

        def pass_calibration_data(model, *args):

            model.eval()

            with torch.no_grad():
                for input_data in tqdm(self._config_dataloader, desc='compute_encodings'):
                    if isinstance(input_data, torch.Tensor):
                        input_batch = input_data.to(self._device)
                        model(input_batch)
                    else:
                        input_batch = tuple(map(lambda d: d.to(self._device), input_data))
                        model(*input_batch)

        return pass_calibration_data

    def _set_config_dataloader(self):
        if self.should_run_adaround():
            if self.aimet_argparse is not None:
                dataset = self.aimet_argparse.adaround_config.dataset
                self._config_dataloader = dataset.dataloader_callback(**dataset.dataloader_kwargs)
        elif self.should_run_amp():
            dataset = self.aimet_argparse.amp_config.dataset
            self._config_dataloader = dataset.dataloader_callback(**dataset.dataloader_kwargs)
        elif self.should_run_autoquant():
            # unlabelled dataloader for AutoQuant instantiation
            dataset = self.aimet_argparse.autoquant_config.dataset
            self._config_dataloader = dataset.dataloader_callback(**dataset.dataloader_kwargs)

    def _set_adaround_param(self):
        """
        Set Adaround parameters.
        """

        #AdaRound with user specified params.
        optional_adaround_args = {}
        #Cases Covered : config file with adaround section present in it.
        if self.aimet_argparse and self.aimet_argparse.adaround_config:
            assert self._config_dataloader is not None
            if self.opts.input_list:
                logger.warning("Ignoring input_list, as config is provided for Adaround")
            else:
                logger.info("Using parameters from config file for AdaRound")
            num_batches = self.aimet_argparse.adaround_config.num_batches
            _adaround_params = AdaroundParameters(data_loader=self._config_dataloader,
                                                  num_batches=num_batches,
                                                  **self.aimet_argparse.adaround_config.optional_adaround_param_args)
            optional_adaround_args = self.aimet_argparse.adaround_config.optional_adaround_args

        #AdaRound with default params.
        else:
            def _default_forward_fn(model, inputs):
                inputs = inputs[0]
                if isinstance(inputs, torch.Tensor):
                    input_batch = inputs.to(self._device)
                    return model(input_batch)
                assert isinstance(inputs, (tuple, list))
                input_batch = tuple(map(lambda d: d.to(self._device), inputs))
                return model(*input_batch)

            logger.info(f"No --config provided, using default parameters for AdaRound")

            #default batch_size
            num_batches = min(len(self._input_list_dataloader), math.ceil(2000/self._input_list_dataloader.batch_size))

            _adaround_params = AdaroundParameters(data_loader=self._input_list_dataloader,
                                                  num_batches=num_batches,
                                                  forward_fn=_default_forward_fn)

        return _adaround_params, optional_adaround_args

    def _apply_adaround(self, quantsim_args):
        """
        Calls AdaRound API in AIMET and saves the param encoding in a file.
        """

        adaround_params, optional_adaround_args = self._set_adaround_param()

        #Config file values will overide the default values of QAIRT/AIMET
        #In case the values are not provided in the config file, default/user-provided QAIRT values will be applied.
        if 'default_param_bw' not in optional_adaround_args:
            optional_adaround_args['default_param_bw'] = quantsim_args['default_param_bw']
        else:
            quantsim_args['default_param_bw'] = optional_adaround_args['default_param_bw']

        if 'default_quant_scheme' not in optional_adaround_args:
            logger.info(f"Using '{quantsim_args['quant_scheme']}' from command line "
                        f"as param quant scheme, as it is not mentioned in config")
            optional_adaround_args['default_quant_scheme'] = quantsim_args['quant_scheme']
        else:
            logger.info(f"Using '{optional_adaround_args['default_quant_scheme']}'"
                        f" from adaround config as param quant scheme")
            quantsim_args['quant_scheme'] = optional_adaround_args['default_quant_scheme']

        if 'default_config_file' not in optional_adaround_args:
            optional_adaround_args['default_config_file'] = quantsim_args['config_file']
        else:
            quantsim_args['config_file'] = optional_adaround_args['default_config_file']

        ada_model = Adaround.apply_adaround(self._prepared_model, quantsim_args['dummy_input'], adaround_params,
                                            path=self._datadir,
                                            filename_prefix="adaround",
                                            **optional_adaround_args
                                            )
        adaround_encoding_path = os.path.join(self._datadir,
                                              "{}.encodings".format("adaround"))
        quantsim_args['model'] = ada_model

        return quantsim_args, adaround_encoding_path

    def _set_amp_param(self):

        def forward_one_batch(model, batch):
            model.to(self._device)
            model.eval()
            sample, label = batch
            if isinstance(sample, torch.Tensor):
                sample = sample.to(self._device)
                return model(sample)
            else:
                sample = tuple(map(lambda d: d.to(self._device), sample))
                return model(*sample)

        def evaluator(model, eval_function):

            result = 0.0
            model.to(self._device)
            model.eval()
            with torch.no_grad():
                for input_data, target_data in tqdm(self._config_dataloader, desc='eval'):
                    if isinstance(input_data, torch.Tensor):
                        input_batch = input_data.to(self._device)
                        predicted_batch = model(input_batch)
                    else:
                        input_batch = tuple(map(lambda d: d.to(self._device), input_data))
                        predicted_batch = model(*input_batch)
                    target_data = target_data.to(self._device) if isinstance(target_data, torch.Tensor) else tuple(map(lambda d: d.to(self._device), target_data))
                    out = eval_function(output=predicted_batch, target=target_data)
                    result += out[0].item()
            result /= len(self._config_dataloader)
            return result

        amp_args = {}

        assert self._config_dataloader is not None

        amp_args['candidates'] = self.aimet_argparse.amp_config.candidates

        if 'eval_callback_for_phase1' in self.aimet_argparse.amp_config.optional_amp_args:
            amp_args['eval_callback_for_phase1'] = self.aimet_argparse.amp_config.optional_amp_args['eval_callback_for_phase1']
        else:
            factory = EvalCallbackFactory(self._config_dataloader, forward_fn=forward_one_batch)
            amp_args['eval_callback_for_phase1'] = factory.sqnr(EvalCallbackFactory._DEFAULT_SQNR_NUM_SAMPLES)

        amp_args['eval_callback_for_phase2'] = CallbackFunc(evaluator, self.aimet_argparse.amp_config.eval_callback_for_phase2,)

        amp_args['allowed_accuracy_drop'] = self.aimet_argparse.amp_config.allowed_accuracy_drop

        results_dir = os.path.join(self.output_dir, f'amp_results_{time.strftime("%d%b%Y_%H-%M-%S")}')
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        amp_args['results_dir'] = results_dir

        amp_args['clean_start'] = self.aimet_argparse.amp_config.optional_amp_args.get("clean_start", True)

        if 'forward_pass_callback' in self.aimet_argparse.amp_config.optional_amp_args:
            amp_args['forward_pass_callback'] = self.aimet_argparse.amp_config.optional_amp_args['forward_pass_callback']
        else:
            amp_args['forward_pass_callback'] = CallbackFunc(self.labeled_data_calibration_cb(), None)

        default_args = ['use_all_amp_candidates', 'phase2_reverse', 'amp_search_algo']

        for args in default_args:
            if args in self.aimet_argparse.amp_config.optional_amp_args:
                amp_args[args] = self.aimet_argparse.amp_config.optional_amp_args[args]

        return amp_args

    def _apply_amp(self, sim, dummy_input):

        amp_args = self._set_amp_param()
        sim.compute_encodings(self.labeled_data_calibration_cb(), None)
        choose_mixed_precision(sim, dummy_input, **amp_args)

    def _set_autoquant_param(self, qsim_args, evaluator):

        results_dir = os.path.join(self.output_dir, f'autoquant_results_{time.strftime("%d%b%Y_%H-%M-%S")}')
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        optional_autoquant_args = self.aimet_argparse.autoquant_config.optional_autoquant_args
        autoquant_to_qsim_args = {
            'param_bw': 'default_param_bw',
            'output_bw': 'default_output_bw',
            'quant_scheme': 'quant_scheme',
            'rounding_mode': 'rounding_mode',
            'config_file': 'config_file'
        }
        for arg in autoquant_to_qsim_args:
            if arg in optional_autoquant_args:
                qsim_args[autoquant_to_qsim_args[arg]] = optional_autoquant_args[arg]
            else:
                optional_autoquant_args[arg] = qsim_args[autoquant_to_qsim_args[arg]]

        auto_quant = AutoQuant(self._prepared_model,
                               dummy_input=qsim_args['dummy_input'],
                               data_loader=self._config_dataloader,
                               eval_callback=evaluator,
                               results_dir=results_dir,
                               model_prepare_required=False,
                               **optional_autoquant_args)

        # TODO: Support QuantschemePairs and Default QuantschemePairs Candidates through config file

        def _default_forward_fn(model, inputs):
            if isinstance(inputs, torch.Tensor):
                input_batch = inputs.to(self._device)
                return model(input_batch)
            assert isinstance(inputs, (tuple, list))
            input_batch = tuple(map(lambda d: d.to(self._device), inputs))
            return model(*input_batch)

        # Set AdaRound Params
        if 'num_batches' in self.aimet_argparse.autoquant_config.optional_adaround_args:
            logger.info('Setting AdaRound Params in AutoQuant')
            if 'forward_fn' not in self.aimet_argparse.autoquant_config.optional_adaround_args:
                self.aimet_argparse.autoquant_config.optional_adaround_args['forward_fn']=_default_forward_fn
            adaround_params = AdaroundParameters(self._config_dataloader,
                                                 **self.aimet_argparse.autoquant_config.optional_adaround_args)
            auto_quant.set_adaround_params(adaround_params)

        # AMP will be enabled when two or more candidates are provided
        if (self.aimet_argparse.autoquant_config.amp_candidates is not None and
                len(self.aimet_argparse.autoquant_config.amp_candidates) >= 2):
            # Set AMP Params
            auto_quant.set_mixed_precision_params(
                candidates=self.aimet_argparse.autoquant_config.amp_candidates,
                **self.aimet_argparse.autoquant_config.optional_amp_args
            )

        return auto_quant, qsim_args

    def _apply_autoquant(self, qsim_args):

        def evaluator(model: torch.nn.Module, *args):

            result = 0.0
            model.to(self._device)
            model.eval()
            eval_function = self.aimet_argparse.autoquant_config.eval_callback
            # labelled dataloader for evaluation
            dataset = self.aimet_argparse.autoquant_config.eval_dataset_container
            dataloader = dataset.dataloader_callback(**dataset.dataloader_kwargs)
            with torch.no_grad():
                for input_data, target_data in tqdm(dataloader, desc='eval'):
                    if isinstance(input_data, torch.Tensor):
                        input_batch = input_data.to(self._device)
                        predicted_batch = model(input_batch)
                    else:
                        input_batch = tuple(map(lambda d: d.to(self._device), input_data))
                        predicted_batch = model(*input_batch)
                    target_data = target_data.to(self._device) if isinstance(target_data, torch.Tensor) else tuple(map(lambda d: d.to(self._device), target_data))
                    out = eval_function(output=predicted_batch, target=target_data)
                    result += out[0].item()
            result /= len(dataloader)
            return result

        assert self._config_dataloader is not None

        auto_quant, qsim_args = self._set_autoquant_param(qsim_args, evaluator)
        model, optimized_accuracy, encoding_path, pareto_front = auto_quant.optimize(
            allowed_accuracy_drop=self.aimet_argparse.autoquant_config.allowed_accuracy_drop
        )
        return model, encoding_path, qsim_args

    def _is_unsigned_symmetric_act(self):
        return self.opts.disable_legacy_quant_scheme_opts and self.opts.quant_schemes['act_quant']['schema'] == 'unsignedsymmetric'

    def _is_unsigned_symmetric_param(self):
        return self.opts.disable_legacy_quant_scheme_opts and self.opts.quant_schemes['param_quant']['schema'] == 'unsignedsymmetric'

    def _required_to_modify_act_quant(self, quantsim_args):
        return (QNN_TO_AIMET_QUANT_SCHEME_MAP[self.act_quant_scheme] != quantsim_args['quant_scheme']
                or self._is_unsigned_symmetric_act() or self.act_quant_scheme == 'percentile')

    def _required_to_modify_param_quant(self):
        # Need not check for quant scheme, as we use param_quant_scheme for Quantsim instantiation
        return self._is_unsigned_symmetric_param() or self.param_quant_scheme == 'percentile'

    def modify_quantizers_if_needed(self, sim: QuantizationSimModel, quantsim_args: dict):

        required_to_modify_param_quant = self._required_to_modify_param_quant()
        required_to_modify_act_quant = self._required_to_modify_act_quant(quantsim_args)

        if required_to_modify_act_quant or required_to_modify_param_quant:
            logger.info('Modifying Quantizer settings based on command line arguments..')
            param_quantizers, input_quantizers, output_quantizers = utils.get_all_quantizers(sim.model)

            if required_to_modify_act_quant:
                for quantizer in itertools.chain(input_quantizers, output_quantizers):
                    if QNN_TO_AIMET_QUANT_SCHEME_MAP[self.act_quant_scheme] != quantsim_args['quant_scheme']:
                        # Set input/output quantizers' quant schemes
                        quantizer.quant_scheme = QNN_TO_AIMET_QUANT_SCHEME_MAP[self.act_quant_scheme]
                    if self._is_unsigned_symmetric_act():
                        # Set the following for unsigned_symmetric
                        quantizer.use_unsigned_symmetric = True
                        quantizer.use_symmetric_encodings = True
                    if quantizer.quant_scheme == QuantScheme.post_training_percentile and \
                            self.opts.percentile_calibration_value is not None:
                        quantizer.set_percentile_value(self.opts.percentile_calibration_value)

            if required_to_modify_param_quant:
                if self.should_run_adaround():
                    logger.warning("Can't modify param quantizer settings now for Adaround!")
                else:
                    for quantizer in param_quantizers:
                        if self._is_unsigned_symmetric_param():
                            # Set the following for unsigned_symmetric
                            quantizer.use_unsigned_symmetric = True
                            quantizer.use_symmetric_encodings = True
                        if quantizer.quant_scheme == QuantScheme.post_training_percentile and \
                                self.opts.percentile_calibration_value is not None:
                            quantizer.set_percentile_value(self.opts.percentile_calibration_value)

        return sim

    def generate_quantized_dlc(self, converted_dlc_path) -> IrGraph:
        """
        Return the DLC reader object for the converted, quantized IR graph
        The DLC reader needs to persist to prevent garbage collection of IRGraph static tensor data
        """

        # Check for any quantization overrides
        mpp_aligned_torch_enc_file = os.path.join(self._datadir, self._filename+'_torch.encoding')
        quantization_overrides = os.path.exists(mpp_aligned_torch_enc_file)

        quantsim_args = self._get_quantsim_args()

        if self.opts.float_fallback:
            sim = QuantizationSimModel(**quantsim_args)
            if quantization_overrides:
                sim.load_encodings(mpp_aligned_torch_enc_file, strict=False, partial=True, allow_overwrite=False)
            else:
                raise RuntimeError('--float_fallback can only be provided when user specify encodings through --quantization_overrides')

        elif self.should_run_adaround():
            #AdaRound Flow
            quantsim_args, adaround_encoding_path = self._apply_adaround(quantsim_args)
            # Initialize Quantsim
            sim = QuantizationSimModel(**quantsim_args)
            # For adaround flow, modify quantizer setting only for activations if required based on cmd line args
            sim = self.modify_quantizers_if_needed(sim, quantsim_args)
            logger.info(f'Using "{self.act_quant_scheme}" as activation quant calibration scheme')
            if quantization_overrides:
                error_str = f"AdaRound cannot be used with Quantization Overrides!"
                logger.error(error_str)
                raise RuntimeError(error_str)
            else:
                # Set and freeze param encoding in case of AdaRound
                sim.set_and_freeze_param_encodings(adaround_encoding_path)
                # Compute Encodings
                if self._config_dataloader is not None:
                    sim.compute_encodings(self.unlabeled_data_calibration_cb(),None)
                else:
                    sim.compute_encodings(self.input_list_data_calibration_cb(), None)

        elif self.should_run_amp():
            if quantization_overrides:
                error_str = f"AMP cannot be used with Quantization Overrides!"
                logger.error(error_str)
                raise RuntimeError(error_str)
            else:
                logger.info('Running AMP with given config....')
                sim = QuantizationSimModel(**quantsim_args)
                # Modify quantizer setting for activations and params if required based on cmd line args
                sim = self.modify_quantizers_if_needed(sim, quantsim_args)
                logger.info(f'Using "{self.act_quant_scheme}" as activation quant calibration scheme '
                            f'and "{self.param_quant_scheme}" as param quant calibration scheme')
                self._apply_amp(sim, quantsim_args['dummy_input'])

        elif self.should_run_autoquant():
            if quantization_overrides:
                error_str = f"AutoQuant cannot be used with Quantization Overrides!"
                logger.error(error_str)
                raise RuntimeError(error_str)
            else:
                logger.info('Running AutoQuant with given config....')
                model, encoding_path, quantsim_args = self._apply_autoquant(qsim_args=quantsim_args)
                quantsim_args['model'] = model
                sim = QuantizationSimModel(**quantsim_args)
                torch_encoding_path = encoding_path[:-10]+"_torch.encodings"
                sim.load_encodings(torch_encoding_path, strict=False, partial=True,
                                   requires_grad=False, allow_overwrite=False)
        else:
            sim = QuantizationSimModel(**quantsim_args)
            # Modify quantizer setting for activations and params if required based on cmd line args
            sim = self.modify_quantizers_if_needed(sim, quantsim_args)
            logger.info(f'Using "{self.act_quant_scheme}" as activation quant calibration scheme '
                        f'and "{self.param_quant_scheme}" as param quant calibration scheme')
            if not self.opts.ignore_encodings:
                if quantization_overrides:
                    logger.info('Quantization overrides provided, AIMET will compute any missing encodings')
                    sim.load_and_freeze_encodings(mpp_aligned_torch_enc_file, ignore_when_quantizer_disabled=True)
                else:
                    logger.info('No quantization overrides provided, AIMET will compute the full encodings')
            else:
                logger.info('--ignore_encodings flag is provided, AIMET will ignore any encodings provided')
            # Compute Encodings
            sim.compute_encodings(self.input_list_data_calibration_cb(), None)

        # Export to DLC
        DlcQuantSimExporter.export(sim, self._datadir, self._output_name + '_quantized', converted_dlc_path,
                                   self._converted_model_info_file, quantize_dlc=True, float_bias_bw=self.opts.float_bias_bw)

        # Post-processing
        quantized_dlc_path = os.path.join(self._datadir, f"{self._output_name}_quantized.dlc")

        logger.info('Quantization using AIMET Quantizer is done!')
        return quantized_dlc_path
