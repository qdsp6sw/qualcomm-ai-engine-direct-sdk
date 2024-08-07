import logging
logger = logging.getLogger('AIMET Quantizer')
from typing import Tuple, List,Dict

from aimet_common.defs import QuantScheme, QuantizationDataType
from aimet_common.amp.utils import AMPSearchAlgo

import importlib

def get_callback_function(module_string: str):
    try:
        module_name, method_name = module_string.rsplit('.', 1)
        callback_module = importlib.import_module(module_name)
        callback_function = getattr(callback_module, method_name)
    except ModuleNotFoundError as e:
        logger.error("Make sure that the callback function is located in a directory that "
                     "is part of python 'sys.path'.")
        raise RuntimeError('Error loading callback function specified in config file') from e

    except AttributeError as e:
        logger.error("Make sure that the callback function is defined in the specified path")
        raise RuntimeError('Error loading callback function specified in config file') from e

    return callback_function

def validate_data_type(args, default_dtypes):
    for param, value in args.items():
        if param in default_dtypes:
            if not isinstance(value, default_dtypes[param]):
                raise TypeError("Incorrect data type of %s in config file" % param)
        else:
            raise ValueError("Invalid argument : %s in config file" % param)

class DatasetContainer:

    def __init__(self, dataloader_callback: str, dataloader_kwargs: Dict = {}):

        self.dataloader_callback = get_callback_function(dataloader_callback)
        self.dataloader_kwargs = dataloader_kwargs


class AdaRoundConfig:

    def __init__(self, dataset: str, num_batches: int, optional_adaround_param_args: Dict = {},
                 optional_adaround_args: Dict = {}):

        self.dataset = dataset

        if isinstance(num_batches, int):
            self.num_batches = num_batches
        else:
            raise TypeError("num_batch in config file")

        self.optional_adaround_param_args = self._validate_and_set_optional_adaround_param_args(optional_adaround_param_args)
        self.optional_adaround_args = self._validate_and_set_optional_adaround_args(optional_adaround_args)

    def _validate_and_set_optional_adaround_param_args(self, optional_adaround_param_args):

        default_adaround_param_args_dtype = {'default_num_iterations': int,
                                        'default_reg_param': float,
                                        'default_beta_range': List,
                                        'default_warm_start': float,
                                        'forward_fn': str
                                        }

        validate_data_type(optional_adaround_param_args, default_adaround_param_args_dtype)

        if 'default_beta_range' in optional_adaround_param_args:
            optional_adaround_param_args['default_beta_range'] = tuple(optional_adaround_param_args['default_beta_range'])

        if 'forward_fn' in optional_adaround_param_args:
            optional_adaround_param_args['forward_fn'] = get_callback_function(optional_adaround_param_args['forward_fn'])

        return optional_adaround_param_args

    def _validate_and_set_optional_adaround_args(self,optional_adaround_args):

        default_adaround_args_dtype = {'default_param_bw': int,
                                  'param_bw_override_list': List,
                                  'ignore_quant_ops_list': List,
                                  'default_quant_scheme': str,
                                  'default_config_file': str
                                  }

        validate_data_type(optional_adaround_args, default_adaround_args_dtype)

        if 'default_quant_scheme' in optional_adaround_args:
            optional_adaround_args['default_quant_scheme'] = QuantScheme[optional_adaround_args['default_quant_scheme']]

        return optional_adaround_args

    def set_dataset(self, dataset_container: DatasetContainer):

        self.dataset = dataset_container

class AMPConfig:

    def __init__(self, dataset: str, candidates: List, allowed_accuracy_drop: float, eval_callback_for_phase2: str,
                 optional_amp_args: Dict = {}):

        self.dataset = dataset
        self.eval_callback_for_phase2 = get_callback_function(eval_callback_for_phase2)
        self.candidates = self._validate_and_set_amp_candidates(candidates)
        self.allowed_accuracy_drop = allowed_accuracy_drop

        if not isinstance(eval_callback_for_phase2, str):
            raise TypeError("eval_callback_for_phase2 in config file")

        if not isinstance(allowed_accuracy_drop, (type(None), float)):
            raise TypeError("allowed_accuracy_drop in config file")

        self.optional_amp_args = self._validate_and_set_optional_amp_args(optional_amp_args)

    def _validate_and_set_amp_candidates(self, candidates):

        candidate_list = []
        for candidate in candidates:
            [[output_bw, output_dtype], [param_bw, param_dtype]] = candidate
            updated_candidate = ((output_bw, QuantizationDataType[output_dtype]), (param_bw, QuantizationDataType[param_dtype]))
            candidate_list.append(updated_candidate)

        #TODO Validation on candidates allowed on target
        return candidate_list

    def _validate_and_set_optional_amp_args(self, optional_amp_args):

        default_amp_args_dtype = {
                                  'eval_callback_for_phase1': str,
                                  'forward_pass_callback': str,
                                  'use_all_amp_candidates': bool,
                                  'phase2_reverse': bool,
                                  'amp_search_algo': str,
                                  'clean_start': bool
                                  }

        validate_data_type(optional_amp_args, default_amp_args_dtype)

        if 'eval_callback_for_phase1' in optional_amp_args:
            optional_amp_args['eval_callback_for_phase1'] = get_callback_function(optional_amp_args['eval_callback_for_phase1'])
        if 'forward_pass_callback' in optional_amp_args:
            optional_amp_args['forward_pass_callback'] = get_callback_function(optional_amp_args['forward_pass_callback'])

        if 'amp_search_algo' in optional_amp_args:
            optional_amp_args['amp_search_algo'] = AMPSearchAlgo[optional_amp_args['amp_search_algo']]

        return optional_amp_args

    def set_dataset(self, dataset_container: DatasetContainer):

        self.dataset = dataset_container


class AutoQuantConfig:

    def __init__(self, dataset: str, eval_callback: str, eval_dataset: str, allowed_accuracy_drop: float,
                 amp_candidates: List = None, optional_autoquant_args: Dict = {},
                 optional_adaround_args: Dict = {}, optional_amp_args: Dict = {}):

        self.dataset = dataset
        self.eval_callback = get_callback_function(eval_callback)
        self.eval_dataset = eval_dataset
        self.allowed_accuracy_drop = allowed_accuracy_drop
        self.amp_candidates = self._validate_and_set_amp_candidates(amp_candidates)

        if not isinstance(allowed_accuracy_drop, float):
            raise TypeError("allowed_accuracy_drop in config file")

        self.optional_autoquant_args = self._validate_and_set_optional_autoquant_args(optional_autoquant_args)
        self.optional_adaround_args = self._validate_and_set_optional_adaround_args(optional_adaround_args)
        self.optional_amp_args = self._validate_and_set_optional_amp_args(optional_amp_args)

    def _validate_and_set_amp_candidates(self, candidates):

        candidate_list = []
        for candidate in candidates:
            [[output_bw, output_dtype], [param_bw, param_dtype]] = candidate
            updated_candidate = ((output_bw, QuantizationDataType[output_dtype]), (param_bw, QuantizationDataType[param_dtype]))
            candidate_list.append(updated_candidate)

        # TODO Validation on candidates allowed on target
        return candidate_list

    def _validate_and_set_optional_autoquant_args(self, optional_autoquant_args):

        default_autoquant_args_dtype = {
                                 'param_bw': int,
                                 'output_bw': int,
                                 'quant_scheme': str,
                                 'rounding_mode': str,
                                 'config_file': str,
                                 'cache_id': str,
                                 'strict_validation': bool
                             }

        validate_data_type(optional_autoquant_args, default_autoquant_args_dtype)

        if 'quant_scheme' in optional_autoquant_args:
            optional_autoquant_args['quant_scheme'] = QuantScheme[optional_autoquant_args['quant_scheme']]

        return optional_autoquant_args

    def _validate_and_set_optional_adaround_args(self, optional_adaround_args):

        default_adaround_args_dtype = {
            'num_batches': int,
            'default_num_iterations': int,
            'default_reg_param': float,
            'default_beta_range': Tuple,
            'default_warm_start': float,
            'forward_fn': str
        }

        validate_data_type(optional_adaround_args, default_adaround_args_dtype)

        if 'default_beta_range' in optional_adaround_args:
            optional_adaround_args['default_beta_range'] = tuple(optional_adaround_args['default_beta_range'])

        if 'forward_fn' in optional_adaround_args:
            optional_adaround_args['forward_fn'] = get_callback_function(optional_adaround_args['forward_fn'])

        return optional_adaround_args

    def _validate_and_set_optional_amp_args(self, optional_amp_args):

        default_amp_args_dtype = {
            'num_samples_for_phase_1': int,
            'forward_fn': str,
            'num_samples_for_phase_2': int
        }

        validate_data_type(optional_amp_args, default_amp_args_dtype)

        if 'forward_fn' in optional_amp_args:
            optional_amp_args['forward_fn'] = get_callback_function(optional_amp_args['forward_fn'])

        return optional_amp_args

    def set_dataset(self, dataset_container: DatasetContainer, eval_dataset_container: DatasetContainer):
        self.dataset = dataset_container
        self.eval_dataset_container = eval_dataset_container


class AimetConfigParser:

    def __init__(self, config: dict, algorithms: list) -> None:
        self.config = config
        self._validate_config_file_structure(algorithms)

        if "adaround" in algorithms:
            self.adaround_config = AdaRoundConfig(**self.config['adaround'])
            dataset_container = DatasetContainer(**self.config['datasets'][self.adaround_config.dataset])
            self.adaround_config.set_dataset(dataset_container)

        if "amp" in algorithms:
            self.amp_config = AMPConfig(**self.config['amp'])
            dataset_container = DatasetContainer(**self.config['datasets'][self.amp_config.dataset])
            self.amp_config.set_dataset(dataset_container)

        if "autoquant" in algorithms:
            self.autoquant_config = AutoQuantConfig(**self.config['autoquant'])
            dataset_container = DatasetContainer(**self.config['datasets'][self.autoquant_config.dataset])
            eval_dataset_container = DatasetContainer(**self.config['datasets'][self.autoquant_config.eval_dataset])
            self.autoquant_config.set_dataset(dataset_container, eval_dataset_container)

    def _validate_config_file_structure(self, algorithms):

        required_args = {
            'adaround': ['dataset', 'num_batches'],
            'amp': ['dataset', 'candidates', 'allowed_accuracy_drop', 'eval_callback_for_phase2'],
            'autoquant': ['dataset', 'eval_dataset', 'eval_callback', 'allowed_accuracy_drop']
        }

        for algo in algorithms:
            if algo not in self.config:
                raise Exception("%s configuration not present in YAML file." % algo)

            if self.config[algo]['dataset'] not in self.config['datasets']:
                raise Exception("dataset configuration not defined in YAML file for %s" % algo)

            if any(args not in self.config[algo] for args in required_args[algo]):
                raise Exception(f"The following are required args for {algo} and must be "
                                f"specified in the config file {required_args[algo]}")