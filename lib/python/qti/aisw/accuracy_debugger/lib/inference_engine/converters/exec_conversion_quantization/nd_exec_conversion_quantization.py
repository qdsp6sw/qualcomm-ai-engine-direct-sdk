# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import os
import json

from qti.aisw.accuracy_debugger.lib.utils.nd_logger import setup_logger
from qti.aisw.accuracy_debugger.lib.utils.nd_constants import Engine, Framework
from qti.aisw.accuracy_debugger.lib.inference_engine.configs import CONFIG_PATH
from qti.aisw.accuracy_debugger.lib.inference_engine import inference_engine_repository
from qti.aisw.accuracy_debugger.lib.utils.nd_constants import Engine, Framework
from qti.aisw.accuracy_debugger.lib.utils.nd_namespace import Namespace
from qti.aisw.accuracy_debugger.lib.inference_engine.environment_manager.nd_get_environment_cls import get_environment_cls


class ExecuteConversionAndQuantization:

    def __init__(self, engine: str, framework: str, host_device: str, engine_path: str,
                 verbose: str, params: dict, output_directory: str = None, logger=None) -> None:
        self._host_device = host_device
        self._logger = setup_logger(verbose, output_directory) if logger is None else logger
        self._engine_type = Engine(engine)
        self._framework = Framework(framework) if framework else None
        file_path = os.path.join(CONFIG_PATH, self._engine_type.value, 'config.json')
        with open(file_path, 'r') as file:
            self._config_data = json.load(file)
        self._registry = inference_engine_repository
        self._output_directory = output_directory
        self._params = params
        self._environment = get_environment_cls(self._engine_type)(engine, self._host_device,
                                                                   self._output_directory,
                                                                   engine_path, self._logger)
        self._environment.initialize()

    def _get_coverter(self, converter_config):
        '''
        Get the converter class object
        '''
        converter_cls = self._registry.get_converter_class(self._framework, self._engine_type, None)
        self._converter = converter_cls(self._get_context(converter_config))

    def _get_context(self, data=None):
        '''
        Returns the context variable
        '''
        return Namespace(data, logger=self._logger)
