# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import subprocess
import zipfile
import os
import json
import sys
from abc import ABC, abstractmethod

from qti.aisw.accuracy_debugger.lib.utils.nd_exceptions import InferenceEngineError
from qti.aisw.accuracy_debugger.lib.utils.nd_errors import get_message
from qti.aisw.accuracy_debugger.lib.utils.nd_constants import ComponentType, Engine, Framework, Devices_list
from qti.aisw.accuracy_debugger.lib.device.nd_device_factory import DeviceFactory
from qti.aisw.accuracy_debugger.lib.inference_engine.configs import CONFIG_PATH


class Environment:

    def __init__(self, engine, host_device, output_directory, logger) -> None:
        self._logger = logger
        self._output_directory = output_directory
        self._host_device = host_device
        self._engine_type = Engine(engine)
        self.engine_path = None
        self.host_device_obj = None
        self.host_env = {}
        file_path = os.path.join(CONFIG_PATH, self._engine_type.value, 'config.json')
        with open(file_path, 'r') as file:
            self._config_data = json.load(file)
        if self._host_device == "x86_64-windows-msvc":
            self.env_variables = self._config_data['inference_engine'][
                'x86_64_windows_msvc_environment_variables']
        elif self._host_device == "wos":
            self.env_variables = self._config_data['inference_engine']['wos_environment_variables']
        else:
            self.env_variables = self._config_data['inference_engine']['environment_variables']

    def initialize(self):
        self._set_engine_path()
        self._set_host_environment()
        self._create_host_device()

    def _set_host_environment(self):
        """This helper function sets up the execution environment on host"""
        for var in self.env_variables:
            self.env_variables[var] = (self.env_variables[var]).format(
                sdk_tools_root=self.engine_path)

        # set environment variables depending on host device architecture
        if self._host_device in ["x86_64-windows-msvc", "wos"]:
            for var in self.env_variables:
                self.host_env[var] = self.env_variables[var] + os.pathsep
        else:
            for var in self.env_variables:
                self.host_env[var] = self.env_variables[var] + os.pathsep + '$' + var
        sys.path.insert(0, os.path.join(self.engine_path, "lib", "python"))
        self._logger.info(f"Host environment: {self.host_env}")

    def _set_engine_path(self):
        '''
        Sets the engine(QNN SDK) pathS
        '''
        if self.engine_path.endswith(".zip"):
            self.engine_path = self._unzip_engine_path()

    def _unzip_engine_path(self):
        """This helper function unzips engine_zip and sets the engine_path to
        the correct path."""
        engine_zip_path = self.engine_path
        #Zipfile is breaking the symlink while extracting. So using subprocess for extracting
        try:
            subprocess.run(['unzip', '-q', engine_zip_path, '-d', self._output_directory],
                           stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as err:
            print("ERROR: Extracting SDK with the following error: ", err.returncode)
        with zipfile.ZipFile(engine_zip_path, 'r') as f:
            filelists = f.namelist()
            for file in filelists:
                os.chmod(os.path.join(self._output_directory, file), 0o755)
        if './' in filelists[0]:
            engine_path = os.path.join(self._output_directory, os.path.dirname(filelists[1]))
        else:
            engine_path = os.path.join(self._output_directory, os.path.dirname(filelists[0]))

        return engine_path

    def _create_host_device(self):
        '''
        Create x86 host device
        '''
        if self._host_device and self._host_device not in self._config_data[
                ComponentType.devices.value]['host']:
            raise InferenceEngineError(
                get_message("ERROR_INFERENCE_ENGINE_FAILED_DEVICE_CONFIGURATION")(
                    self._host_device))
        if self._host_device is not None:
            if self._host_device not in Devices_list.devices.value:
                raise InferenceEngineError(
                    get_message("ERROR_INFERENCE_ENGINE_FAILED_DEVICE_CONFIGURATION")(
                        self._host_device))
            device_setup_path = self.engine_path if self._host_device in [
                "x86_64-windows-msvc", "wos"
            ] else None

            self.host_device_obj = DeviceFactory.factory(self._host_device, None, self._logger,
                                                         device_setup_path=device_setup_path)
