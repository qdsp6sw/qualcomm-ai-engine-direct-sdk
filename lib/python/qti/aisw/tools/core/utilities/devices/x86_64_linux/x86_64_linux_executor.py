# ==============================================================================
#
#  Copyright (c) 2024 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
import logging
import os
from qti.aisw.tools.core.utilities.devices.api.executor import *
from qti.aisw.tools.core.utilities.devices.utils import subprocess_helper


class X86LinuxExecutor(Executor):
    _logger = logging.getLogger("X86Executor")  # need to have a root logger, this should be a child

    def __init__(self):
        super().__init__(protocol_helper=None)

    def push(self, src: Union[str, PathLike], dst: Union[str, PathLike]):
        raise NotImplementedError(f"Method: push is not applicable to class: {self.__name__}")

    def pull(self, src: Union[str, PathLike], dst: Union[str, PathLike]) -> DeviceReturn:
        raise NotImplementedError(f"Method: pull is not applicable to class: {self.__name__}")

    def execute(self, command: str, args: Optional[List[str]] = None, shell: bool = True,
                cwd: str = os.getcwd()) -> DeviceReturn:
        """
        Execute the command on x86 host

        Args:
            command (str): The command to run.
            args (Optional[List[str]], optional): List of arguments for the command. Defaults to None.
            shell (bool, optional): Whether to use shell. Defaults to True.
            cwd (str): Current working directory. Defaults to os.getcwd()

        Returns:
            DeviceReturn Union[DeviceCompletedProcess, DeviceFailedProcess]
        """
        return subprocess_helper.execute(command, args, cwd=cwd, shell=shell)

    def get_available_devices(self, *args, **kwargs):
        raise NotImplementedError(f"Method: get_available_devices is not applicable to class: {self.__name__}")
