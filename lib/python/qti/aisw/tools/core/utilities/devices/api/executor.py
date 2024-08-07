# ==============================================================================
#
#  Copyright (c) 2024 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
from os import PathLike
from typing import Any
from abc import abstractmethod, ABC
from qti.aisw.tools.core.utilities.devices.utils.device_code import *
from qti.aisw.tools.core.utilities.devices.protocol_helpers.protocol_helper import ProtocolHelper


class Executor(ABC):

    def __init__(self, protocol_helper: Optional[ProtocolHelper] = None):
        self.protocol_helper = protocol_helper

    @property
    def protocol_helper(self) -> ProtocolHelper:
        """
        Returns the protocol helper associated with this executor
        """
        return self._protocol_helper

    @protocol_helper.setter
    def protocol_helper(self, protocol_helper: ProtocolHelper):
        """
        Set the protocol helper for the executor

        Args:
            protocol_helper: The new protocol helper
        """
        self._protocol_helper = protocol_helper

    @abstractmethod
    def push(self, src: Union[str, PathLike], dst: Union[str, PathLike]) -> DeviceReturn:
        """
        Abstract method to push a file from local storage to remote device

        Args:
            src(Union[str, PathLike]): Source file to be pushed
            dst(Union[str, PathLike]): Destination directory on the remote device

        Returns:
            DeviceReturn (Union[DeviceCompletedProcess, DeviceFailedProcess])
        """

    @abstractmethod
    def pull(self, src: Union[str, PathLike], dst: Union[str, PathLike]) -> DeviceReturn:
        """
        Abstract method to push a file from local storage to remote device

        Args:
            src(Union[str, PathLike]): Source file to be pushed
            dst(Union[str, PathLike]): Destination directory on the remote device

        Returns:
            DeviceReturn (Union[DeviceCompletedProcess, DeviceFailedProcess])
        """

    @abstractmethod
    def execute(self, command: str, args: Optional[List[str]] = None, **kwargs) -> DeviceReturn:
        """
        Abstract method to execute a command on the remote device

        Args:
            command(str): Command to be executed
            args(Optional[List[str]]): List of arguments to be passed to the command
                Defaults to None

        Returns:
            DeviceReturn (Union[DeviceCompletedProcess, DeviceFailedProcess])
        """

    @abstractmethod
    def get_available_devices(self, *args, **kwargs) -> Any:
        """
        Abstract method to get available devices connected to the system
        """
