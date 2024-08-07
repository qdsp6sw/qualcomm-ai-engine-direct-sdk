# ==============================================================================
#
#  Copyright (c) 2024 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
import logging
from typing import List, Union, Iterable, AnyStr
from abc import abstractmethod, ABC
from os import PathLike

from qti.aisw.tools.core.utilities.devices.utils.device_code import DeviceReturn
from qti.aisw.tools.core.utilities.devices.api.device_definitions import *
from qti.aisw.tools.core.utilities.devices.api.executor import Executor


class DeviceInterface(ABC):
    """
    Abstract base class representing a device interface.

    Attributes:
        _logger (Optional[logging.Logger]): The logger for the device interface. Defaults to None.
        _device_info (Optional[DeviceInfo]): The device information. Defaults to None.
    """

    def __init__(self, device_info: Optional[DeviceInfo] = None, logger: Optional[logging.Logger] = None):
        """
        Initializes the device interface.

        Args:
            device_info (Optional[DeviceInfo]): The device information. Defaults to None.
            logger (Optional[logging.Logger]): The logger for the device interface. Defaults to None.
        """
        self._device_info = device_info
        self._logger = None
        if logger:
            self.set_logger(logger)
        self._executors = None

    @property
    def device_info(self) -> DeviceInfo:
        """
        Returns the device information.

        Returns:
            DeviceInfo: The device information.
        """
        return self._device_info

    @property
    def executors(self) -> Iterable[Executor]:
        """
        Returns the executors for the device interface.

        Returns:
            Optional[Iterable[Executor]]: The executors for the device interface.
        """
        return self._executors

    @device_info.setter
    @abstractmethod
    def device_info(self, device_info: DeviceInfo):
        """
        Abstract method to set the device information.

        Args:
            device_info (DeviceInfo): The device information.
        """

    @abstractmethod
    def execute(self, commands: List[str],
                device_env_context: Optional[DeviceEnvironmentContext] = None) -> DeviceReturn:
        """
        Abstract method to execute commands on the device.

        Args:
            commands (List[str]): The commands to execute.
            device_env_context (Optional[DeviceEnvironmentContext]): The device environment context. Defaults to None.

        Returns:
            DeviceCode: The device code after executing the commands.
        """

    @abstractmethod
    def pull(self, src_path: Union[str, PathLike], dst_path: Union[str, PathLike]) -> DeviceReturn:
        """
        Abstract method to pull files from the device.

        Args:
            src_path (Union[str, PathLike]): The source path on the device.
            dst_path (Union[str, PathLike]): The destination path on the local machine.
        """

    @abstractmethod
    def push(self, src_path: Union[str, PathLike], dst_path: Union[str, PathLike]) -> DeviceReturn:
        """
        Abstract method to push files to the device.

        Args:
            src_path (Union[str, PathLike]): The source path on the local machine.
            dst_path (Union[str, PathLike]): The destination path on the device.
        """

    @abstractmethod
    def make_directory(self, dir_name: Union[str, PathLike]) -> DeviceReturn:
        """
        Abstract method to make a directory on the device.

        Args:
            dir_name (Union[str, PathLike]): The name of the directory to create.
        """

    @abstractmethod
    def remove(self, target_path: Union[str, PathLike]) -> DeviceReturn:
        """
        Abstract method to remove a file or directory from the device.

        Args:
            target_path (Union[str, PathLike]): The path of the file or directory to remove.

        Returns:
            DeviceCode: The device code after removing the file or directory.
        """

    @abstractmethod
    def close(self) -> DeviceReturn:
        """
        Abstract method to close the device interface.
        """

    def set_logger(self, logger: logging.Logger) -> None:
        """
        Sets the logger for the device interface.

        Args:
            logger (logging.Logger): The logger for the device interface.
        """
        self._logger = logger

    def get_device_log(self, **kwargs) -> Union[AnyStr, None]:
        """
        Gets the device log.

        Returns:
            Union[AnyStr, None]: The device log.

        Raises:
            NotImplementedError: If the method is not implemented for the class.
        """
        raise NotImplementedError("The method: get_device_log has not been implemented for this class")

    @staticmethod
    @abstractmethod
    def get_available_devices(connection_type: ConnectionType = ConnectionType.LOCAL,
                              **kwargs) -> Optional[List[DeviceInfo]]:
        """
        Abstract method to get the available devices.

        Args:
            connection_type (ConnectionType): The connection type. Defaults to ConnectionType.LOCAL.

        Returns:
            Optional[List[DeviceInfo]]: The available devices.
        """
