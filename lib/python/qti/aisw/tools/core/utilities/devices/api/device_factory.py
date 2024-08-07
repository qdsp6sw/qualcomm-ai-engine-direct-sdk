# ==============================================================================
#
#  Copyright (c) 2024 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
import logging

from qti.aisw.tools.core.utilities.devices.android.android_device import AndroidDevice
from qti.aisw.tools.core.utilities.devices.x86_64_linux.x86_64_linux_device import X86LinuxDevice
from qti.aisw.tools.core.utilities.devices.linux_embedded.linux_embedded_device import LinuxEmbeddedDevice
from qti.aisw.tools.core.utilities.devices.utils.device_utils import NoInitFactory
from qti.aisw.tools.core.utilities.devices.utils.device_code import *
from qti.aisw.tools.core.utilities.devices.api.device_interface import *
from typing import Optional, Dict, List, Type, TypeVar, Union

# to avoid linter highlighting on non-base class methods
DeviceInterfaceType = TypeVar('DeviceInterfaceType', bound=DeviceInterface)


def set_logging_level(level: Union[int, str] = "INFO") -> None:
    """
    Set the logging level for the root logger. All subsequent calls
    to loggers will have this level.

    level (Union[int, str]): A valid logging level for the python logging module
    """
    level_name = logging.getLevelName(level)
    file_info = ''
    if level_name == logging.DEBUG:
        file_info = '[%(filename)s:%(lineno)d in function %(funcName)s]'
    logging.basicConfig(format=f'%(asctime)s,%(msecs)d %(levelname)-3s {file_info} %(message)s',
                        datefmt='%Y-%m-%d:%H:%M:%S', level=level_name)


class DeviceFactory(metaclass=NoInitFactory):
    _logger = logging.getLogger(__name__)

    @staticmethod
    def create_device(device_info: DeviceInfo) -> DeviceInterfaceType:
        """
        Creates a device object given a device info and an optional device status.

        Args:
            device_info: Information about the device

        Returns:
            An instance of the appropriate device class
        """
        if device_info.platform_type == DevicePlatformType.ANDROID:
            return AndroidDevice(device_info)
        elif device_info.platform_type == DevicePlatformType.X86_64_LINUX:
            return X86LinuxDevice(device_info)
        elif device_info.platform_type == DevicePlatformType.LINUX_EMBEDDED:
            return LinuxEmbeddedDevice(device_info)
        else:
            raise DevicePlatformError(f'Device could not be created from type:{device_info.platform_type}')

    @staticmethod
    def create_device_info(platform_type: DevicePlatformType,
                           connection_type: ConnectionType,
                           username: Optional[str] = None,
                           password: Optional[str] = None,
                           ip_addr: Optional[str] = None,
                           hostname: Optional[str] = None,
                           port_number: Optional[int] = None) -> DeviceInfo:
        """
        Creates a device info using user-supplied information. The device info contains properties about a device,
        and should be used to create a Device object.

        Args:
            platform_type: Platform type of the device
            connection_type: Whether to connect to the device locally or remotely
            username: Username to use when connecting to the device remotely
            password: Password to use when connecting to the device
            ip_addr: IP address to use when connecting to the device. Either hostname or ip_addr should be provided
            for remote devices.
            hostname: Hostname to use when connecting to the device. Either hostname or ip_addr should be provided
            for remote devices.
            port_number: Port number to use when connecting to the device remotely

        Returns:
            A device info containing the specified properties
        """
        if connection_type == connection_type.LOCAL:
            return DeviceInfo(platform_type=platform_type,
                              connection_type=connection_type)
        elif connection_type == connection_type.REMOTE:
            credentials = DeviceCredentials(username=username, password=password)

            if not (ip_addr or hostname):
                raise ValueError("IP Address or Hostname must be set when connecting to a remote device.")

            return RemoteDeviceInfo(platform_type=platform_type,
                                    connection_type=connection_type,
                                    credentials=credentials,
                                    identifier=RemoteDeviceIdentifier(ip_addr=ip_addr, hostname=hostname,
                                                                      port_number=port_number))
        else:
            raise DeviceError(f"Unknown connection type: {connection_type!r}")

    @staticmethod
    def get_device_by_platform_type(device_platform_type: DevicePlatformType) -> Optional[Type[DeviceInterface]]:
        """
        Returns a device class based on the device platform type.

        Args:
            device_platform_type: The platform type of the device

        Returns:
            The device class corresponding to the device platform type, or None if no class exists

        Raises:
            DevicePlatformError: If the device platform type is unknown
        """
        if device_platform_type == DevicePlatformType.ANDROID:
            return AndroidDevice
        elif device_platform_type == DevicePlatformType.X86_64_LINUX:
            return X86LinuxDevice
        elif device_platform_type == DevicePlatformType.LINUX_EMBEDDED:
            return LinuxEmbeddedDevice
        elif device_platform_type in DevicePlatformType:
            return None
        else:
            raise DevicePlatformError(f'Unknown device platform type: {device_platform_type}')

    @classmethod
    def get_available_devices(cls, connection_type: ConnectionType,
                              username: Optional[str] = None,
                              password: Optional[str] = None,
                              hostname: Optional[str] = None,
                              ip_addr: Optional[str] = None,
                              port_number: Optional[int] = None) -> Dict[DevicePlatformType, List[DeviceInfo]]:
        """
        Returns a dictionary of available devices. Device availability is determined by the protocol helpers
        associated with each device interface

        Args:
            connection_type: Whether to connect to the device locally or remotely
            username: The username to use when connecting to the device remotely
            password: The password to use when connecting to the device remotely
            hostname: The hostname to use when connecting to the device remotely
            ip_addr: The IP address to use when connecting to the device remotely
            port_number: The port number to use when connecting to the device remotely

        Returns:
            A dictionary mapping device platform types to lists of available device infos
        """
        available_devices = {}

        for device_platform_type in DevicePlatformType:
            if device_interface := cls.get_device_by_platform_type(device_platform_type):
                if connection_type == ConnectionType.REMOTE:
                    device_infos = device_interface.get_available_devices(connection_type,
                                                                          device_credentials=DeviceCredentials(
                                                                              username=username,
                                                                              password=password),
                                                                          hostname=hostname,
                                                                          ip_addr=ip_addr,
                                                                          port_number=port_number)
                    cls._logger.debug(f" Discovered {len(device_infos)} {connection_type.value.lower()} device for "
                                      f"platform: "
                                      f"{device_platform_type!s}")
                else:
                    device_infos = device_interface.get_available_devices()
                    cls._logger.debug(f" Discovered {len(device_infos)} {connection_type.value.lower()} device for "
                                      f"platform: "
                                      f"{device_platform_type!s}")

                if device_infos:
                    cls._logger.info(f" Number of available devices for {device_platform_type!s}: {len(device_infos)}")
                    available_devices[device_platform_type] = device_infos

        return available_devices
