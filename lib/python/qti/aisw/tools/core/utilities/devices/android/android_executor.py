# ==============================================================================
#
#  Copyright (c) 2024 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

import os
import logging
from qti.aisw.tools.core.utilities.devices.api.executor import *
from qti.aisw.tools.core.utilities.devices.protocol_helpers.adb import AdbProtocolHelper

default_adb_shell_timeout = 1800


class AndroidExecutor(Executor):
    """
    Executor class for executing commands on android devices using adb
    """
    _logger = logging.getLogger("AndroidExecutor")  # need to have a root logger, this should be a child

    def __init__(self, device_id: Optional[str] = None, hostname: Optional[str] = None,
                 port_number: Optional[int] = None):
        super().__init__(protocol_helper=AdbProtocolHelper())
        self.selected_device_id = device_id
        self.hostname = hostname
        self.port_number = port_number

    @property
    def protocol_helper(self) -> AdbProtocolHelper:
        return self._protocol_helper

    @property
    def _device_id(self) -> Optional[str]:
        """
        Determines a device id using the following hierarchy:
          1. if selected_device_id is set, it is returned
          2. ANDROID_SERIAL environment variable is returned
          3. first available device using self.get_available_devices()
          4. None if no device ids were obtained using 1-3

        Returns:
            str: The device id
        """
        if self.selected_device_id:
            return self.selected_device_id
        elif device_id := os.getenv("ANDROID_SERIAL", None):
            return device_id
        elif self.get_available_devices(hostname=self.hostname, port_number=self.port_number):
            return self.get_available_devices()[0]
        else:
            self._logger.warning("No available devices were found")
            return None

    @protocol_helper.setter
    def protocol_helper(self, protocol_helper: AdbProtocolHelper):
        self._protocol_helper = protocol_helper

    def push(self, src: str, dst: str) -> DeviceReturn:
        """
        Pushes a file from local machine to target android device

        Args:
            src(str): Source file or directory path on local machine
            dst(str): Source file or directory path on android device

        Returns:
            DeviceReturn(Union[DeviceCompletedProcess, DeviceFailedProcess])
        """
        dst_exists = False

        self._logger.debug(f"Pushing {src} to {dst} for device: {self._device_id}")

        # check if the dst already exists
        device_return = self.execute('ls', [dst], shell=True)

        if device_return.returncode == 0:
            # if dst exists and then it will be updated
            dst_exists = True
            self._logger.debug(f" {dst} exists on device: {self._device_id}. Only newer artifacts will be pushed")
            device_return = self.execute('push', ['--sync', src, dst])
        else:
            device_return = self.execute('push', [src, dst])

        if device_return.returncode != 0:
            raise DeviceError(f"Failed to push {src} to device")

        # if dst exists and is a directory
        expected_remote_dst = dst
        # Retrieve source base name
        src_basename = os.path.basename(src) if src[-1] != '/' else os.path.basename(src[:-1])
        if dst_exists and dst[-1] == '/':
            # Add source base name to path directly  -> /path/to/remote/ + <src_basename>
            expected_remote_dst = expected_remote_dst + src_basename
        else:
            # Append source base name to path after'/' -> /path/to/remote + '/' + <src_basename>
            expected_remote_dst = expected_remote_dst + '/' + src_basename

        # check expected remote destination exists on device
        device_return = self.execute('ls', [expected_remote_dst], shell=True)

        if device_return.returncode != 0:
            raise DeviceError(f"Failed to push {src} to device location: {dst}")

        self._logger.info(f"Pushed {src} to {dst} for device: {self._device_id}")

        return device_return

    def pull(self, src: str, dst: str) -> DeviceReturn:
        """
        Pulls a file from target android device to local machine

        Args:
            src(str): Source file or directory path on android device
            dst(str): Source file or directory path on local machine

        Returns:
            DeviceReturn(Union[DeviceCompletedProcess, DeviceFailedProcess])
        """
        device_return = self.execute('pull', [src, dst])

        if device_return.returncode != 0:
            raise DeviceError(f"Failed to pull {src} from device to : {dst}")

        self._logger.info(f"Pulled {src} to {dst} for device: {self._device_id}")
        return device_return

    def execute(self, command: str, args: Optional[List[str]] = None, shell=False) -> DeviceReturn:
        """
        Executes a given command on an android device

        Args:
            command(str): Command to run on android device
            args(Optional[List[str]]): List of arguments to pass to the command
            shell(bool): Whether to execute as a shell command or not. Defaults to False

        Returns:
            DeviceReturn(Union[DeviceCompletedProcess, DeviceFailedProcess]): The execution result
        """
        if shell:
            return self.protocol_helper.shell(command, args, device_id=self._device_id, hostname=self.hostname,
                                              port_number=self.port_number)
        else:
            return self.protocol_helper.execute(command, args, device_id=self._device_id, hostname=self.hostname,
                                                port_number=self.port_number)

    def get_available_devices(self, *, hostname: Optional[str] = None,
                              port_number: Optional[int] = None) -> Optional[List[str]]:
        """
        Returns a list of all connected devices

        Args:
            hostname(str): The hostname of the device to connect to. Defaults to localhost
            port_number: The port number of the device to connect to. Defaults to 5037

        Returns:
            Optional[List[str]]: A list of device ids or None if none were found
        """
        return self.protocol_helper.get_available_devices(hostname=hostname, port_number=port_number)

    def query_soc_id(self) -> Optional[str]:
        """
        Retrieves the captured output from querying "/sys/devices/soc0/soc_id"
        on the specified device. The output will contain the soc id of the device if the query
        was successful.


        Returns:
           str: The captured output of the command if successful, else None


        """
        soc_id_command = f'cat /sys/devices/soc0/soc_id'

        device_return = self.protocol_helper.shell(soc_id_command,
                                                   device_id=self._device_id,
                                                   hostname=self.hostname,
                                                   port_number=self.port_number)

        if isinstance(device_return, DeviceFailedProcess):
            self._logger.error(f"Error in fetching SOC id for device: {self._device_id} "
                               f"with error {device_return.stderr}")
            return None

        return device_return.stdout
