# ==============================================================================
#
#  Copyright (c) 2024 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
from qti.aisw.tools.core.utilities.devices.android.android_device import *


class LinuxEmbeddedDevice(AndroidDevice):

    def __init__(self, device_info: Optional[DeviceInfo] = None, logger: Optional[logging.Logger] = None,
                 mount_file_system: bool = False):
        """
        Initializes an instance of the X86WindowsMSVCDevice class.

        Args:
            device_info (Optional[DeviceInfo]): Information about the device. Defaults to None.
            logger (Optional[logging.Logger]): Logger for logging messages during device operations. Default to None.
            mount_file_system: Flag that indicates whether to mount the file system during initialization. Defaults
            to False.
        """
        super().__init__(device_info, logger)

        if mount_file_system:
            self._logger.info("Mount file system set to true. Proceeding with mount operations")
            self._mount_file_system()

    def _mount_file_system(self) -> DeviceReturn:
        """
        This function mounts the file system on the device in read, write and execution mode.

        :return: DeviceReturn(Union[DeviceFailedProcess, DeviceCompletedProcess])
        """
        device_return = self.executor.execute('setenforce', ['0'], shell=True)
        if isinstance(device_return, DeviceFailedProcess):
            self._logger.warning("Could not set SELinux to permissive mode. Exiting mount call")
            return device_return

        device_return = self.executor.execute('mount', ['-o', 'remount,rw,exec', '/'], shell=True)
        if isinstance(device_return, DeviceFailedProcess):
            self._logger.warning("Could not remount root filesystem with read/write and execution permissions")
            return device_return

        if device_return.returncode == DeviceCode.DEVICE_SUCCESS:
            self._logger.info("Mount operation completed successfully")

        return device_return

    def check_dir_permissions(self, dir_name: str) -> None:
        """
        Check if directory exists and has read, write and execute permissions.

        Args:
            dir_name (str): Directory path.

        Raises:
            DeviceError: If command fails to run.
            PermissionError: If directory does not exist or does not have required permissions.
        """
        # Run adb shell ls -Z <directory>
        device_return = self.executor.execute("ls", ["-Z ", {dir_name}], shell=True)

        if isinstance(device_return, DeviceFailedProcess):
            raise DeviceError(f"Failed to check permissions with error: {device_return.orig_error!r}")

        # Check if "rwx" (read, write, execute) permissions are present in the output
        if "rwx" not in device_return.stdout:
            raise PermissionError(f"Permissions not set correctly for directory: {dir_name}")
        else:
            self._logger.debug(f"Permissions are set correctly for directory: {dir_name}")

    def execute(self, commands: List[str],
                device_env_context: Optional[DeviceEnvironmentContext] = None) -> DeviceCompletedProcess:
        """
        Executes a command on the device using the linux embedded executor.

        Args:
             commands (list[str]): A list of commands to execute.
             device_env_context (DeviceEnvironmentContext): The environment context of the device.

        Returns:
            DeviceReturn (Union[DeviceCompletedProcess, DeviceFailedProcess]): Execution result
        """

        if device_env_context:
            # TODO: Add cache for directories with permissions already checked
            self.check_dir_permissions(device_env_context.cwd)

        return super().execute(commands, device_env_context)

    @staticmethod
    def get_available_devices(connection_type: ConnectionType = ConnectionType.LOCAL,
                              hostname: Optional[str] = None,
                              ip_addr: Optional[str] = None,
                              port_number: Optional[str] = None,
                              **kwargs) -> Optional[List[DeviceInfo]]:
        """
        Static method to get the available devices.

        Args:
            connection_type (Optional[ConnectionType]): The type of connection. Defaults to ConnectionType.LOCAL.
            hostname (Optional[str]): The hostname of the machine connected to the device. Defaults to None.
            ip_addr (Optional[str]): The IP address of the machine connected to the device. Defaults to None
            port_number (Optional[int]): The port number for the connection. Defaults to None

        Returns:
            Optional[List[DeviceInfo]]: The available devices.
        """
        return LinuxEmbeddedDevice._get_available_devices_for_platform(connection_type, hostname, ip_addr, port_number,
                                                                       platform_type=DevicePlatformType.LINUX_EMBEDDED)
