# ==============================================================================
#
#  Copyright (c) 2024 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

from pathlib import Path
from datetime import datetime
from qti.aisw.tools.core.utilities.devices.api.device_interface import *
from qti.aisw.tools.core.utilities.devices.utils.device_utils import DateRange, format_output_as_list
from qti.aisw.tools.core.utilities.devices.android.android_executor import *
from qti.aisw.tools.core.utilities.devices.android.android_device_constants import *


class AndroidDevice(DeviceInterface):

    def __init__(self, device_info: Optional[DeviceInfo] = None, logger: Optional[Any] = None):
        super().__init__(device_info, logger)
        self._executor = AndroidExecutor()
        self._set_device_dependencies()

    def _set_device_dependencies(self):
        """
        Sets dependencies that are needed by the device object such as the device id, logger
        and the hostname.
        """

        self._executor.selected_device_id = self.id

        if not self._logger:
            self.set_logger(logging.getLogger(f'{self.__class__.__name__}: {self.id}'))

        if isinstance(self.device_info, RemoteDeviceInfo):
            self._executor.hostname = self.device_info.identifier.hostname
            self._executor.port_number = self.device_info.identifier.port_number

        if self.device_info and not self.device_info.identifier:
            self._logger.warning("Device identifier was not specified. Note the first available "
                                 " device will be used if an identifier is not specified. ")

    @property
    def id(self) -> Optional[str]:
        if self.device_info:
            if isinstance(self.device_info, RemoteDeviceIdentifier):
                return self.device_info.identifier.serial_id
            else:
                return self.device_info.identifier
        return None

    def get_soc_name(self) -> Union[str, Literal[UNKNOWN_SOC_NAME]]:
        """
        This function returns the soc name of the device using the internal device id.

        Returns:
            str: The soc name of the device if parsed, else it returns `UNKNOWN_SOC_NAME`
        """
        soc_id = self.executor.query_soc_id()

        soc_name = UNKNOWN_SOC_NAME

        if soc_id:
            soc_id = soc_id.split()[0]
            if soc_id in SOC_ID_TO_SOC_NAME:
                soc_name = SOC_ID_TO_SOC_NAME[soc_id]
                self._logger.debug(f"Retrieved soc name: {soc_name} for soc id: {soc_id}")
            else:
                self._logger.error(f"Could not determine the soc name for Android device: {self.id}.")

        return soc_name

    @property
    def device_info(self) -> DeviceInfo:
        return self._device_info

    @property
    def executors(self):
        return [self.executor]

    @property
    def executor(self):
        return self._executor

    @device_info.setter
    def device_info(self, device_info: DeviceInfo):
        """
        Sets the device information if previously unset.

        Raises:
            ValueError: If device info is already set and new value is different from the current.
        """
        if self._device_info is None:
            self._device_info = device_info
            self._set_device_dependencies()
        elif self._device_info != device_info:
            raise ValueError("Device info cannot be reset\n"
                             "Create a new device instance instead")

    def execute(self, commands: List[str],
                device_env_context: Optional[DeviceEnvironmentContext] = None) -> DeviceCompletedProcess:
        """
         Executes a command on the device using the android executor.

         Args:
             commands (list): A list of commands to execute.
             device_env_context (DeviceEnvironmentContext): The environment context of the device.

        Returns:
            DeviceReturn (Union[DeviceCompletedProcess, DeviceFailedProcess]): Execution result
        """

        if device_env_context is None:
            device_env_context = DeviceEnvironmentContext(shell=True)

        env_vars = []
        cwd = device_env_context.cwd

        if device_env_context.environment_variables:
            env_vars = [f'export {env_var} = {value}' for env_var, value in
                        device_env_context.environment_variables.items()]

        android_shell_commands = [f'cd {cwd}'] + env_vars + commands
        android_shell_command = ' && '.join(android_shell_commands)
        device_return = self.executor.execute(android_shell_command, shell=device_env_context.shell)

        if isinstance(device_return, DeviceCompletedProcess):
            self._logger.info(f"Executed command: {android_shell_commands} with return code: {device_return.returncode}")

        return device_return

    def pull(self, src_path: Union[str, PathLike], dst_path: Union[str, PathLike]) -> DeviceReturn:
        """
        Method to pull files from the device.

        Args:
            src_path (Union[str, PathLike]): The source path on the device.
            dst_path (Union[str, PathLike]): The destination path on the local machine.
        """

        # before pulling, we must resolve paths to be created on host
        path_to_create = self._resolve_host_path_creation(dst_path, src_path)

        path_to_create.resolve().mkdir(exist_ok=True, parents=True)

        return self.executor.pull(str(src_path), str(dst_path))

    def _resolve_host_path_creation(self, dst_path: Union[str, PathLike], src_path: Union[str, PathLike]) -> Path:
        """
        Method to resolve the path where the file would be pulled to on the host machine.

        Args:
            dst_path (Union[str, PathLike]): The destination path on the local machine.
            src_path (Union[str, PathLike]): The source path on the device.

        Returns:
            Path: The path where the file would be pulled to on the host machine.
        """
        # assume dst is a directory and its entire path should be created
        create_path = Path(dst_path)

        # unless dst is determined to be a file, in which case only its parent directories should be
        # created.
        if self._check_file_exists_on_device(src_path):

            # check if file path is an existing file or can be assumed to be a file given file extension
            dst_path_is_file = Path(dst_path).suffix or Path(dst_path).is_file()

            # If path names match, then dst_path is a file given src_path is a file
            path_names_match = Path(src_path).name == Path(dst_path).name

            if dst_path_is_file or path_names_match:
                create_path = Path(dst_path).parent

        return create_path

    def push(self, src_path: Union[str, PathLike], dst_path: Union[str, PathLike]) -> DeviceReturn:
        """
        Method to push files or directories to the device.

        Args:
            src_path (Union[str, PathLike]): The source path on the local machine.
            dst_path (Union[str, PathLike]): The destination path on the device.
        """

        if not os.path.exists(src_path):
            raise FileNotFoundError('Path: {src_path} does not exist')
        elif not os.path.isabs(src_path):
            src_path = os.path.abspath(src_path)

        return self.executor.push(str(src_path), str(dst_path))

    def make_directory(self, dir_name: Union[str, PathLike]):
        """
        Method to make a directory on an android device.

        Args:
            dir_name (Union[str, PathLike]): The name of the directory to create.
        """
        mk_dir_command = 'mkdir'
        return self.executor.execute(mk_dir_command, args=["-p", str(dir_name)], shell=True)

    def remove(self, target_path: Union[str, PathLike]) -> DeviceReturn:
        """
        Method to remove a file or directory from the device.

        Args:
            target_path (Union[str, PathLike]): The path of the file or directory to remove.

        Returns:
            DeviceReturn: The device code after removing the file or directory.
        """
        rm_rf_command = 'rm'
        return self.executor.execute(rm_rf_command, args=["-rf", str(target_path)], shell=True)

    def close(self):
        raise NotImplementedError("The method: close has not been implemented for this class")

    def get_device_log(self, date_range: Optional[DateRange] = None) -> Union[bytes, None]:
        """
        Get the device log within a specific time range.

        Args:
            date_range (Optional[DateRange], optional): The date range of the device log. Defaults to None.


        Returns:
            bytes | None: The device log content

        """

        def get_log_in_date_range(stdout: bytes, date_range: DateRange) -> bytes:
            """
            Get the log content within a specific date range.

            Args:
                stdout: The output of the logcat command.
                date_range: The date range of the log.

            Returns:
                bytes: The log content within the date range.
            """
            log_content = ""
            log_lines = format_output_as_list(stdout)
            for line in log_lines:
                try:
                    line_timestamp = " ".join(line.split()[:2])
                    log_datetime = datetime.strptime(line_timestamp, "%Y-%m-%d %H:%M:%S.%f")
                    if log_datetime > date_range.end:
                        break
                    log_content += line + "\n"
                except (IndexError, ValueError):  # log line with no timestamp
                    pass
            stdout = log_content.encode("utf-8")
            return stdout

        command = 'logcat'
        args = ["-d", "*:D", "-v", "year"]

        if date_range:
            args.extend(["-t", str(date_range.start)])

        device_return = self.executor.execute(command, args)

        if isinstance(DeviceReturn, DeviceFailedProcess):
            self._logger.error("Failed to retrieve device log")
            return None

        adb_stdout = device_return.stdout

        if date_range:
            adb_stdout = get_log_in_date_range(adb_stdout, date_range)

        if not adb_stdout:
            return b'No log was found for device'

        return adb_stdout

    @staticmethod
    def _get_available_devices_for_platform(connection_type: ConnectionType = ConnectionType.LOCAL,
                                            hostname: Optional[str] = None,
                                            ip_addr: Optional[str] = None,
                                            port_number: Optional[str] = None,
                                            platform_type: Optional[Union[Literal[DevicePlatformType.ANDROID],
                                                                    Literal[DevicePlatformType.LINUX_EMBEDDED]]] = None
                                            ) -> Optional[List[DeviceInfo]]:
        """
        Static method to get the available devices for a given android platform

        Args:
            connection_type (ConnectionType): The type of connection. Defaults to ConnectionType.LOCAL.
            hostname (Optional[str]): The hostname of the machine connected to the device. Defaults to None.
            ip_addr (Optional[str]): The IP address of the machine connected to the device. Defaults to None
            port_number (Optional[int]): The port number for the connection. Defaults to None

        Returns:
            Optional[List[DeviceInfo]]: The available devices.
        """
        available_device_info = list()
        platform_type = platform_type if platform_type else DevicePlatformType.ANDROID
        if connection_type == connection_type.LOCAL:
            device_ids = AndroidExecutor().get_available_devices()
            for device_id in device_ids:
                available_device_info.append(DeviceInfo(identifier=device_id,
                                                        connection_type=connection_type,
                                                        platform_type=platform_type))
        else:
            if not hostname and ip_addr:
                hostname = ip_addr  # can use either for adb commands
            device_ids = AndroidExecutor().get_available_devices(hostname=hostname,
                                                                 port_number=port_number)
            for device_id in device_ids:
                available_device_info.append(RemoteDeviceInfo(platform_type=platform_type,
                                                              identifier=RemoteDeviceIdentifier(hostname=hostname,
                                                                                                ip_addr=ip_addr,
                                                                                                port_number=port_number,
                                                                                                serial_id=device_id)))

        return available_device_info

    @staticmethod
    def get_available_devices(connection_type: ConnectionType = ConnectionType.LOCAL,
                              hostname: Optional[str] = None,
                              ip_addr: Optional[str] = None,
                              port_number: Optional[str] = None,
                              **kwargs) -> Optional[List[DeviceInfo]]:
        """
        Static method to get the available devices.

        Args:
            connection_type (ConnectionType): The type of connection. Defaults to ConnectionType.LOCAL.
            hostname (Optional[str]): The hostname of the machine connected to the device. Defaults to None.
            ip_addr (Optional[str]): The IP address of the machine connected to the device. Defaults to None
            port_number (Optional[int]): The port number for the connection. Defaults to None

        Returns:
            Optional[List[DeviceInfo]]: The available devices.
        """
        return AndroidDevice._get_available_devices_for_platform(connection_type, hostname, ip_addr, port_number,
                                                                 platform_type=DevicePlatformType.ANDROID)

    def _check_file_exists_on_device(self, file_name: Union[str, PathLike]) -> bool:
        """
        Check whether a file exists on the device.

        Args:
            file_name (Union[str, PathLike]): The name of the file to check.

        Returns:
            bool: True if the file exists, False otherwise.
        """
        return_process = self.executor.execute('test', args=["-f", str(file_name)], shell=True)
        return return_process.returncode == 0
