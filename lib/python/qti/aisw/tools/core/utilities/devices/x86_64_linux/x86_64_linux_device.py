# ==============================================================================
#
#  Copyright (c) 2024 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
import shutil
import os
from pathlib import Path
from pydantic import validate_call
from qti.aisw.tools.core.utilities.devices.api.device_interface import *
from qti.aisw.tools.core.utilities.devices.x86_64_linux.x86_64_linux_executor import *


class X86LinuxDevice(DeviceInterface):

    def __init__(self, device_info: Optional[DeviceInfo] = None, logger: Optional[Any] = None):
        """
        Initializes the device interface for X86 Linux

        Args:
            device_info (Optional[DeviceInfo]): The device info object. Defaults to None.
            logger (Optional[Any]): The logger for the device interface. Defaults to None.
        """
        super().__init__(device_info=device_info, logger=logger)
        self._executor = X86LinuxExecutor()
        if not self._logger:
            self.set_logger(logging.getLogger(f'X86LinuxDevice'))

    @property
    def executor(self):
        return self._executor

    @property
    def executors(self):
        return [self.executor]

    @property
    def device_info(self) -> Optional[DeviceInfo]:
        return self.device_info

    @device_info.setter
    def device_info(self, device_info: DeviceInfo):
        self.device_info = device_info

    def execute(self, commands: List[str],
                device_env_context: Optional[DeviceEnvironmentContext] = None) -> DeviceReturn:
        """
        Method to execute commands on the device.

        Args:
            commands (List[str]): The commands to execute.
            device_env_context (Optional[DeviceEnvironmentContext]): The device environment context. Defaults to None.

        Returns:
            DeviceReturn (Union[DeviceCompletedProcess, DeviceFailedProcess])
        """

        if device_env_context is None:
            device_env_context = DeviceEnvironmentContext()

        env_vars = []
        cwd = device_env_context.cwd

        if device_env_context.environment_variables:
            env_vars = [f'export {env_var} = {value}' for env_var, value in
                        device_env_context.environment_variables.items()]

        x86_shell_commands = [f'cd {cwd}'] + env_vars + commands
        x86_shell_command = ' && '.join(x86_shell_commands)
        return self.executor.execute(x86_shell_command, shell=device_env_context.shell,
                                     cwd=device_env_context.cwd)

    @validate_call
    def pull(self, src_path: Union[str, Path], dst_path: Union[str, Path]) -> DeviceReturn:
        """
        Method to pull files from the device. Pull is analogous to push in that the
        device is the local host.

        Args:
            src_path (Union[str, Path]): The source path on the device.
            dst_path (Union[str, Path]): The destination path on the local machine.
        """
        # TODO: Implement support for remote device in which case pull != push
        return self.push(src_path, dst_path)

    @validate_call
    def push(self, src_path: Union[str, Path], dst_path: Union[str, Path]) -> DeviceReturn:
        """
        Method to push files to the device.

        Args:
            src_path (Union[str, Path]): The source path on the local machine.
            dst_path (Union[str, Path]): The destination path on the device.
        """
        if os.path.isdir(src_path):
            return self._copy_directory(src_path, dst_path)
        return self._copy_file(src_path, dst_path)

    @validate_call
    def make_directory(self, dir_name: Union[str, Path]) -> DeviceReturn:
        """
        Method to make a directory on the device.

        Args:
            dir_name (Union[str, Path]): The name of the directory to create.
        """
        try:
            os.makedirs(dir_name, exist_ok=True)
            self._logger.debug(f"Created directory: {dir_name}")
        except OSError as err:
            self._logger.error(f"Error creating directory: {dir_name!r}\n str({err})")
            return DeviceFailedProcess(args=[dir_name], returncode=DeviceCode.DEVICE_UNKNOWN_ERROR, stderr=str(err),
                                       orig_error=err)

        return DeviceCompletedProcess(args=[dir_name], returncode=DeviceCode.DEVICE_SUCCESS)

    @validate_call
    def remove(self, target_path: Union[str, Path]) -> DeviceReturn:
        """
        Method to remove a file or directory from the device.

        Args:
            target_path (Union[str, Path]): The path of the file or directory to remove.

        Returns:
            DeviceCompletedProcess after removing the file or directory.
        """
        if os.path.isfile(target_path):
            os.remove(target_path)
        elif os.path.isdir(target_path):
            shutil.rmtree(target_path)

        return DeviceCompletedProcess(args=[str(target_path)], returncode=DeviceCode.DEVICE_SUCCESS)

    def close(self) -> DeviceReturn:
        raise NotImplementedError("The method: get_device_log is not applicable for this class")

    def _copy_directory(self, src_path: Union[str, Path], dst_path: Union[str, Path]) -> DeviceReturn:
        """
        This function copies the content of src_path into dst_path

        Args:
            src_path (Union[str, Path]): The source path to copy from.
            dst_path (Union[str, Path]): The destination path to copy to.

        Return:
            DeviceReturn (Union[DeviceCompletedProcess, DeviceFailedProcess])
        """

        if not os.path.exists(src_path):
            self._logger.error(f"Source path: {src_path!r} not found for X86 Device")
            return DeviceFailedProcess(args=[src_path, dst_path], returncode=DeviceCode.DEVICE_UNKNOWN_ERROR)
        try:
            dir_name = os.path.basename(src_path)  # will be empty if dir_name has a trailing slash
            if not dir_name:  # copy only contents of src_path
                shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
            else:
                # copy the entire src_path
                shutil.copytree(src_path, os.path.join(dst_path, dir_name), dirs_exist_ok=True)
        except shutil.Error as err:
            self._logger.error(f"File: {src_path!r} could not be copied "
                               f" to location: {dst_path!r}")
            return DeviceFailedProcess(args=[src_path, dst_path],
                                       returncode=DeviceCode.DEVICE_UNKNOWN_ERROR,
                                       orig_error=err, stderr=str(err))

        return DeviceCompletedProcess(args=[src_path, dst_path])

    def _copy_file(self, src_path: Union[str, Path], dst_path: Union[str, Path]) -> DeviceReturn:
        """
        This method copies the file from src_path to dst_path

        Args:
            src_path (Union[str, Path]): Source path of the file to be copied.
            dst_path (Union[str, Path]): Destination path of the file to be copied.

        Returns:
            DeviceReturn (Union[DeviceCompletedProcess, DeviceFailedProcess])
        """
        if not os.path.exists(src_path):
            self._logger.error(f"Source path: {src_path} not found for X86_64_Linux Device")

        if os.path.isfile(src_path):
            if not os.path.exists(os.path.dirname(dst_path)):
                self.make_directory(os.path.dirname(dst_path))
            shutil.copy(src_path, dst_path)

            return DeviceCompletedProcess(args=[src_path, dst_path])

        return DeviceFailedProcess(args=[src_path, dst_path], returncode=DeviceCode.DEVICE_UNKNOWN_ERROR)

    @staticmethod
    def get_available_devices(connection_type: ConnectionType = ConnectionType.LOCAL,
                              **kwargs) -> Optional[List[DeviceInfo]]:
        """
        This static method returns available devices on the system

        Args:
            connection_type (ConnectionType): The type of connection. Defaults to ConnectionType.LOCAL

        Returns:
            list: A list of DeviceInfo objects representing the available devices on the system
        """
        if connection_type == ConnectionType.REMOTE:
            raise NotImplementedError("Remote connections are not supported for x86_64 linux")
        return [DeviceInfo(platform_type=DevicePlatformType.X86_64_LINUX)]
