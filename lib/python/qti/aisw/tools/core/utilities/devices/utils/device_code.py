# ==============================================================================
#
#  Copyright (c) 2024 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
from enum import Enum
import subprocess
from typing import Optional, Union, List, AnyStr


class DeviceCode(int, Enum):
    DEVICE_SUCCESS = 0
    DEVICE_UNKNOWN_ERROR = -1


class DeviceCompletedProcess(subprocess.CompletedProcess):

    """
    A custom implementation of a subprocess completed process with additional attributes set by a
    device utility.

    Attributes:
        args: The arguments passed to the process.
        returncode (int): Return code from the process.
        stdout (Optional[AnyStr]): The standard output captured (empty list if none was captured)
        stderr (Optional[AnyStr]): The standard error captured (empty list if none was captured).
        timeout (Optional[int]): Timeout value used during process creation.

    """
    def __init__(self, args: Union[List[str], str], returncode: int = DeviceCode.DEVICE_SUCCESS,
                 stdout: Optional[AnyStr] = None, stderr: Optional[AnyStr] = None,
                 *,
                 timeout: Optional[int] = None):
        super().__init__(args, returncode, stdout, stderr)
        self.stdout = self.stdout
        self.stderr = self.stderr
        self.timeout = timeout


class DeviceFailedProcess(DeviceCompletedProcess):
    """
    A custom implementation of a subprocess failed process with additional attributes set by a
    device utility.

    Attributes:
        args: The arguments passed to the process.
        returncode (int): Return code from the process.
        stdout (Optional[AnyStr]): The standard output captured (empty if none was captured)
        stderr (Optional[AnyStr]): The standard error captured (empty if none was captured).
        timeout (Optional[int]): Timeout value used during process creation.
        orig_error (Optional[Exception]): Exception that caused the failure.
    """
    def __init__(self, args: Union[List[str], str], returncode: int = DeviceCode.DEVICE_SUCCESS,
                 stdout: Optional[AnyStr] = None, stderr: Optional[AnyStr] = None,
                 *, timeout: Optional[int] = None, orig_error: Optional[Exception] = None):
        super().__init__(args, returncode, stdout, stderr, timeout=timeout)
        self.orig_error = orig_error


class DeviceError(RuntimeError):
    """Defines an error for any generic device runtime errors."""
    pass


class DevicePlatformError(TypeError):
    """Defines an error for issues related to device platforms"""
    pass


class InvalidIPAddressError(ValueError):
    """Raised when an invalid IP address is provided."""


DeviceReturn = Union[DeviceCompletedProcess, DeviceFailedProcess]
