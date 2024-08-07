# =============================================================================
#
#  Copyright (c) 2024 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
import logging
import os
from typing import Optional, List
from qti.aisw.tools.core.utilities.devices.utils.device_code import *

default_execute_timeout = 1800
_LOGGER = logging.getLogger(__name__)


def execute(command: str,
            args: Optional[List[str]] = None,
            *,
            cwd: str = os.getcwd(),
            shell: bool = False,
            timeout: int = default_execute_timeout,
            logger: Optional[logging.Logger] = None) -> DeviceReturn:
    """
    Executes a command with optional arguments and settings.

    Args:
        command (str): The command to execute.
        args (Optional[List[str]]): A list of arguments for the command.
        cwd (str): The working directory to execute the command in.
        shell (bool): Whether to use shell to execute the command.
        timeout (int): The timeout for the command execution.
        logger (Optional[logging.Logger]): The logger to use for logging debug and error information.

    Returns:
        DeviceReturn (Union[DeviceCompletedProcess, DeviceFailedProcess]): The result of the command execution.
    """

    if args is None:
        args = []

    if logger is None:
        logger = _LOGGER

    logger.debug(f"Host command: {command} {args}")

    cmd_args = [command] + args

    try:
        process = subprocess.run(cmd_args,
                                 cwd=cwd,
                                 shell=shell,
                                 capture_output=True,
                                 check=True,
                                 text=True,
                                 timeout=timeout)

        logger.debug(f"Process return code: ({process.returncode}), stderr: ({process.stderr})")

        if process.returncode:
            return DeviceFailedProcess(process.args, process.returncode, process.stdout,
                                       process.stderr)
        return DeviceCompletedProcess(process.args, process.returncode, process.stdout,
                                      process.stderr)
    except subprocess.CalledProcessError as error:
        logger.error(str(error))
        return DeviceFailedProcess(cmd_args, error.returncode, error.stdout, error.stderr,
                                   orig_error=error)
    except subprocess.TimeoutExpired as error:
        logger.error(str(error))
        return DeviceFailedProcess(error.cmd, DeviceCode.DEVICE_UNKNOWN_ERROR, error.output,
                                   error.stderr, timeout=int(error.timeout), orig_error=error)
    except OSError as error:
        logger.error(str(error))
        return DeviceFailedProcess(cmd_args, DeviceCode.DEVICE_UNKNOWN_ERROR, "", str(error),
                                   orig_error=error)
