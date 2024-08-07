# =============================================================================
#
#  Copyright (c) 2024 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

from pydantic import BaseModel
from datetime import datetime
from typing import Union, List, Optional


class NoInitFactory(type):
    def __call__(cls, *args, **kwargs):
        raise TypeError("Cannot instantiate factory directly")


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class DateRange(BaseModel):
    """
    DateRange contains start and end datetime timestamps
    :datetime start: The start time of range
    :datetime end: The end time of range
    """

    start: datetime
    end: datetime


def format_output_as_list(output: Union[str, bytes]) -> List[str]:
    """
    Formats the str output of a command execution into a list of whitespace and
    newline stripped lines.

    Args:
        output (Union[str, bytes]): The output to format.

    Returns:
        List[str]: The formatted output as a list of strings.
    """

    if not output:
        return []
    output = output.decode("utf-8") if isinstance(output, bytes) else output
    return output.splitlines()
