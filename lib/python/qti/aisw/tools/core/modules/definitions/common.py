#==============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
#==============================================================================

from enum import Enum
from ipaddress import ip_address
from os import PathLike
from pydantic import BaseModel, Field, model_validator, ConfigDict, PositiveInt
from typing import Optional, Union, Any

from qti.aisw.tools.core.utilities.devices.api.device_definitions import DevicePlatformType, \
    RemoteDeviceIdentifier, DeviceCredentials

"""
This module contains the definition for AISWBaseModel class which is a pydantic class derived from BaseModel,
and AISWVersion which is a pydantic class that stores fields that categorize a semantic versioning scheme.
"""


class AISWBaseModel(BaseModel):
    """ Internal variation of a BaseModel"""

    model_config = ConfigDict(extra='forbid', validate_assignment=True)


class AISWVersion(AISWBaseModel):
    """
    A dataclass that conveys when modifications are made to a module's interface
    or its properties.
    """

    _MAJOR_VERSION_MAX = 15
    _MINOR_VERSION_MAX = 40
    _PATCH_VERSION_MAX = 15
    _PRE_RELEASE_MAX_LENGTH = 26

    major: int = Field(ge=0, le=_MAJOR_VERSION_MAX)  # Backwards incompatible changes to a module
    minor: int = Field(ge=0, le=_MINOR_VERSION_MAX)  # Backwards compatible changes
    patch: int = Field(ge=0, le=_PATCH_VERSION_MAX)  # Bug fixes are made in a backwards compatible manner
    pre_release: str = Field(default="", max_length=_PRE_RELEASE_MAX_LENGTH)

    @model_validator(mode='after')
    def check_allowed_sem_ver(self):
        """
        Sanity checks a version to ensure it is not all zeros

        Raises:
            ValueError if no version is set
        """
        if self.major == self.minor == self.patch == 0:
            raise ValueError(f'Version: {self.__repr__()} is not allowed')
        return self

    def __str__(self):
        """
         Formats the version as a string value: "major.minor.patch"
         or "major.minor.patch" if the release tag is set
        """
        if not self.pre_release:
            return f'{self.major}.{self.minor}.{self.patch}'
        return f'{self.major}.{self.minor}.{self.patch}-{self.pre_release}'


class BackendType(str, Enum):
    """
    Enum representing backend types that are supported by a module.
    """

    CPU = 'CPU'
    GPU = 'GPU'
    HTP = 'HTP'
    HTP_MCP = 'HTP_MCP'
    AIC = 'AIC'


class Target(AISWBaseModel):
    """
    Defines the type of device to be used by the module, optionally including device identifiers
    and connection parameters for remote devices.

    Attributes:
        type (DevicePlatformType): The type of device platform to be used
        identifier (Optional[RemoteDeviceIdentifier]): The identifier of the device. Defaults to None.
        credentials (Optional[DeviceCredentials]): The credentials for the device. Defaults to
        None.
    """
    type: DevicePlatformType
    identifier: Optional[RemoteDeviceIdentifier] = None
    credentials: Optional[DeviceCredentials] = None


class Model(AISWBaseModel):
    """
    Describes a model in a form that can be consumed by a module. Only one field should be set based
    on the model's format

    Attributes:
        qnn_model_library_path (Optional[Union[str, PathLike]]: Path to a QNN model library
        context_binary_path (Optional[Union[str, PathLike]]): Path to a context binary
        dlc_path (Optional[Union[str, PathLike]]): Path to a DLC
    """
    qnn_model_library_path: Optional[Union[str, PathLike]] = None
    context_binary_path: Optional[Union[str, PathLike]] = None
    dlc_path: Optional[Union[str, PathLike]] = None

    @model_validator(mode='after')
    def validate_one_field_set(self) -> 'Model':
        """
        Validates that only a single model type can be provided per Model instance.

        Args:
            self (Model): The instance of Model.

        Returns:
            Model: The instance of Model if it contains a single model type.

        Raises:
            ValueError: If no model types or > 1 model type was provided
        """
        if len(self.model_fields_set) != 1:
            raise ValueError("Exactly one Model field must be set")
        return self
