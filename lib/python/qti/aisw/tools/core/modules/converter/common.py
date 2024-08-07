# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

from pydantic import Field, model_validator
from qti.aisw.tools.core.modules.definitions.common import AISWBaseModel
from qti.aisw.converters.common import backend_info


class BackendInfoConfig(AISWBaseModel):
    backend: str = Field(description="Option to specify the backend on which the model needs to run. "
                                     "Providing this option will generate a graph optimized for the given backend.")
    soc_model: str = Field(default="", description="Option to specify the SOC on which the model needs to run. "
                                       "This can be found from SOC info of the device and it starts with strings "
                                       "such as SDM, SM, QCS, IPQ, SA, QC, SC, SXR, SSG, STP, QRB, or AIC.")

    @model_validator(mode="after")
    def validate_backend(self):
        if self.backend not in backend_info.supported_backends():
            raise ValueError("'{}' backend is not supported.".format(self.backend))
        return self


