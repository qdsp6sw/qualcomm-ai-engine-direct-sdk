# ==============================================================================
#
# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

""" Contains constants specific to android device manager """

SOC_ID_TO_SOC_NAME = {
    # Mobile Devices
    "371": "SXR1130",
    "336": "SDM670",
    "349": "SDM632",
    "338": "SDM450",
    "339": "SM8150",
    "360": "SDM710",
    "353": "SDM439",
    "481": "QCS8250",
    "365": "SM7150",
    "366": "SM7150",
    "394": "SM6125",
    "345": "SDM636",
    "317": "SDM660",
    "352": "QCS405",
    "400": "SM7250",
    "347": "QCS605",
    "407": "SM6250",
    "417": "SM4250",
    "420": "SM4250P",
    "443": "SM7125",
    "444": "SM6115",
    "445": "SM6115P",
    "415": "SM8350",
    "457": "SM8450",
    "434": "SM6350",
    "459": "SM7225",
    "441": "SM4125",
    "318": "SDM630",
    "401": "QCS610_LE",
    "406": "QCS410",
    "475": "SM7325",
    "454": "SM4350",
    "450": "SM7350",
    "411": "QCS405",
    "356": "SM8250",
    "476": "Fraser_SM7250",
    "455": "QRB5165",
    "410": "QCS404",
    "355": "SM6150",
    "467": "QCM6125",
    "501": "SM8325",
    "469": "QCM4290",
    "470": "QCS4290",
    "497": "QCM6490",
    "498": "QCS6490",
    "507": "SM6375",
    "506": "SM7450",
    "530": "SM8475",
    "519": "SM8550",
    "564": "SM7425",
    "591": "SM7475",
    "537": "SM6450",
    "482": "SM8450",
    "536": "SM8550",
    "568": "SM4450",
    "557": "SM8650",
    # XR Devices
    "525": "SXR1230P",  # Aurora/Neo
    "549": "SXR2230P",  # Halliday
    "554": "SSG2115P",  # Aurora LA / Luna
    "579": "SSG2125P",  # Luna V2
    "649": "SXR2250P",  # XR2Gen2+ or Halliday+
}

# Interval time in seconds
POLL_DEVICES_INTERVAL = 5
UNKNOWN_SOC_NAME = "UNKNOWN"