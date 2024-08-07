#!/bin/bash
#==============================================================================
#
#  Copyright (c) 2020-2024 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
#==============================================================================

# This script sets up the various environment variables needed to run sdk binaries and scripts
OPTIND=1

_usage()
{
cat << EOF
Usage: source $(basename ${BASH_SOURCE[${#BASH_SOURCE[@]} - 1]}) [-h]

Script sets up environment variables needed for running sdk binaries and scripts

EOF
}

function _setup_aisw_sdk()
{
  # get directory of the bash script
  local SOURCEDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" > /dev/null && pwd )"
  local AISW_SDK_ROOT=$(readlink -f ${SOURCEDIR}/..)
  if [ "x${PYTHONPATH}" = "x" ]; then
    export PYTHONPATH="${AISW_SDK_ROOT}/lib/python/"
  else
    export PYTHONPATH="${AISW_SDK_ROOT}/lib/python/":${PYTHONPATH}
  fi
  export PATH=${AISW_SDK_ROOT}/bin/x86_64-linux-clang:${PATH}
  if [ "x${LD_LIBRARY_PATH}" = "x" ]; then
    export LD_LIBRARY_PATH=${AISW_SDK_ROOT}/lib/x86_64-linux-clang
  else
    export LD_LIBRARY_PATH=${AISW_SDK_ROOT}/lib/x86_64-linux-clang:${LD_LIBRARY_PATH}
  fi

  if [ -d "${AISW_SDK_ROOT}/include/QNN" ]; then
    export QNN_SDK_ROOT="$( cd "${AISW_SDK_ROOT}" && pwd )"
    export PYTHONPATH="${QNN_SDK_ROOT}/benchmarks/QNN/":${PYTHONPATH}
    if ls ${QNN_SDK_ROOT}/bin/x86_64-linux-clang/hexagon-* > /dev/null 2>&1; then
        export HEXAGON_TOOLS_DIR=${QNN_SDK_ROOT}/bin/x86_64-linux-clang
    fi
  fi

  if [ -d "${AISW_SDK_ROOT}/include/SNPE" ]; then
    export SNPE_ROOT="$( cd "${AISW_SDK_ROOT}" && pwd )"
  fi
}

function _cleanup()
{
  unset -f _usage
  unset -f _setup_aisw_sdk
  unset -f _cleanup
}

if [ "${BASH_SOURCE[0]}" -ef "$0" ]; then
  echo "[ERROR] This file should be run with 'source'"
  _usage;
  exit 1;
fi

# parse arguments
while getopts "h?" opt; do
  case ${opt} in
    h  ) _usage; return 0 ;;
    \? ) echo "See -h for help."; return 1 ;;
  esac
done

_setup_aisw_sdk

# cleanup
_cleanup

echo "[INFO] AISW SDK environment set"

if [ "x${QNN_SDK_ROOT}" != "x" ]; then
  echo "[INFO] QNN_SDK_ROOT: ${QNN_SDK_ROOT}"
fi

if [ "x${SNPE_ROOT}" != "x" ]; then
  echo "[INFO] SNPE_ROOT: ${SNPE_ROOT}"
fi

