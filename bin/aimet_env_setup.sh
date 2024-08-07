#!/bin/bash

# Check if running as root"
if [ "$EUID" -eq 0 ]; then
    echo "Please do not run this script with sudo"
    showHelp
    return 1
fi

# return back to the original directory if a command fails
# Since there are some commands using cd, this is necessary
original_dir=$(pwd)
cleanup() {
    # TODO: cleanup based on pwd
    cd "$original_dir"
    rm -rf .setup_tmp
}
trap cleanup EXIT;

showHelp() {
cat << EOF
Usage: source aimet_env_setup.sh --env-path <path> [--aimet-sdk-tar <sdk_dir>]
Creates a python virtual environment (if not already present) at the specified <path> and installs AIMET SDK along with necessary dependencies. If AIMET is installed in a directory other than the default QPM installation path, the user needs to specify it using --aimet-sdk-tar

-h, --help          Display help

--env-path          Specify the location where the virtual environment should be created or indicate the path to an existing virtual environment.

--aimet-sdk-tar    Specifies the location where the AIMET SDK installation files (.tar.gz) are stored.

EOF
}

# Ensure that the script is sourced and not directly executed
# This is important to set the "AIMET_ENV_PYTHON" in the environment variable
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "The script should not be executed directly"
    showHelp
    exit 1
fi

env_path=""
aimet_sdk_tar=""

options=$(getopt -l "help,env-path,aimet-sdk-tar:" -o "h")

#eval set -- "$options"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --env-path)
            env_path="$2"
            shift 2
            ;;
        --aimet-sdk-tar)
            aimet_sdk_tar="$2"
            shift 2
            ;;
        -h|--help)
            showHelp
            return 0
            ;;
        --)
            shift
            break
            ;;
        *)
            echo "Invalid option: $1"
            showHelp
            return 1
    esac
done

if [[ -z "$env_path" ]]; then
    echo "Error: Please specify the env path using --env-path"
    showHelp
    return 1
fi

# Each QNN release is tied with a supported min and max AIMET version
# The script will choose the latest supported version that is installed
# If none of the supported versions are installed, then the script prompts the
# user to install a supported version and exits

MIN_VERSION="1.30" # Inclusive

verlte() {
    printf '%s\n' "$1" "$2" | sort -C -V
}

verlt() {
    ! verlte "$2" "$1"
}


validate_aimet_torch_tarball() {
    if [[ "$1" =~ .*aimetpro-release-(.*)\.torch-(.*)-.*\.tar\.gz ]]; then
        version="${BASH_REMATCH[1]}"
        backend="${BASH_REMATCH[2]}"
        if [[ "$version" =~ ([0-9]+\.[0-9]+\.[0-9]+).* ]]; then
              version="${BASH_REMATCH[1]}"
        fi
        echo "AIMET Version: $version ; Backend: $backend "
        if verlt "$version" "$MIN_VERSION"; then
            echo "Provided AIMET tarball version can't be used. Please provide AIMET version >= 1.30.0"
            return 0
        fi
        if [ "$backend" != "cpu" ]  &&  [ "$backend" != "gpu" ]; then
            echo "Provided backend is not supported by AIMET."
            return 0
        fi
    else
        echo "Provided AIMET tarball doesn't follow supported naming convention ( aimetpro-release-<version>.torch-<backend>-release.tar.gz)"
        return 0
    fi
    return 1
}


aimet_torch_tarball=""

if [ ! -z "$aimet_sdk_tar" ]; then
    aimet_sdk_tar=$(readlink -m "$aimet_sdk_tar")
    valid_tarball=$(validate_aimet_torch_tarball "$aimet_sdk_tar")
    echo "VALIDITY FLAG: $valid_tarball "
    if [ "$valid_tarball" ]; then
        aimet_torch_tarball="$aimet_sdk_tar"
    else
        echo "The specified tarball path $aimet_sdk_tar is not a valid."
        echo "Please provide a valid AIMET tarball path (aimetpro-release-*-torch-*-release.tar.gz)"
        echo "Checking QPM installation"
    fi
fi


pick_aimet_version() {
    max_version=""
    for file in "$1"/*; do
        version=$(basename "$file")
        if ! verlt "$version" "$MIN_VERSION"; then
            if [ -z "$max_version" ] || verlt "$max_version" "$version" ; then
                max_version=$version
            fi
        fi
    done
    echo $max_version
}

# QPM installation conformance
# Assuming QNN root is the following path: /<qpm install dir>/aistack/qnn/<qnn version>/
# The script will be present in: /<qpm install dir>/aistack/qnn/<qnn version>/bin/
# Aimet tarball will be present in: /<qpm install dir>/aistack/aimet/<aimet version>
# Relative path to aimet traball dir: <Script source>/../../../aimet/<aimet version>/

# Absolute path of <qpm install dir>/aistack/qnn/<qnn version>/bin folder

if [ -z "$aimet_torch_tarball" ]; then

    script_dir="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"

    # Go three levels up to <qpm install dir>/aistack
    aistack_root="$(dirname "$(dirname "$(dirname "$script_dir")")")"

    # Append "/aimet" to aistack root to get default AIMET installation directory
    aimet_root="${aistack_root}/aimet"

    echo "Checking for AIMET installation in "$aimet_root" "
    if [ ! -d "$aimet_root" ]; then
        echo "AIMET not be found in the default QPM installation directory ${aistack_root}"
        echo "Please install AIMET (version >= $MIN_VERSION ) using QPM or specify an alternate location using --aimet-sdk-tar <sdk_dir>"
        return
    else
        aimet_version=$(pick_aimet_version "$aimet_root")

        if [ -z "$aimet_version" ]; then
            echo "Unsupported AIMET version(s) installed. Please install AIMET (version >= $MIN_VERSION ) using QPM"
            return
        else
            qpm_aimet_sdk_tar="${aimet_root}/${aimet_version}"
            tarball=$(get_aimet_torch_tarball "$qpm_aimet_sdk_tar")
            if [ -z "$tarball" ]; then
                echo "Required AIMET SDK files(aimetpro-release-"$aimet_version"-*-torch-gpu-*.tar.gz or aimetpro-release-"$aimet_version"-*-torch-cpu) missing in "$qpm_aimet_sdk_tar""
                return
            fi
            aimet_torch_tarball="$tarball"
        fi
    fi
fi

echo "Using AIMET SDK: "$aimet_torch_tarball""

# At this point, "aimet_install_dir" contains the tarball files required for setting up the AIMET environment

env_path=$(readlink -m $env_path)

# Virtual environment where AIMET requirements will be installed
echo "Creating virtual environment at $env_path";

if [ -d "$env_path" ]; then
    echo "AIMET virtual environment already exists at $env_path"
    if ! source "$env_path/bin/activate"; then
        echo "Failed to activate virtual environment. Please delete the folder $env_path and re-run this script"
        return 1
    fi
else
    echo "Creating a new AIMET virtual environment..."
    if ! dpkg -s "python3-venv" &> /dev/null; then
        echo "python3-venv is not installed. This is needed to install and manage python dependencies for AIMET"
        sudo apt update -y
        if ! sudo apt install -y python3-venv; then
            "Could not install python3-venv"
            return 1
        fi
    fi
    # Create a virtual environment and activate it
    if python3 -m venv --system-site-packages "$env_path"; then
        echo "Python virtual environment created successfully at $env_path";
        source "$env_path/bin/activate"
    else
        echo "Unable to create virtual environment: $env_path";
        return 1;
    fi
fi


pip3 install --upgrade pip

# Create a new temporary directory for setup artifacts
[ -d ".setup_tmp" ] && rm -rf ".setup_tmp"; mkdir ".setup_tmp"
cd ".setup_tmp"


install_aimet_sdk () {
    # Extract the tarball and install the dependencies
    PACKAGE=aimetpro-release
    mkdir $PACKAGE
    tar xzf $aimet_torch_tarball --strip-components 1 -C $PACKAGE
    cd $PACKAGE

    deb_dependency_files=( dependencies/reqs_deb_*.txt )
    missing_packages=""
    for file in "${deb_dependency_files[@]}"; do
        all_packages=$(cat $file | xargs -I{} echo "{}" | awk '{print $NF}')

        for package in $all_packages; do
            if ! dpkg -s "$package" > /dev/null; then
                missing_packages+=" $package"
            fi
        done
    done

    echo "Missing packages "$missing_packages""

    if [ -n "$missing_packages" ]; then
        echo "The following dependencies are missing and need to be installed:"$missing_packages""
        echo "Installing the missing dependencies"
        sudo apt update -y
        if ! sudo apt -y install$missing_packages; then
            echo "Failed to install missing dependencies"
            return 1
        fi
    else
        echo "All linux dependencies are already installed"
    fi

    echo "Installing python dependencies..."

    if ! pip install pip/*.whl --no-deps; then
       echo "Could not install the AIMET SDK"
       return 1
    fi

    pip_dependency_files=( dependencies/reqs_pip_*.txt )
    for file in "${pip_dependency_files[@]}"; do
        TEMP_CACHE_DIR=`mktemp -d`
        if ! pip install -r $file --extra-index-url https://download.pytorch.org/whl/cpu --cache-dir $TEMP_CACHE_DIR; then
            echo "Could not install AIMET dependencies"
            return 1
        fi
    done
}

if ! install_aimet_sdk; then
    echo "Failed to setup AIMET environment"
    cleanup
    return 1
fi

# HACK
# TODO: Find a permanent fix for this version conflict
# grpcio-tools installs protobuf v4.5 which is incompatbile with the requirements(3.20)
# Manually install the appropriate version here
pip install protobuf==3.20.1

# HACK - module dataclasses has a bug. Can be removed safely
# https://github.com/huggingface/transformers/issues/8638
pip uninstall -y dataclasses

cd $original_dir # To the root
rm -rf .setup_tmp

# Use this environment variable in any program to reference the python executable for running AIMET-specific code
export AIMET_ENV_PYTHON=$env_path/bin/python;

echo "'AIMET_ENV_PYTHON' is set to "$AIMET_ENV_PYTHON" in this environment. Use this environment variable to
to reference the python executable for running AIMET specific code"

# Deactivate in the current context
deactivate;
