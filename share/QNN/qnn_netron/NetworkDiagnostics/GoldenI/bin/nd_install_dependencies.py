# =============================================================================
#
#  Copyright (c) 2019 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import os
import sys
import argparse
import json
import shutil
import re
import site
import subprocess

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../'))
from lib.utils.nd_logger import setup_logger
from lib.utils.nd_errors import get_message
from lib.utils.nd_exceptions import DependencyError
from lib.utils.nd_path_utility import get_absolute_path

def _parse_args_tool(args):
    """
    type: (List[str]) -> argparse.Namespace

    Parses first cmd line argument to determine which tool to run
    :param args: User inputs, fed in as a list of strings
    :return: Namespace object
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Script that builds correct virtual environments from requirements.txt"
    )

    parser.add_argument('-j', '--json_file', type=str, required=True, help="The json file.")

    parsed_args = parser.parse_args(args)
    parsed_args.json_file = get_absolute_path(parsed_args.json_file)

    return parsed_args


def install_dependencies(logger, requirements, output_directories, versions):
    virtualenv_package_dir = "virtualenv_packages"
    if os.path.exists(virtualenv_package_dir):
        shutil.rmtree(virtualenv_package_dir)
    os.mkdir(virtualenv_package_dir)
    os.environ['PYTHONUSERBASE'] = os.path.join(os.getcwd(), virtualenv_package_dir)
    os.environ['PATH'] = os.path.join(os.getcwd(),
                                      virtualenv_package_dir, "bin") + ":" + os.environ['PATH']
    logger.info("Installing virtualenv and virtualenv-api packages")
    sp = subprocess.call("pip3 install --user virtualenv virtualenv-api",
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, shell=True)
    if sp != 0:
        raise DependencyError(get_message("ERROR_VIRTUALENV_INSTALLATION_FAILURE"))

    dependency_pattern = re.compile(r'(.)+==(?:(\d+\.[.\d]*\d+))')
    python_version = str(sys.version_info[0]) + "." + str(sys.version_info[1])
    virtualenv_site_dir = os.path.join(os.getcwd(), virtualenv_package_dir,
                                       "lib/python{}/site-packages".format(python_version))

    try:
        site.addsitedir(virtualenv_site_dir)
        from virtualenvapi.manage import VirtualEnvironment
    except ImportError:
        raise DependencyError(get_message("ERROR_VIRTUALENVAPI_IMPORT_FAILURE")
                              (virtualenv_site_dir))

    for i in range(len(requirements)):
        logger.info("Creating virtual environment directory '{}'".format(output_directories[i]))
        if not os.path.exists(output_directories[i]):
            os.makedirs(output_directories[i])

        env = VirtualEnvironment(output_directories[i], python=shutil.which(versions[i]),
                                 system_site_packages=False)

        with open(requirements[i], 'r') as reqfile:
            logger.info("Installing dependencies for '{}'".format(requirements[i]))
            dependency = reqfile.readline()
            while dependency:
                dep_stripped = dependency.rstrip('\n')
                if dependency_pattern.match(dependency):
                    env.install(dependency.format(dep_stripped))
                else:
                    raise DependencyError(get_message("ERROR_DEPENDENCY_INVALID_NAME")
                                          (dep_stripped))
                dependency = reqfile.readline()

    logger.info("Cleaning temporary package directory.")
    shutil.rmtree(virtualenv_package_dir)

    logger.info("Virtual environments installed successfully.")


def main():
    system_args = sys.argv[1:]
    args = _parse_args_tool(system_args)
    logger = setup_logger(verbose=False)
    json_file = args.json_file
    with open(json_file, 'r') as json_file:
        json_list = json.load(json_file)

    requirements = []
    output_directories = []
    versions = []

    for package in json_list:
        requirements.append(package["requirements"])
        if not os.path.exists(requirements[-1]):
            raise ValueError("'{}' does not exist.".format(requirements[-1]))

        output_directories.append(package["output_directory"])
        if (os.path.exists(output_directories[-1]) and
            len(os.listdir(output_directories[-1])) != 0) or \
                os.path.isfile(output_directories[-1]):
            raise ValueError("'{}' already exists and is non-empty.".format(output_directories[-1]))

        versions.append(package["version"])
        if shutil.which(versions[-1]) is None and not \
                os.path.exists('/usr/bin/{}'.format(versions[-1])):
            raise ValueError("'{}' must be installed.".format(versions[-1]))

    install_dependencies(logger, requirements, output_directories, versions)


if __name__ == '__main__':
    main()
