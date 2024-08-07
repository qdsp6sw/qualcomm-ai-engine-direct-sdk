# =============================================================================
#
#  Copyright (c) 2023 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

$SUGGESTED_VS_BUILDTOOLS_VERSION = "14.34"
$SUGGESTED_WINSDK_VERSION = "10.0.22621"
$SUGGESTED_VC_VERSION = "19.34"
$SUGGESTED_CMAKE_VERSION = "3.21"
$SUGGESTED_CLANG_CL_VERSION = "15.0.1"

$global:CHECK_RESULT = 1
$global:tools = @{}
$global:tools.add( 'vswhere', "C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe" )

$options = @{
    checkMSVC = [bool] 0;
    checkTensorflow = [bool]0;
    checkTfLite = [bool] 0;
    checkOnnx = [bool] 0;
    checkPyTorch = [bool] 0;
    enableVerbose = [bool] 0
}

$help = @"
    Usage: envcheck.ps1 [-h] [-a] [-m] [-t] [-l] [-o] [-p]

    Checks if the environment is properly setup for supported toolchains and ml-frameworks.

    optional arguments:

    -h Shows the help content.
    -m Checks if environment is set for MSVC toolchain.
    -t Checks if environment is set for using TensorFlow.
    -l Checks if environment is set for using TFLite.
    -o Checks if environment is set for using ONNX.
    -p Checks if environment is set for using PyTorch.
    -a Checks if environment is set for all supported toolchains and ml-frameworks.
    -v Enable verbose logs to help get more logs.
"@

Function PrintHelp {
    param()

    process {
        Write-Host $help -ForegroundColor Cyan; break 1
    }
}

Function ParseOptions {
    param (
        $argv,
        [System.Collections.Hashtable] $options
    )

    process {
        $opts = @()
        if (!$argv) { return $null }
        foreach ($arg in $argv) {
            # Make sure the argument is something you are expecting
            $test = ($arg -is [int]) -or
                    ($arg -is [string]) -or
                    ($arg -is [float])
            if (!$test) {
                Write-Host "Bad argument: $arg is not an integer, float, nor string." -ForegroundColor Red
                throw "Error: Bad Argument"
            }
            if ($arg -like '-*') { $opts += $arg }
        }
        $argv = [Collections.ArrayList]$argv
        if ($opts) {
            foreach ($opt in $opts) {
                switch ($opt) {
                    "-m"        { $options.checkMSVC = [bool] 1; break }
                    "--msvc"    { $options.checkMSVC = [bool] 1; break }
                    "-t"        { $options.checkTensorflow = [bool] 1; break }
                    "--tf"      { $options.checkTensorflow = [bool] 1; break }
                    "-l"        { $options.checkTfLite = [bool] 1; break }
                    "--tflite"  { $options.checkTfLite = [bool] 1; break }
                    "-o"        { $options.checkOnnx = [bool] 1; break }
                    "--onnx"    { $options.checkOnnx = [bool] 1; break }
                    "-p"        { $options.checkPyTorch = [bool] 1; break }
                    "--pytorch" { $options.checkPyTorch = [bool] 1; break }
                    "-a" {
                        $options.checkMSVC = [bool] 1;
                        $options.checkTensorflow = [bool] 1;
                        $options.checkTfLite = [bool] 1;
                        $options.checkOnnx = [bool] 1;
                        $options.checkPyTorch = [bool] 1;
                        break
                    }
                    "--all"   {
                        $options.checkMSVC = [bool] 1;
                        $options.checkTensorflow = [bool] 1;
                        $options.checkTfLite = [bool] 1;
                        $options.checkOnnx = [bool] 1;
                        $options.checkPyTorch = [bool] 1;
                        break
                    }
                    "-v" {
                        $options.enableVerbose = [bool] 1;
                        break
                    }
                    "--verbose" {
                        $options.enableVerbose = [bool] 1;
                        break
                    }
                    "-h"     { PrintHelp; break }
                    "--help" { PrintHelp; break }
                    default {
                        Write-Host "Bad option: $opt is not a valid option." -ForegroundColor Red
                        throw "Error: Bad Option"
                    }
                }
            $argv.Remove($opt)
            }
        }
        return [array]$argv,$options
    }
}

Function Show-Recommended-Version-Message {
    param (
        [String] $SuggestVersion,
        [String] $FoundVersion,
        [String] $SoftwareName
    )

    process {
        Write-Warning "The version of $SoftwareName $FoundVersion found has not been validated. Recommended to use known stable $SoftwareName version $SuggestVersion"
    }
}

Function Show-Required-Version-Message {
    param (
        [String] $RequiredVersion,
        [String] $FoundVersion,
        [String] $SoftwareName
    )

    process {
        Write-Host "ERROR: Require $SoftwareName version $RequiredVersion. Found $SoftwareName version $FoundVersion" -ForegroundColor Red
    }
}


Function Compare-Version {
    param (
        [String] $TargetVersion,
        [String] $FoundVersion,
        [String] $SoftwareName
    )

    process {
        if ( (([version]$FoundVersion).Major -eq ([version]$TargetVersion).Major) -and (([version]$FoundVersion).Minor -eq ([version]$TargetVersion).Minor) ) { }
        elseif ( (([version]$FoundVersion).Major -ge ([version]$TargetVersion).Major) -and (([version]$FoundVersion).Minor -ge ([version]$TargetVersion).Minor) ) {
            Show-Recommended-Version-Message $TargetVersion $FoundVersion $SoftwareName
        }
        else {
            Show-Required-Version-Message $TargetVersion $FoundVersion $SoftwareName
            $global:CHECK_RESULT = 0
        }
    }
}

Function Locate-Prerequisite-Tools-Path {
    param ()

    process {
        # Get and Locate VSWhere
        if (!(Test-Path $global:tools['vswhere'])) {
            Write-Host "No Visual Studio Instance(s) Detected, Please Refer To The Product Documentation For Details" -ForegroundColor Red
            Exit
        }
    }
}

Function Detect-VS-Instance {
    param ()

    process {

        Locate-Prerequisite-Tools-Path

        $INSTALLED_VS_VERSION = & $global:tools['vswhere'] -latest -property installationVersion

        $INSTALLED_PATH = & $global:tools['vswhere'] -latest -property installationPath

        $productId = & $global:tools['vswhere'] -latest -property productId

        return $productId, $INSTALLED_PATH, $INSTALLED_VS_VERSION
    }
}

Function Check-VS-BuildTools-Version {

    param (
        [String] $SuggestVersion = $SUGGESTED_VS_BUILDTOOLS_VERSION
    )

    process {
        $INSTALLED_PATH = & $global:tools['vswhere'] -latest -property installationPath
        $version_file_path = Join-Path $INSTALLED_PATH "VC\Auxiliary\Build\Microsoft.VCToolsVersion.default.txt"

        if (Test-Path $version_file_path) {
            $INSTALLED_VS_BUILDTOOLS_VERSION = Get-Content $version_file_path
            Compare-Version $SuggestVersion $INSTALLED_VS_BUILDTOOLS_VERSION "VS BuildTools"
            return $INSTALLED_VS_BUILDTOOLS_VERSION
        }
        else {
            Write-Error "VS BuildTools not installed"
            $global:CHECK_RESULT = 0
        }

        return "Not Installed"
    }

}

Function Check-WinSDK-Version {

    param (
        [String] $SuggestVersion = $SUGGESTED_WINSDK_VERSION
    )

    process {
        $INSTALLED_WINSDK_VERSION = Get-ItemPropertyValue -Path 'HKLM:\SOFTWARE\WOW6432Node\Microsoft\Microsoft SDKs\Windows\v10.0' -Name ProductVersion

        if($?) {
            Compare-Version $SuggestVersion $INSTALLED_WINSDK_VERSION "Windows SDK"
            return $INSTALLED_WINSDK_VERSION
        }
        else {
            Write-Error "Windows SDK not installed"
            $global:CHECK_RESULT = 0
        }

        return "Not Installed"
    }
}

Function Check-VC-Version {
    param (
        [String] $VsInstallLocation,
        [String] $BuildToolVersion,
        [String] $Arch,
        [String] $SuggestVersion = $SUGGESTED_VC_VERSION
    )

    process {
        $VcExecutable = Join-Path $VsInstallLocation "VC\Tools\MSVC\" | Join-Path -ChildPath $BuildToolVersion | Join-Path -ChildPath "bin\Hostx64" | Join-Path -ChildPath $Arch | Join-Path -ChildPath "cl.exe"

        if(Test-Path $VcExecutable) {

            #execute $VcExecutable and retrieve stderr since version is in it.
            $process_alloutput = & "$VcExecutable" 2>&1
            $process_stderror = $process_alloutput | Where-Object { $_ -is [System.Management.Automation.ErrorRecord] }

            $CMD = $process_stderror | Out-String | select-string "Version\s+(\d+\.\d+\.\d+)" # The software version is output in STDERR
            $INSTALLED_VC_VERSION = $CMD.matches.groups[1].value

            if($INSTALLED_VC_VERSION) {
                Compare-Version $SuggestVersion $INSTALLED_VC_VERSION ("Visual C++(" + $Arch + ")")
                return $INSTALLED_VC_VERSION
            }
            else {
                Write-Error "Visual C++ not installed"
                $global:CHECK_RESULT = 0
            }
        }

        return "Not Installed"
    }

}

Function Check-CMake-Version {
    param (
        [String] $SuggestVersion = $SUGGESTED_CMAKE_VERSION
    )

    process {
        $INSTALLED_CMAKE_VERSION = (cmake --version).Split(' ')[2].Split('-')[0]

        if($?) {
            Compare-Version $SuggestVersion $INSTALLED_CMAKE_VERSION "CMake"
            return $INSTALLED_CMAKE_VERSION
        }
        else {
            Write-Error "CMake not installed"
            $global:CHECK_RESULT = 0
        }

        return "Not Installed"
    }

}

Function Check-Clang-CL-Version {
    param (
        [String] $SuggestVersion = $SUGGESTED_CLANG_CL_VERSION
    )

    process {
        $INSTALLED_CLANG_CL_VERSION = (clang-cl.exe --version).Split(' ')[2]

        if($?) {
            Compare-Version $SuggestVersion $INSTALLED_CLANG_CL_VERSION "clang-cl"
            return $INSTALLED_CLANG_CL_VERSION
        }
        else {
            Write-Error "clang-cl not installed"
            $global:CHECK_RESULT = 0
        }

        return "Not Installed"
    }

}

Function Check-MSVC-Components-Version {
    param ()

    process {
        $check_result = @()

        $productId, $vs_install_path, $vs_installed_version = Detect-VS-Instance

        if ($productId) {
            $check_result += [pscustomobject]@{Name = "Visual Studio"; Version = $vs_installed_version}
        }
        else {
            $check_result += [pscustomobject]@{Name = "Visual Studio"; Version = "Not Installed"}
            $global:CHECK_RESULT = 0
        }

        $buildtools_version = Check-VS-BuildTools-Version
        $check_result += [pscustomobject]@{Name = "VS Build Tools"; Version = $buildtools_version}
        $check_result += [pscustomobject]@{Name = "Visual C++(x86)"; Version = Check-VC-Version $vs_install_path $buildtools_version "x64"}
        $check_result += [pscustomobject]@{Name = "Visual C++(arm64)"; Version = Check-VC-Version $vs_install_path $buildtools_version "arm64"}

        $check_result += [pscustomobject]@{Name = "Windows SDK"; Version = Check-WinSDK-Version}
        $check_result += [pscustomobject]@{Name = "CMake"; Version = Check-CMake-Version}
        $check_result += [pscustomobject]@{Name = "clang-cl"; Version = Check-Clang-CL-Version}

        Write-Host ($check_result | Format-Table| Out-String).Trim()
    }
}

Function Check-Tensorflow {
    param (
        [bool] $verbose = 0
    )

    process {
        if ($verbose) {
            py -3 -c "import tensorflow"
        }
        else {
            py -3 -c "import tensorflow" 2> $null
        }

        if($?) {
            Write-Host "TensorFlow is set-up successfully"
        }
        else {
            Write-Host "[ERROR] Unable to import tensorflow using python3" -ForegroundColor Red
        }
    }
}

Function Check-TfLite {
    param (
        [bool] $verbose = 0
    )

    process {
        if ($verbose) {
            py -3 -c "import tflite"
        }
        else {
            py -3 -c "import tflite" 2> $null
        }

        if($?) {
            Write-Host "TFLite is set-up successfully"
        }
        else {
            Write-Host "[ERROR] Unable to import tflite using python3" -ForegroundColor Red
        }
    }
}

Function Check-Onnx {
    param (
        [bool] $verbose = 0
    )

    process {
        if ($verbose) {
            py -3 -c "import onnx"
        }
        else {
            py -3 -c "import onnx" 2> $null
        }

        if($?) {
            Write-Host "ONNX is set-up successfully"
        }
        else {
            Write-Host "[ERROR] Unable to import onnx using python3" -ForegroundColor Red
        }
    }
}

Function Check-PyTorch {
    param (
        [bool] $verbose = 0
    )

    process {
        if ($verbose) {
            py -3 -c "import torch"
        }
        else {
            py -3 -c "import torch" 2> $null
        }

        if($?) {
            Write-Host "PyTorch is set-up successfully"
        }
        else {
            Write-Host "[ERROR] Unable to import torch using python3" -ForegroundColor Red
        }
    }
}

# Main
if ($Args.Count -eq 0) {
    PrintHelp
}

$argv, $options = ParseOptions $Args $options

Write-Host ($options | Format-Table | Out-String)

if ($options.checkMSVC) {
    Write-Host "Checking MSVC Toolchain"
    Write-Host "--------------------------------------------------------------"
    Check-MSVC-Components-Version
    Write-Host "--------------------------------------------------------------"
    Write-Host ""
}

if ($options.checkTensorflow) {
    Write-Host "Checking Tensorflow"
    Write-Host "--------------------------------------------------------------"
    Check-Tensorflow $options.enableVerbose
    Write-Host "--------------------------------------------------------------"
    Write-Host ""
}

if ($options.checkTfLite) {
    Write-Host "Checking TfLite"
    Write-Host "--------------------------------------------------------------"
    Check-TfLite $options.enableVerbose
    Write-Host "--------------------------------------------------------------"
    Write-Host ""
}

if ($options.checkOnnx) {
    Write-Host "Checking ONNX"
    Write-Host "--------------------------------------------------------------"
    Check-Onnx $options.enableVerbose
    Write-Host "--------------------------------------------------------------"
    Write-Host ""
}

if ($options.checkPyTorch) {
    Write-Host "Checking PyTorch"
    Write-Host "--------------------------------------------------------------"
    Check-PyTorch $options.enableVerbose
    Write-Host "--------------------------------------------------------------"
    Write-Host ""
}
