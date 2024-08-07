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
$SUGGESTED_VS_VERSION = "17.4"

$global:tools = @{}
$global:tools.add( 'vswhere', "C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe" )

$global:need_installed = [System.Collections.ArrayList]::new()

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
        }
    }
}


Function Locate-Prerequisite-Tools-Path {
    param ()

    process {

        $productId = Detect-VS-Instance
        $global:tools.add( 'productId', $productId )

        # https://learn.microsoft.com/en-us/visualstudio/install/use-command-line-parameters-to-install-visual-studio?view=vs-2022
        $vs_installer_path = "C:\Program Files (x86)\Microsoft Visual Studio\Installer\setup.exe"
        $global:tools.add( 'installer', $vs_installer_path )
    }
}

Function Install-VS-Instance {
    param()

    process {
        Write-Host "No Visual Studio instance(s) found"
        $user_consent = Read-Host "Do you want to install Visual Studio 2022 Community Edition?[y/n]"
        if ($user_consent.ToUpper() -eq "Y") {
            $installation_process = Start-Process "winget" "install Microsoft.VisualStudio.2022.Community" -PassThru
            $installation_process.WaitForExit()
        }
        else {
            Write-Host "Process Aborted" -ForegroundColor Red
            Exit
        }
    }
}

Function Detect-VS-Instance {
    param ()

    process {

        if (Test-Path $global:tools['vswhere'] ) {
            $INSTALLED_VS_VERSION = & $global:tools['vswhere'] -latest -property installationVersion
            if (!$INSTALLED_VS_VERSION) {
                Install-VS-Instance
                $INSTALLED_VS_VERSION = & $global:tools['vswhere'] -latest -property installationVersion
                if (!$INSTALLED_VS_VERSION) {
                    Write-Host "Installation Failed, Process Aborted" -ForegroundColor Red
                    Exit
                }
                else {
                    if ([version]$INSTALLED_VS_VERSION -lt [version]$SUGGESTED_VS_VERSION ) {
                        Write-Warning "Recommend to use Visual Studio 2022(17.4), Found " $INSTALLED_VS_VERSION
                    }
                }
            }
            else {
                if ([version]$INSTALLED_VS_VERSION -lt [version]$SUGGESTED_VS_VERSION ) {
                    Write-Warning "Recommend to use Visual Studio 2022(17.4), Found " $INSTALLED_VS_VERSION
                }
            }
        }
        else {
            Install-VS-Instance
            if (Test-Path $global:tools['vswhere'] ) {
                $INSTALLED_VS_VERSION = & $global:tools['vswhere'] -latest -property installationVersion
                if (!$INSTALLED_VS_VERSION) {
                    Write-Host "Installation Failed, Process Aborted" -ForegroundColor Red
                    Exit
                }
            }
            else {
                Write-Host "Installation Failed, Process Aborted" -ForegroundColor Red
                Exit
            }
        }

        $INSTALLED_PATH = & $global:tools['vswhere'] -latest -property installationPath
        $DEV_POWERSHELL_SCRIPT = Join-Path $INSTALLED_PATH "\Common7\Tools\Launch-VsDevShell.ps1"

        & $DEV_POWERSHELL_SCRIPT | Out-Null

        $productId = & $global:tools['vswhere'] -latest -property productId
        return $productId

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
            $global:need_installed.Add("Microsoft.Component.MSBuild")
            $global:need_installed.Add("Microsoft.VisualStudio.Component.VC.Tools.x86.x64")
            $global:need_installed.Add("Microsoft.VisualStudio.Component.VC.Tools.ARM64")
        }
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
            $global:need_installed.Add("Microsoft.VisualStudio.Component.Windows11SDK.22621")
        }

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
            $global:need_installed.Add("Microsoft.VisualStudio.Component.VC.CMake.Project")
        }

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
            $global:need_installed.Add("Microsoft.VisualStudio.ComponentGroup.NativeDesktop.Llvm.Clang")
        }

    }

}

Function Invoke-Components-Installation {
    param ()

    process {
        $args = ' modify --productId ' + $global:tools["productId"] + ' --channelId VisualStudio.17.Release '

        foreach ( $component in $global:need_installed ) {
            $args = $args + ' --add ' + $component
        }

        Write-Host "Please continue process installation process at Visual Studio Installer, and close the Installer after the installation process completed." -ForegroundColor Yellow
        $process = Start-Process -FilePath $global:tools["installer"] -ArgumentList $args -PassThru
        $process.WaitForExit()
    }
}

Locate-Prerequisite-Tools-Path

Check-VS-BuildTools-Version | Out-Null
Check-WinSDK-Version | Out-Null
Check-CMake-Version | Out-Null
Check-Clang-CL-Version | Out-Null

if ($global:need_installed.Count -ne 0) {
    Invoke-Components-Installation
}

Write-Host "All Done"