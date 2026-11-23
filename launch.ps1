<#
.SYNOPSIS
    ARES Unified Process Launcher (PowerShell Companion Engine)
.DESCRIPTION
    Launches the ARES frontend, backend, or both concurrently.
    Provides native Windows terminal integration, automatic environment detection,
    clean process tree termination on Ctrl+C, port cleanup, and health verification.
.PARAMETER Target
    Target service to launch: both (default), suite, backend, frontend, targets, all, lora, status, clean
.PARAMETER BackendPort
    FastAPI port (default: 8000)
.PARAMETER FrontendPort
    Blazor Server port (default: 5149)
.PARAMETER TargetsPort
    Reference Client Targets port (default: 8001)
.PARAMETER LoraPort
    LoRA inference server port (default: 11435)
.PARAMETER HostAddress
    Backend bind address (default: 127.0.0.1)
.PARAMETER NoReload
    Disable auto-reloading in uvicorn
.PARAMETER SkipHealthCheck
    Skip backend readiness health probe
.PARAMETER MockTargets
    Use offline deterministic Victim LLM simulator on Client Targets
.PARAMETER UseSqlite
    Use local SQLite (ares.db) instead of PostgreSQL
.PARAMETER RequirePostgres
    Abort instead of falling back to SQLite if PostgreSQL is unreachable
.PARAMETER NoColor
    Disable ANSI color output
.PARAMETER NoClean
    Do not auto-clean occupied ports before starting
.PARAMETER Timeout
    Readiness health probe timeout in seconds (default: 45)
.EXAMPLE
    .\launch.ps1
    .\launch.ps1 both
    .\launch.ps1 suite -UseSqlite
    .\launch.ps1 targets -MockTargets
    .\launch.ps1 backend
    .\launch.ps1 frontend
    .\launch.ps1 all
    .\launch.ps1 status
    .\launch.ps1 clean
#>

[CmdletBinding()]
param (
    [Parameter(Position = 0)]
    [ValidateSet("both", "suite", "backend", "frontend", "targets", "all", "lora", "status", "clean", "kill", "exit")]
    [string]$Target,

    [Alias("bp", "backend-port")]
    [int]$BackendPort = 8000,

    [Alias("fp", "frontend-port")]
    [int]$FrontendPort = 5149,

    [Alias("tp", "targets-port")]
    [int]$TargetsPort = 8001,

    [Alias("lp", "lora-port")]
    [int]$LoraPort = 11435,

    [Alias("host")]
    [string]$HostAddress = "127.0.0.1",

    [switch]$NoReload,
    [switch]$SkipHealthCheck,
    [switch]$MockTargets,
    [switch]$UseSqlite,
    [switch]$RequirePostgres,
    [switch]$NoColor,
    [switch]$NoClean,
    [double]$Timeout = 45.0
)

$ErrorActionPreference = "Stop"
$RepoRoot = $PSScriptRoot

# Locate Python Interpreter
$PythonCandidates = @(
    "$RepoRoot\venv\Scripts\python.exe",
    "$RepoRoot\.venv\Scripts\python.exe",
    "$RepoRoot\backend\venv\Scripts\python.exe",
    "$RepoRoot\backend\.venv\Scripts\python.exe"
)

$SelectedPython = $null
foreach ($Candidate in $PythonCandidates) {
    if (Test-Path $Candidate) {
        # Check if candidate can run and load modules cleanly
        try {
            $testOut = & $Candidate -c "import sys, ssl; print('OK')" 2>$null
            if ($LASTEXITCODE -eq 0 -and $testOut -match "OK") {
                $SelectedPython = $Candidate
                break
            }
        } catch {}

        # If DLL or execution issue occurs, inspect pyvenv.cfg for base executable
        $VenvRoot = Split-Path (Split-Path $Candidate)
        $Cfg = Join-Path $VenvRoot "pyvenv.cfg"
        if (Test-Path $Cfg) {
            $Lines = Get-Content $Cfg -ErrorAction SilentlyContinue
            foreach ($Line in $Lines) {
                if ($Line -match "^\s*executable\s*=\s*(.+)$") {
                    $BaseExe = $Matches[1].Trim()
                    if (Test-Path $BaseExe) {
                        $SelectedPython = $BaseExe
                        break
                    }
                }
            }
        }
        if (-not $SelectedPython) {
            $SelectedPython = $Candidate
        }
        if ($SelectedPython) {
            break
        }
    }
}

if (-not $SelectedPython) {
    $SysPython = Get-Command python -ErrorAction SilentlyContinue
    if ($SysPython) {
        $SelectedPython = $SysPython.Source
    } else {
        $PyLauncher = Get-Command py -ErrorAction SilentlyContinue
        if ($PyLauncher) {
            $SelectedPython = $PyLauncher.Source
        }
    }
}

if (-not $SelectedPython) {
    Write-Error "Python was not found in .\venv\Scripts\python.exe or system PATH. Please install Python 3.11/3.12."
    exit 1
}

$LaunchScript = "$RepoRoot\launch.py"
if (-not (Test-Path $LaunchScript)) {
    Write-Error "launch.py not found at $LaunchScript"
    exit 1
}

# Build argument array
$ArgsList = @($LaunchScript)
if ($Target) {
    $ArgsList += $Target
}
if ($BackendPort -ne 8000) {
    $ArgsList += @("--backend-port", $BackendPort.ToString())
}
if ($FrontendPort -ne 5149) {
    $ArgsList += @("--frontend-port", $FrontendPort.ToString())
}
if ($TargetsPort -ne 8001) {
    $ArgsList += @("--targets-port", $TargetsPort.ToString())
}
if ($LoraPort -ne 11435) {
    $ArgsList += @("--lora-port", $LoraPort.ToString())
}
if ($HostAddress -ne "127.0.0.1") {
    $ArgsList += @("--host", $HostAddress)
}
if ($NoReload) {
    $ArgsList += "--no-reload"
}
if ($SkipHealthCheck) {
    $ArgsList += "--skip-health-check"
}
if ($MockTargets) {
    $ArgsList += "--mock-targets"
}
if ($UseSqlite) {
    $ArgsList += "--sqlite"
}
if ($RequirePostgres) {
    $ArgsList += "--require-postgres"
}
if ($NoColor) {
    $ArgsList += "--no-color"
}
if ($NoClean) {
    $ArgsList += "--no-clean"
}
if ($Timeout -ne 45.0) {
    $ArgsList += @("--timeout", $Timeout.ToString())
}

# Execute launch.py directly in current terminal session
& $SelectedPython @ArgsList
$ExitCode = $LASTEXITCODE
exit $ExitCode
