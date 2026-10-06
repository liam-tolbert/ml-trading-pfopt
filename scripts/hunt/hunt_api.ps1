<#
.SYNOPSIS
Run the hunt API (scripts\hunt\hunt_api.py) in the ml-trading env. The logon task's entry point.

.DESCRIPTION
Needs HUNT_API_TOKEN in the environment (a user environment variable on this PC). Exits
non-zero when the env's python.exe is missing or the token is unset, so the task's restart
settings do not spin on a misconfiguration for long.
#>
[CmdletBinding()]
param()

$EnvRoot = if ($env:HUNT_ENV) { $env:HUNT_ENV } else { "$env:USERPROFILE\miniforge3\envs\ml-trading" }
$Python = Join-Path $EnvRoot 'python.exe'
if (-not (Test-Path $Python)) { Write-Error "no python.exe under $EnvRoot (set HUNT_ENV)"; exit 1 }
if (-not $env:HUNT_API_TOKEN) { Write-Error 'HUNT_API_TOKEN is not set'; exit 2 }

# The env's DLL dirs MUST lead PATH (see scripts\hunt\hunt.sh).
$env:Path = "$EnvRoot;$EnvRoot\Library\bin;$EnvRoot\Library\mingw-w64\bin;$EnvRoot\Library\usr\bin;$EnvRoot\Scripts;$env:Path"
$env:PYTHONIOENCODING = 'utf-8'
Set-Location (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
& $Python (Join-Path $PSScriptRoot 'hunt_api.py')
exit $LASTEXITCODE
