<#
.SYNOPSIS
Register (or remove) the scheduled task that runs the unattended weekend hunt.

.DESCRIPTION
Creates "SEPA Weekend Hunt": weekly, in the current user's interactive session, waking the
PC from sleep. It runs scripts\hunt\weekend_hunt.ps1. Re-running replaces the task.

The task wakes the PC from sleep or hibernate only. From a shutdown or a logged-out
session nothing runs until the next logon, when a missed run starts.

.PARAMETER Day
Day of the week. Default Friday.

.PARAMETER At
Local time, HH:mm. Default 18:00: the Pi's EOD screen starts 16:20 ET and may take 40 minutes.

.PARAMETER WakeTestInMinutes
Instead of the weekly task, register a one-off "SEPA Weekend Hunt (wake test)" that runs
weekend_hunt.ps1 -WakeTest that many minutes from now. Put the PC to sleep and watch it wake.

.PARAMETER Api
Instead of the weekly task, register and start "SEPA Hunt API": at logon, runs
scripts\hunt\hunt_api.ps1, which the Pi's cockpit calls to start a hunt. Needs the user
environment variable HUNT_API_TOKEN and a firewall rule allowing TCP 8765 from the LAN.

.PARAMETER Unregister
Remove all three tasks.
#>
[CmdletBinding()]
param(
    [System.DayOfWeek]$Day = 'Friday',
    [string]$At = '18:00',
    [int]$WakeTestInMinutes = 0,
    [switch]$Api,
    [switch]$Unregister
)

$ErrorActionPreference = 'Stop'
$TaskName     = 'SEPA Weekend Hunt'
$WakeTestName = "$TaskName (wake test)"
$ApiName      = 'SEPA Hunt API'
$Repo         = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$Script       = Join-Path $PSScriptRoot 'weekend_hunt.ps1'
$ApiScript    = Join-Path $PSScriptRoot 'hunt_api.ps1'

if ($Unregister) {
    foreach ($name in $TaskName, $WakeTestName, $ApiName) {
        if (Get-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue) {
            Unregister-ScheduledTask -TaskName $name -Confirm:$false
            Write-Host "removed '$name'"
        }
    }
    return
}

if ($Api) {
    if (-not [Environment]::GetEnvironmentVariable('HUNT_API_TOKEN', 'User')) {
        Write-Warning 'HUNT_API_TOKEN is not set as a user environment variable; the API will refuse to start until it is.'
    }
    $action = New-ScheduledTaskAction -Execute 'powershell.exe' -WorkingDirectory $Repo `
        -Argument "-NoProfile -WindowStyle Hidden -File `"$ApiScript`""
    $trigger = New-ScheduledTaskTrigger -AtLogOn -User "$env:USERDOMAIN\$env:USERNAME"
    $principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType Interactive -RunLevel Limited
    # No time limit: it listens until logoff. Restarted if it dies; one instance only.
    $settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
        -MultipleInstances IgnoreNew -ExecutionTimeLimit ([TimeSpan]::Zero) `
        -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1)
    Register-ScheduledTask -TaskName $ApiName -Action $action -Trigger $trigger -Principal $principal `
        -Settings $settings -Force `
        -Description 'Hunt API: lets the Pi cockpit start a weekend hunt on this PC (scripts\hunt\hunt_api.py).' |
        Out-Null
    Start-ScheduledTask -TaskName $ApiName
    Write-Host "registered and started '$ApiName' (at logon). Allow inbound TCP 8765 from the LAN in Windows Firewall."
    return
}

$wakeTimers = powercfg /q SCHEME_CURRENT SUB_SLEEP RTCWAKE | Select-String 'Current AC Power Setting Index'
if ($wakeTimers -and $wakeTimers.Line -match '0x0+$') {
    Write-Warning 'Wake timers are disabled in the active power plan; the task cannot wake the PC. Enable "Allow wake timers" under Power Options > Sleep.'
}

$isTest    = $WakeTestInMinutes -gt 0
$name      = if ($isTest) { $WakeTestName } else { $TaskName }
$arguments = "-NoProfile -WindowStyle Hidden -File `"$Script`""
if ($isTest) { $arguments += ' -WakeTest' }

$action = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument $arguments -WorkingDirectory $Repo
$trigger = if ($isTest) {
    New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes($WakeTestInMinutes)
} else {
    New-ScheduledTaskTrigger -Weekly -DaysOfWeek $Day -At $At
}
# Interactive: the run needs the user's ssh key, claude login and env vars, and no password
# is stored. IgnoreNew: a second start while one runs would review the same hunt twice.
$principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType Interactive -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -WakeToRun -StartWhenAvailable `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Hours 4)

Register-ScheduledTask -TaskName $name -Action $action -Trigger $trigger -Principal $principal `
    -Settings $settings -Force `
    -Description 'Unattended SEPA weekend hunt: pulls the Pi scan, runs the weekend-hunt skill headless, writes docs\hunt\<date>\report.html.' |
    Out-Null

$info = Get-ScheduledTaskInfo -TaskName $name
Write-Host "registered '$name'; next run $($info.NextRunTime)"
