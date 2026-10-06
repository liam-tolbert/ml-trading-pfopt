<#
.SYNOPSIS
Register (or remove) the scheduled task that runs the unattended weekend hunt.

.DESCRIPTION
Creates "SEPA Weekend Hunt": weekly, in the current user's interactive session, waking the
PC from sleep. It runs scripts\hunt\hunt_poller.ps1 -Once, which runs the hunt the Pi's
Friday timer requested (deploy\units\cockpit-huntrequest.timer). Re-running replaces the task.

The task wakes the PC from sleep or hibernate only. From a shutdown or a logged-out
session nothing runs until the next logon, when a missed run starts.

.PARAMETER Day
Day of the week. Default Friday.

.PARAMETER At
Local time, HH:mm. Default 18:00: the Pi's EOD screen starts 16:20 ET and may take 40 minutes.

.PARAMETER WakeTestInMinutes
Instead of the weekly task, register a one-off "SEPA Weekend Hunt (wake test)" that runs
weekend_hunt.ps1 -WakeTest that many minutes from now. Put the PC to sleep and watch it wake.

.PARAMETER Poller
Instead of the weekly task, register and start "SEPA Hunt Poller": at logon, runs
scripts\hunt\hunt_poller.ps1, which checks the Pi over ssh every 30 s for a hunt request
(the cockpit's Start button) and runs it. Nothing connects to this PC.

.PARAMETER Unregister
Remove all three tasks.
#>
[CmdletBinding()]
param(
    [System.DayOfWeek]$Day = 'Friday',
    [string]$At = '18:00',
    [int]$WakeTestInMinutes = 0,
    [switch]$Poller,
    [switch]$Unregister
)

$ErrorActionPreference = 'Stop'
$TaskName     = 'SEPA Weekend Hunt'
$WakeTestName = "$TaskName (wake test)"
$PollerName   = 'SEPA Hunt Poller'
$Repo         = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$Script       = Join-Path $PSScriptRoot 'weekend_hunt.ps1'
$PollerScript = Join-Path $PSScriptRoot 'hunt_poller.ps1'

if ($Unregister) {
    foreach ($name in $TaskName, $WakeTestName, $PollerName) {
        if (Get-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue) {
            Unregister-ScheduledTask -TaskName $name -Confirm:$false
            Write-Host "removed '$name'"
        }
    }
    return
}

if ($Poller) {
    $action = New-ScheduledTaskAction -Execute 'powershell.exe' -WorkingDirectory $Repo `
        -Argument "-NoProfile -WindowStyle Hidden -File `"$PollerScript`""
    $trigger = New-ScheduledTaskTrigger -AtLogOn -User "$env:USERDOMAIN\$env:USERNAME"
    $principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType Interactive -RunLevel Limited
    # No time limit: it polls until logoff. Restarted if it dies; one instance only.
    $settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
        -MultipleInstances IgnoreNew -ExecutionTimeLimit ([TimeSpan]::Zero) `
        -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1)
    Register-ScheduledTask -TaskName $PollerName -Action $action -Trigger $trigger -Principal $principal `
        -Settings $settings -Force `
        -Description 'Hunt poller: runs the weekend hunts the Pi cockpit asks for (scripts\hunt\hunt_poller.ps1).' |
        Out-Null
    Start-ScheduledTask -TaskName $PollerName
    Write-Host "registered and started '$PollerName' (at logon)."
    return
}

$wakeTimers = powercfg /q SCHEME_CURRENT SUB_SLEEP RTCWAKE | Select-String 'Current AC Power Setting Index'
if ($wakeTimers -and $wakeTimers.Line -match '0x0+$') {
    Write-Warning 'Wake timers are disabled in the active power plan; the task cannot wake the PC. Enable "Allow wake timers" under Power Options > Sleep.'
}

$isTest    = $WakeTestInMinutes -gt 0
$name      = if ($isTest) { $WakeTestName } else { $TaskName }
# The weekly task wakes the PC and runs whatever request the Pi's Friday timer left; the
# wake test exercises the wake and the sleep rule without a hunt.
$arguments = if ($isTest) { "-NoProfile -WindowStyle Hidden -File `"$Script`" -WakeTest" }
             else { "-NoProfile -WindowStyle Hidden -File `"$PollerScript`" -Once" }

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
    -Description 'Unattended SEPA weekend hunt: wakes the PC, runs the hunt the Pi requested (scripts\hunt\hunt_poller.ps1 -Once), pushes the result to the Pi.' |
    Out-Null

$info = Get-ScheduledTaskInfo -TaskName $name
Write-Host "registered '$name'; next run $($info.NextRunTime)"
