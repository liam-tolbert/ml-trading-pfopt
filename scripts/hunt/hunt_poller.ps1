<#
.SYNOPSIS
Run weekend hunts the Pi asks for. Polls the Pi over ssh; nothing connects to this PC.

.DESCRIPTION
A request is data\cockpit\hunt\request.json on the Pi, written by the cockpit's Start
button or by the Pi's Friday timer. This script claims it (an atomic rename on the Pi),
runs weekend_hunt.ps1, and reports to the Pi's status.json as it goes: claimed, running
(with the run's last log line), then done or error. weekend_hunt.ps1 pushes the finished
folder itself and applies its own sleep rule, so a run claimed right after a timer wake
puts the PC back to sleep, and one claimed while you are at the keyboard does not.

Two tasks run it (register_task.ps1): "SEPA Hunt Poller" at logon, looping; and the
Friday 18:00 wake task with -Once. Both may see the same request; the claim makes sure
only one runs it. Log: data\cockpit\hunt\logs\poller.log.

.PARAMETER Once
Check for a request for up to three minutes (the network returns slowly after a wake),
run it if there is one, then exit. Exit code: that of the run, 0 when there was nothing.

.PARAMETER IntervalSeconds
Seconds between checks when looping. Default 30.
#>
[CmdletBinding()]
param([switch]$Once, [int]$IntervalSeconds = 30)

$ErrorActionPreference = 'Continue'

$Repo    = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$RepoFwd = $Repo -replace '\\', '/'
$LogDir  = Join-Path $Repo 'data\cockpit\hunt\logs'
$Log     = Join-Path $LogDir 'poller.log'
$Local   = Join-Path $Repo 'data\cockpit\hunt\status.json'
$HuntPs1 = Join-Path $PSScriptRoot 'weekend_hunt.ps1'
$OnceWaitSeconds = 180
$ReportEverySeconds = 60

function Write-Log([string]$Text) {
    New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
    $line = '{0}  {1}' -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $Text
    Add-Content -Path $Log -Value $line -Encoding UTF8
    Write-Host $line
}

# Git for Windows' launcher. MUST NOT resolve to System32\bash.exe, which is WSL.
function Find-Bash {
    $roots = @("$env:ProgramFiles\Git", "${env:ProgramFiles(x86)}\Git", "$env:LOCALAPPDATA\Programs\Git")
    $git = Get-Command git -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($git) {
        $dir = Split-Path $git.Source
        for ($i = 0; $i -lt 3 -and $dir; $i++) { $roots += $dir; $dir = Split-Path $dir }
    }
    foreach ($root in $roots) {
        $candidate = Join-Path $root 'bin\bash.exe'
        if (Test-Path $candidate) { return $candidate }
    }
    return $null
}

function Invoke-Bash([string]$Command) {
    $lines = @(& $script:Bash -lc $Command 2>&1 | ForEach-Object { "$_" })
    return [pscustomobject]@{ Code = $LASTEXITCODE; Lines = $lines }
}

# Tries to claim a request on the Pi. Returns the request object, or $null.
function Get-Claim {
    $r = Invoke-Bash "bash '$RepoFwd/scripts/hunt/pi_request.sh' claim"
    if ($r.Code -eq 3) { return $null }
    if ($r.Code -ne 0) { Write-Log "claim failed ($($r.Code)): $($r.Lines -join ' | ')"; return $null }
    try { return ($r.Lines -join "`n") | ConvertFrom-Json } catch { Write-Log "unreadable request: $($r.Lines -join ' ')"; return $null }
}

function Send-Status([string]$State, [string]$Date, [string]$Message, [string]$RequestedAt) {
    $body = [ordered]@{ state = $State; date = $Date; message = $Message;
                        requested_at = $RequestedAt
                        updated_at = (Get-Date -Format 'yyyy-MM-ddTHH:mm:ss') }
    New-Item -ItemType Directory -Force -Path (Split-Path $Local) | Out-Null
    [IO.File]::WriteAllText($Local, ($body | ConvertTo-Json), (New-Object Text.UTF8Encoding $false))
    $r = Invoke-Bash "bash '$RepoFwd/scripts/hunt/pi_request.sh' status '$($Local -replace '\\', '/')'"
    if ($r.Code -ne 0) { Write-Log "status not delivered: $($r.Lines -join ' | ')" }
}

function Get-LastLogLine([string]$Date) {
    $f = Join-Path $LogDir "$Date.log"
    if (-not (Test-Path $f)) { return '' }
    $last = Get-Content $f -Tail 1 -Encoding UTF8
    if ($last) { return ($last -replace '^\d\d:\d\d:\d\d\s+', '').Trim() } else { return '' }
}

# Runs one claimed request to its end, reporting as it goes. Returns the run's exit code.
function Invoke-Request($Req) {
    $date = Get-Date -Format 'yyyy-MM-dd'
    $asked = "$($Req.requested_at)"
    Write-Log "claimed a request from '$($Req.source)' made $asked"
    Send-Status 'claimed' $date 'starting the hunt' $asked
    # No -NoSleep: the run's own rule decides, from how the PC came to be awake.
    $proc = Start-Process -FilePath 'powershell.exe' -PassThru -NoNewWindow -WorkingDirectory $Repo `
        -ArgumentList @('-NoProfile', '-File', "`"$HuntPs1`"")
    $null = $proc.Handle
    while (-not $proc.WaitForExit($ReportEverySeconds * 1000)) {
        Send-Status 'running' $date (Get-LastLogLine $date) $asked
    }
    $code = $proc.ExitCode
    if ($code -eq 0) {
        Send-Status 'done' $date 'the hunt is on the Pi' $asked
    } else {
        $why = "exit code $code"
        $failed = Join-Path $Repo "data\cockpit\hunt\$date\FAILED.txt"
        if (Test-Path $failed) { $why = (Get-Content $failed -First 1 -Encoding UTF8) }
        Send-Status 'error' $date $why $asked
    }
    Write-Log "run $date finished with exit code $code"
    return $code
}

# ---------------------------------------------------------------------------------------
$script:Bash = Find-Bash
if (-not $script:Bash) { Write-Log 'Git Bash not found'; exit 1 }
Write-Log ("poller started " + $(if ($Once) { '(once)' } else { "(every $IntervalSeconds s)" }))

if ($Once) {
    $deadline = (Get-Date).AddSeconds($OnceWaitSeconds)
    $req = $null
    while (-not $req -and (Get-Date) -lt $deadline) {
        $req = Get-Claim
        if (-not $req) { Start-Sleep -Seconds 15 }
    }
    if (-not $req) { Write-Log 'no request on the Pi'; exit 0 }
    exit (Invoke-Request $req)
}

while ($true) {
    $req = Get-Claim
    if ($req) { [void](Invoke-Request $req) }
    Start-Sleep -Seconds $IntervalSeconds
}
