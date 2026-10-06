<#
.SYNOPSIS
Unattended weekend hunt: pull the Pi's scan, run the weekend-hunt skill headless, leave the report.

.DESCRIPTION
Entry point of the "SEPA Weekend Hunt" scheduled task (register_task.ps1) and of the hunt
API (hunt_api.py). Runs on this Windows box only. The Pi is read for its scan and written
once per run: the finished hunt folder is pushed to its data\cockpit\hunt\<date>\, where
the cockpit's Weekend Hunt page reads it.

The deliverable is docs\hunt\<date>\report.html with its charts\ beside it, and the same
hunt folder on the Pi. The working state stays in data\cockpit\hunt\<date>\: summary.md
(the reviewer's closing message), and FAILED.txt when the run did not finish.
Log: data\cockpit\hunt\logs\<date>.log.
Exit code: 0 when the run left a current report and pushed it, 1 otherwise.

.PARAMETER NoSleep
Leave the PC awake afterwards, whatever woke it.

.PARAMETER SkipPull
Hunt off the scan already in data\cockpit instead of copying the Pi's.

.PARAMETER NoPush
Leave the result on this PC; do not copy the hunt folder to the Pi.

.PARAMETER WakeTest
Skip the hunt. Hold the PC awake for three minutes, then apply the sleep rule. Proves the
wake timer, the keep-awake and the return to sleep without spending a review.
#>
[CmdletBinding()]
param([switch]$NoSleep, [switch]$SkipPull, [switch]$NoPush, [switch]$WakeTest)

# Native stderr MUST NOT abort the run: Windows PowerShell 5.1 turns it into error records.
$ErrorActionPreference = 'Continue'

$Repo     = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$RepoFwd  = $Repo -replace '\\', '/'
$Started  = Get-Date
$Date     = $Started.ToString('yyyy-MM-dd')
$HuntDir  = Join-Path $Repo "data\cockpit\hunt\$Date"
$LogDir   = Join-Path $Repo 'data\cockpit\hunt\logs'
$Log      = Join-Path $LogDir "$Date.log"
$Failed   = Join-Path $HuntDir 'FAILED.txt'
$Report   = Join-Path $Repo "docs\hunt\$Date\report.html"
$Summary  = Join-Path $HuntDir 'summary.md'
$Settings = Join-Path $PSScriptRoot 'unattended_settings.json'
$Prompt   = Join-Path $PSScriptRoot 'unattended_prompt.txt'

$ReviewMinutes     = 90   # one headless review is stopped after this long
$ReviewAttempts    = 2    # the skill resumes from the tickers still missing a verdict
$WakeWindowMinutes = 10   # a resume this close to the start counts as a wake for the run
$WakeTestSeconds   = 180  # longer than Windows' 2-minute unattended sleep timeout

# 0x80000000 is a negative Int32 literal in PowerShell, hence the decimals.
$ES_CONTINUOUS      = [uint32]2147483648
$ES_SYSTEM_REQUIRED = [uint32]1

Add-Type -Namespace Hunt -Name Win32 -MemberDefinition @'
[DllImport("kernel32.dll")]
public static extern uint SetThreadExecutionState(uint esFlags);
[DllImport("kernel32.dll")]
public static extern uint GetTickCount();
[StructLayout(LayoutKind.Sequential)]
public struct LASTINPUTINFO { public uint cbSize; public uint dwTime; }
[DllImport("user32.dll")]
public static extern bool GetLastInputInfo(ref LASTINPUTINFO plii);
'@

function Write-Log([string]$Text) {
    $line = '{0}  {1}' -f (Get-Date -Format 'HH:mm:ss'), $Text
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

# Runs $Command in a login bash. Returns Code (exit code) and Lines (stdout and stderr);
# every line is also logged.
function Invoke-Bash([string]$Command) {
    $lines = @(& $script:Bash -lc $Command 2>&1 | ForEach-Object { "$_" })
    $code = $LASTEXITCODE
    foreach ($line in $lines) { Write-Log "    $line" }
    return [pscustomobject]@{ Code = $code; Lines = $lines }
}

# Seconds since the last keyboard or mouse input in this session. 0 when it cannot be read,
# which reads as "someone is here".
function Get-IdleSeconds {
    $info = New-Object 'Hunt.Win32+LASTINPUTINFO'
    $info.cbSize = [uint32][Runtime.InteropServices.Marshal]::SizeOf($info)
    if (-not [Hunt.Win32]::GetLastInputInfo([ref]$info)) { return 0 }
    $ms = ([int64][Hunt.Win32]::GetTickCount() - [int64]$info.dwTime + 4294967296) % 4294967296
    return $ms / 1000.0
}

# When the PC resumed for this run, or $null if it was already on. Read at the end of the
# run: the resume event can be logged after the task has started.
function Get-RunWake {
    try {
        $resume = Get-WinEvent -MaxEvents 1 -ErrorAction Stop -FilterHashtable @{
            LogName = 'System'; ProviderName = 'Microsoft-Windows-Power-Troubleshooter'; Id = 1 }
    } catch { return $null }
    if ($resume.TimeCreated -lt $Started.AddMinutes(-$WakeWindowMinutes)) { return $null }
    if ($resume.TimeCreated -gt $Started.AddMinutes(2)) { return $null }
    return $resume.TimeCreated
}

# One headless review. Returns claude's exit code, or -1 when it was stopped for running long.
function Invoke-Review([string]$Claude, [int]$Attempt) {
    $out = Join-Path $HuntDir "review_$Attempt.out.txt"
    $err = Join-Path $HuntDir "review_$Attempt.err.txt"
    # The allowlist in $Settings is the safety boundary: dontAsk refuses everything else.
    $argv = @('-p', '--permission-mode', 'dontAsk', '--settings', "`"$Settings`"", '--strict-mcp-config')
    $proc = Start-Process -FilePath $Claude -ArgumentList $argv -WorkingDirectory $Repo `
        -NoNewWindow -PassThru -RedirectStandardInput $Prompt `
        -RedirectStandardOutput $out -RedirectStandardError $err
    $null = $proc.Handle    # without this, 5.1 loses ExitCode
    if (-not $proc.WaitForExit($ReviewMinutes * 60000)) {
        Write-Log "review still running after $ReviewMinutes min; stopping it"
        & taskkill.exe /PID $proc.Id /T /F | Out-Null
        return -1
    }
    foreach ($line in @(Get-Content $err -Encoding UTF8 -ErrorAction SilentlyContinue)) {
        Write-Log "    stderr: $line"
    }
    if ((Test-Path $out) -and (Get-Item $out).Length -gt 0) {
        Copy-Item $out $Summary -Force
    }
    return $proc.ExitCode
}

function Invoke-Hunt {
    Remove-Item $Failed -ErrorAction SilentlyContinue
    $script:Bash = Find-Bash
    if (-not $script:Bash) { throw 'Git Bash not found (Git for Windows is required).' }
    $huntSh = "bash '$RepoFwd/scripts/hunt/hunt.sh'"

    if (-not $SkipPull) {
        if ((Invoke-Bash "bash '$RepoFwd/scripts/hunt/pull_scan.sh'").Code -ne 0) {
            throw 'could not pull the scan from the Pi.'
        }
    }
    $status = Invoke-Bash "$huntSh status"
    if ($status.Code -ne 0) { throw 'the scan is missing or more than 3 days old.' }
    $note = ''
    $scanDay = ($status.Lines -join "`n") -replace '(?s).*"scan_time":\s*"(\d{4}-\d{2}-\d{2}).*', '$1'
    if ($scanDay -match '^\d{4}-\d{2}-\d{2}$' -and $scanDay -ne $Date) {
        $note = "WARNING: the scan is from $scanDay, not today ($Date). The Pi's EOD screen may not have run."
        Write-Log $note
    }

    $claude = Get-Command claude -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $claude) { throw 'claude CLI not found on PATH (npm install -g @anthropic-ai/claude-code).' }
    if (-not $env:CLAUDE_CODE_OAUTH_TOKEN -and -not $env:ANTHROPIC_API_KEY) {
        Write-Log 'no CLAUDE_CODE_OAUTH_TOKEN in the environment; relying on a stored claude login'
    }

    $valid = $false
    for ($attempt = 1; $attempt -le $ReviewAttempts -and -not $valid; $attempt++) {
        Write-Log "review attempt $attempt of $ReviewAttempts"
        $code = Invoke-Review $claude.Source $attempt
        Write-Log "claude exited $code"
        $valid = (Invoke-Bash "$huntSh validate-verdicts --date $Date").Code -eq 0
    }
    if (-not $valid) { throw 'the review did not leave one verdict per candidate.' }

    $built = Get-Item $Report -ErrorAction SilentlyContinue
    if (-not $built -or $built.LastWriteTime -lt $Started) {
        Write-Log 'the review did not build the report; building it now'
        if ((Invoke-Bash "$huntSh report --date $Date").Code -ne 0) {
            throw 'building the report failed.'
        }
    }
    if ($note) {
        $body = ''
        if (Test-Path $Summary) { $body = Get-Content $Summary -Raw -Encoding UTF8 }
        Set-Content -Path $Summary -Value "$note`r`n`r`n$body" -Encoding UTF8
    }
    if (-not $NoPush) {
        if ((Invoke-Bash "bash '$RepoFwd/scripts/hunt/push_result.sh' $Date").Code -ne 0) {
            throw 'pushing the result to the Pi failed.'
        }
    }
    Write-Log "done: $Report"
}

# ---------------------------------------------------------------------------------------
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
if (-not $WakeTest) { New-Item -ItemType Directory -Force -Path $HuntDir | Out-Null }
$exitCode = 1

# Held for the whole run: after a timer wake with nobody present, Windows sleeps again
# after two minutes unless the system is marked required.
[void][Hunt.Win32]::SetThreadExecutionState($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED)
try {
    Write-Log "weekend hunt started ($Repo)"
    if ($WakeTest) {
        Write-Log "wake test: holding the PC awake for $WakeTestSeconds s"
        Start-Sleep -Seconds $WakeTestSeconds
    } else {
        Invoke-Hunt
    }
    $exitCode = 0
} catch {
    Write-Log "FAILED: $($_.Exception.Message)"
    if (-not $WakeTest) {
        Set-Content -Path $Failed -Encoding UTF8 -Value @(
            "The weekend hunt of $Date did not finish: $($_.Exception.Message)", "Log: $Log")
    }
} finally {
    [void][Hunt.Win32]::SetThreadExecutionState($ES_CONTINUOUS)
}

if ($NoSleep) {
    Write-Log 'staying awake (-NoSleep)'
} else {
    $wake = Get-RunWake
    if (-not $wake) {
        Write-Log 'staying awake: the PC was already on when the run started'
    } elseif ((Get-IdleSeconds) -lt ((Get-Date) - $wake).TotalSeconds) {
        Write-Log 'staying awake: keyboard or mouse used since the wake'
    } else {
        Write-Log "going back to sleep (woke $($wake.ToString('HH:mm:ss')), untouched since)"
        Add-Type -AssemblyName System.Windows.Forms
        [void][System.Windows.Forms.Application]::SetSuspendState(
            [System.Windows.Forms.PowerState]::Suspend, $false, $false)
    }
}
exit $exitCode
