' Run a PowerShell script with no window at all. The scheduled tasks start their scripts
' through this, because powershell.exe -WindowStyle Hidden still creates a console that can
' appear and be closed, which kills the poller or a hunt in progress.
'     wscript.exe //B //Nologo hidden.vbs <script.ps1> [args...]
Option Explicit
Dim shell, args, cmd, i
Set shell = CreateObject("WScript.Shell")
Set args = WScript.Arguments
If args.Count < 1 Then
    WScript.Echo "usage: hidden.vbs <script.ps1> [args...]"
    WScript.Quit 2
End If
cmd = "powershell.exe -NoProfile -ExecutionPolicy RemoteSigned -File """ & args(0) & """"
For i = 1 To args.Count - 1
    cmd = cmd & " " & args(i)
Next
' 0 = hidden window; True = wait, so the task shows Running while the script runs and
' reports the script's exit code.
WScript.Quit shell.Run(cmd, 0, True)
