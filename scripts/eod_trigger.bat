@echo off
rem Nightly SEPA EOD trigger check + full_us cache prewarm (see HANDOFF section 6.14).
rem Scheduled by Windows Task Scheduler ("SEPA EOD Trigger", weekdays 17:30 local).
cd /d "C:\Users\Unity\Documents\Liam\School\UVA\ME SE\1st Year\Independent Study\ml-trading-pfopt"
if not exist "data\cockpit\triggers" mkdir "data\cockpit\triggers"
echo ---- %date% %time% ---- >> "data\cockpit\triggers\eod_trigger.log"
"C:\Users\Unity\miniconda3\envs\ml-trading\python.exe" "src\stock_screener\cockpit\eod_trigger.py" --prewarm >> "data\cockpit\triggers\eod_trigger.log" 2>&1
