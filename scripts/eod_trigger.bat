@echo off
rem Watchlist trigger check (see HANDOFF section 6.18). Scheduled by Windows Task
rem Scheduler ("SEPA Intraday Trigger": weekdays, every 30 min 09:30-16:30 local).
rem Extra args pass through to the python CLI (e.g. --date / --no-write).
cd /d "C:\Users\Unity\Documents\Liam\School\UVA\ME SE\1st Year\Independent Study\ml-trading-pfopt"
if not exist "data\cockpit\triggers" mkdir "data\cockpit\triggers"
echo ---- %date% %time% ---- >> "data\cockpit\triggers\eod_trigger.log"
"C:\Users\Unity\miniconda3\envs\ml-trading\python.exe" "src\stock_screener\cockpit\eod_trigger.py" %* >> "data\cockpit\triggers\eod_trigger.log" 2>&1
