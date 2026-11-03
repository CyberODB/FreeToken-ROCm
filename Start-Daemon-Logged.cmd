@echo off
rem Starts the ROCm daemon (port 19000) with all output going to E:\freetoken-rocm\logs\daemon.log
call "E:\freetoken-rocm\bin\ft.cmd" daemon --host 127.0.0.1 --port 19000 > "E:\freetoken-rocm\logs\daemon.log" 2>&1
