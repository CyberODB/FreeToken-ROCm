@echo off
rem Phase 0 decode measurement (2026-11-06, plan in T:\robot\docs\18-freetoken-rocm-speed-plan.md).
rem Usage: FreeToken-Desktop-021-measure.cmd [mode]   (default: stamps)
rem   off      baseline tok/s, nothing added (to see what the stamps cost)
rem   stamps   [graph-stamps] in serve-1919.log every 64 decode steps
rem   fetchall stamps + every miss fetched over PCIe, no CPU routes (--moe-hybrid-max-fetch 8)
rem   cpuall   stamps + every miss computed on the CPU (--moe-hybrid-max-fetch 0)
rem   serial   stamps + CPU and GPU MoE work serialized (FREETOKEN_HYBRID_OVERLAP=0)
rem Locked layers keep their own fetch setting in every mode. One launch per mode; graphs stay on (bs 1).
setlocal
set "MODE=%~1"
if "%MODE%"=="" set "MODE=stamps"
set "FT_MEASURE_ARGS="
if /i "%MODE%"=="off" goto run
set "FREETOKEN_GRAPH_STAMPS=64"
if /i "%MODE%"=="stamps" goto run
if /i "%MODE%"=="fetchall" (set "FT_MEASURE_ARGS=--moe-hybrid-max-fetch 8" & goto run)
if /i "%MODE%"=="cpuall" (set "FT_MEASURE_ARGS=--moe-hybrid-max-fetch 0" & goto run)
if /i "%MODE%"=="serial" (set "FREETOKEN_HYBRID_OVERLAP=0" & goto run)
echo Unknown mode "%MODE%". Use off, stamps, fetchall, cpuall or serial.
exit /b 2
:run
echo [measure] mode %MODE%: FREETOKEN_GRAPH_STAMPS=%FREETOKEN_GRAPH_STAMPS% FT_MEASURE_ARGS=%FT_MEASURE_ARGS% FREETOKEN_HYBRID_OVERLAP=%FREETOKEN_HYBRID_OVERLAP%
if defined FT_MEASURE_DRYRUN exit /b 0
call "%~dp0FreeToken-Desktop-021.cmd" 1
