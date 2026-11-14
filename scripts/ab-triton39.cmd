@echo off
rem A/B: Triton 3.8 (venv) vs Triton 3.9 (E:\freetoken-triton39), same 35B model, hybrid 0.85, stamps on.
rem Run with AI on, no model loaded, games closed. ~15 min. Results: speedprobe\results.jsonl, names t38-* / t39-*.
rem Order is 38, 39, 39, 38 so drift cancels. The first 39 run recompiles kernels; judge it by the second.
rem Output identity: compare the "text" field of t38-1 and t39-2 in results.jsonl.
setlocal
set "SP=E:\freetoken-021\scripts\ft-speedprobe.py"
set "ARGS=--moe-strategy hybrid --memory-ratio 0.85"
set "ENVS=FT_AB=1"
set "FT_LAUNCHER=ft.cmd"
py -u "%SP%" t38-1 --env %ENVS% --args "%ARGS%" --tests decode,p5k || exit /b 1
set "FT_LAUNCHER=ft39.cmd"
py -u "%SP%" t39-1 --env %ENVS% --args "%ARGS%" --tests decode,p5k || exit /b 1
py -u "%SP%" t39-2 --env %ENVS% --args "%ARGS%" --tests decode,p5k || exit /b 1
set "FT_LAUNCHER=ft.cmd"
py -u "%SP%" t38-2 --env %ENVS% --args "%ARGS%" --tests decode,p5k || exit /b 1
echo done
