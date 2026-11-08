@echo off
rem Launches the FreeToken desktop app using the ROCm fork's engine (RX 9070, gfx1201). Testing only.
set "REPO=E:\freetoken-rocm"
set "SRC=E:\freetoken-021"
set "PYTHONPATH=%SRC%\python"
set "FREETOKEN_ROCM_ARCH=gfx1201"
set "CPATH=%REPO%\.venv\Lib\site-packages\_rocm_sdk_devel\include"
set "ROCM=%REPO%\.venv\Lib\site-packages\_rocm_sdk_core"
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64 >nul
set "HIP_PATH=%ROCM%"
set "ROCM_HOME=%ROCM%"
set "ROCM_PATH=%ROCM%"
set "TVM_FFI_ROCM_ARCH_LIST=gfx1201"
set "PYTORCH_ROCM_ARCH=gfx1201"
set "TRITON_OVERRIDE_ARCH=gfx1201"
set "ROCM_SDK_TARGET_FAMILY=gfx1201"
set "HIP_DEVICE_LIB_PATH=%ROCM%\lib\llvm\amdgcn\bitcode"
set "TVM_FFI_CACHE_DIR=%SRC%\.tvm-ffi-cache"
set "PATH=%ROCM%\bin;%REPO%\.venv\Lib\site-packages\_rocm_sdk_libraries\bin;%REPO%\.venv\Scripts;%PATH%"
set "CC=%ROCM%\lib\llvm\bin\clang-cl.exe"
set "FREETOKEN_FT_BIN=%SRC%\bin\ft.cmd"
rem CUDA graph ladder for the daemon: 0 = off, then 1, 2, 4. See serve_manager.py. Default 1 since 2026-11-05.
set "FT_GRAPH_MAX_BS=1"
rem 2026-11-05: graphs in hybrid decode via the kernel spin-wait handshake (test B: 47-58 tok/s vs 12-14.5 off).
rem An argument sets the ladder top: FreeToken-Desktop-021.cmd 0 turns graphs off.
if not "%~1"=="" set "FT_GRAPH_MAX_BS=%~1"
rem Diagnostic 2026-11-03: serialize kernels so a GPU fault names the real kernel. Slow; remove after.
rem Off since 2026-11-03 22:41 (hybrid works); remove "rem " to turn back on for debugging.
rem set "AMD_SERIALIZE_KERNEL=3"
rem set "FT_DEBUG_GLU=1"
rem Diagnostic 2026-11-03: check prefill MoE GEMM inputs (ids, view, MMQ zero tail) before launch.
rem set "FT_DEBUG_MMQ=1"
rem 2026-11-03: hybrid. auto + a fixed pin budget picks the split from the budget (no pin probe), so the
rem PINNED layers can be born pinned (host_banks.backing_for). 12 GiB = the code's 40%-of-RAM WDDM rule.
rem Test 2026-11-04: 14 GiB (driver refused near ~14 in the 11-03 21:40 pin probe). 7 CPU layers instead of 12.
rem Undo: set back to 12. A refused pin is a load error with a fixed budget.
rem FT_MEASURE_ARGS comes from FreeToken-Desktop-021-measure.cmd (empty otherwise).
set "FT_EXTRA_SERVE_ARGS=--moe-cpu-layers auto %FT_MEASURE_ARGS%"
rem Test 2026-11-06: 15 GiB (was 14): one more GiB pinned instead of locked, same total RAM. Driver refused near ~14
rem in the 11-03 probe, so a load error here means back to 14. Undo: set back to 14.
set "FREETOKEN_PIN_BUDGET_GB=15"
rem 2026-11-06: RAM check before each load (python\freetoken\ram_preflight.py). 14 GiB = desktop with a YouTube video
rem and Claude running, 11.8 measured by T:\robot\tools\tools\ram-baseline.py, + 2 margin. If the model won't fit in
rem free RAM a window says why and offers to close programs, then loads anyway. Undo: delete this line.
set "FREETOKEN_HOST_RESERVE_GB=14"
rem Diagnostic 2026-11-03 A/B: expert banks born pinned (hipHostMalloc), not mmap + hipHostRegister. Remove after.
set "FREETOKEN_BANK_CUDA_ALLOC=1"
rem Measurement 2026-11-03: per-phase decode timing, logged as [hybrid-timing] every 16 steps in serve-1919.log.
rem Slows decode (syncs the GPU around every phase, overlap off). Off since 2026-11-03 23:35; remove "rem " on both lines to turn back on.
rem set "FREETOKEN_HYBRID_TIMING=16"
rem set "FREETOKEN_HYBRID_OVERLAP=0"
rem Measurement 2026-11-04: time 32 decode steps once (after 24 warm-up steps): CPU enqueue vs GPU finish,
rem logged in serve-1919.log as [decode-profile]. FREETOKEN_DECODE_PROFILE_MODE=torch gives the torch.profiler table instead.
rem set "FREETOKEN_DECODE_PROFILE=32"
rem Measurement 2026-11-06: in-graph GPU timestamps, [graph-stamps] in serve-1919.log every 64 decode steps:
rem ms per phase (summed over layers), host gap, hits / PCIe fetches / CPU routes per layer. Graphs stay on.
rem Adds ~200 tiny kernels per step, so compare tok/s with it off. Remove "rem " to turn on.
rem set "FREETOKEN_GRAPH_STAMPS=64"
rem 2026-11-07: GPU guard for every test (scripts\gpu-guard.py). --check refuses a test that was running when the PC died;
rem the hidden watcher kills the FreeToken GPU processes if free VRAM stays under FT_GUARD_MIN_FREE_GIB (default 1.0) and
rem journals VRAM to FT_GUARD_DIR so a hard reset leaves a trail. A launcher may set FT_GUARD_NAME first. Undo: delete these 7 lines.
set "FT_GUARD_DIR=T:\robot\state\gpu-guard"
if not defined FT_GUARD_NAME set "FT_GUARD_NAME=desktop-graphs%FT_GRAPH_MAX_BS%"
python "%SRC%\scripts\gpu-guard.py" --check --name "%FT_GUARD_NAME%" || (pause & exit /b 1)
start "" /b pythonw "%SRC%\scripts\gpu-guard.py" --watch --name "%FT_GUARD_NAME%"
powershell -NoProfile -ExecutionPolicy Bypass -File "E:\freetoken-rocm\guard-daemon.ps1" -Build "TEST" || exit /b 1
rem freetoken-desktop-test.exe is a patched copy (orange tray icon, "FreeToken TEST" label) made by
rem E:\freetoken-rocm\patch_tray.py. It gets no app updates: re-run the patch after updating FreeToken Desktop.
start "" "C:\Users\CyberODB\AppData\Local\FreeToken Desktop\freetoken-desktop-test.exe"
