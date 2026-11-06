@echo off
rem ROCm shim for FreeToken Desktop: sets MSVC + ROCm env, then runs the ROCm-fork ft.
rem For "serve" it adds --cuda-graph-max-bs 0 (gfx1201 graph-replay crash workaround) before the caller's args.
setlocal
set "REPO=E:\freetoken-rocm"
set "SRC=E:\freetoken-021"
set "PYTHONPATH=%SRC%\python"
set "FREETOKEN_ROCM_ARCH=gfx1201"
set "CPATH=%REPO%\.venv\Lib\site-packages\_rocm_sdk_devel\include"
set "ROCM=%REPO%\.venv\Lib\site-packages\_rocm_sdk_core"
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64 >nul 2>&1
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
set "ARGS=%*"
rem %ARGS% inside a ( ) block expands before the block runs, so strip "serve" here, outside it (fixed 2026-11-05).
if /i "%~1"=="serve" goto :serve
"%REPO%\.venv\Scripts\ft.exe" %ARGS%
exit /b %ERRORLEVEL%
:serve
set "ARGS=%ARGS:~5%"
"%REPO%\.venv\Scripts\ft.exe" serve --cuda-graph-max-bs 0 %ARGS%
exit /b %ERRORLEVEL%
