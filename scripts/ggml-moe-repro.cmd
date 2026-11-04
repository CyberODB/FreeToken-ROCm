@echo off
rem Isolated GGUF MoE kernel repro (ggml mul_mat_id) on the RX 9070, with the same ROCm env as bin\ft.cmd.
rem Close the FreeToken TEST app first so the GPU is free. Args go to scripts\ggml_moe_repro.py (--help).
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
rem serialize kernels so a fault is reported at the kernel that caused it
set "AMD_SERIALIZE_KERNEL=3"
"%REPO%\.venv\Scripts\python.exe" "%SRC%\scripts\ggml_moe_repro.py" %*
set "RC=%ERRORLEVEL%"
echo.
echo exit code %RC%. Results are also in %SRC%\logs\
if "%~1"=="" pause
exit /b %RC%
