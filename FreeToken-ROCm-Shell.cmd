@echo off
rem Opens a console with the FreeToken ROCm environment ready (RX 9070, gfx1201).
rem Serve a model:  ft serve --model <path> --cuda-graph-max-bs 0
rem Or the helper:   powershell -File dist\run-server.ps1 -Model <path> -ExtraArgs "--cuda-graph-max-bs 0"
set "REPO=E:\freetoken-rocm"
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
set "TVM_FFI_CACHE_DIR=%REPO%\.tvm-ffi-cache"
set "PATH=%ROCM%\bin;%REPO%\.venv\Scripts;%PATH%"
set "CC=%ROCM%\lib\llvm\bin\clang-cl.exe"
cd /d "%REPO%"
echo FreeToken ROCm shell ready (gfx1201). Try: ft serve --model ^<path^> --cuda-graph-max-bs 0
cmd /k
