@echo off
rem ROCm/Windows rebuild: C++ extensions (setup.py build_ext --inplace), then the ggml JIT kernels.
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
rem MSVC cl cannot parse the HIP headers: compile with clang-cl (recipe from the 2026-11-03 build).
set "CXX=%ROCM%\lib\llvm\bin\clang-cl.exe"
set "DISTUTILS_USE_SDK=1"
set "MSSdk=1"
cd /d "%SRC%"
if /i "%~1"=="jit" goto :jit
echo [1/2] C++ extensions (setup.py build_ext --inplace)
"%REPO%\.venv\Scripts\python.exe" setup.py build_ext --inplace || exit /b 1
:jit
rem the tvm-ffi JIT links with lld-link when CXX is clang-cl and fails (subsystem must be defined)
set "CXX="
echo [2/2] ggml kernels (JIT, cached in %TVM_FFI_CACHE_DIR%)
"%REPO%\.venv\Scripts\python.exe" -c "import time,torch;t=time.time();from freetoken.kernel.ggml import _module;_module();import freetoken.kernel._cpu_moe as m;print('ggml ok',round(time.time()-t),'s; cpu_moe',m.__file__)" || exit /b 1
echo done
