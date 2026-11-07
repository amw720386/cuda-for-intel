@echo off
rem Run all tests under the GPU lock protocol. Needs SHIM_DIR with the shim nvcuda.dll (+ cublas64_12.dll).
rem Env: HYBRID_SRC = path of the signed nvcudart_hybrid64.dll (SHA-256 BF430D65...22B3); CUDA_FOR_INTEL_TRACE=1 for traces.
rem Usage: run.bat bindir shimdir cudart13dir cuda12bindir
set BIN=%~1
set SHIM=%~2
set C13=%~3
set C12=%~4
mkdir %USERPROFILE%\cfi-work\gpu.lock 2>nul || (echo GPU LOCK HELD & exit /b 2)
mkdir "%BIN%\hybrid" 2>nul
copy /y "%HYBRID_SRC%" "%BIN%\hybrid\nvcudart_hybrid64.dll" >nul
set CUDA_FOR_INTEL_HYBRID_DIR=%BIN%\hybrid
set RD=%BIN%\run13
mkdir "%RD%" 2>nul
copy /y "%SHIM%\nvcuda.dll" "%RD%\" >nul
copy /y "%C13%\cudart64_13.dll" "%RD%\" >nul
copy /y "%SHIM%\cublas64_12.dll" "%RD%\" >nul
for %%F in (basic_kernels atomics_warp streams_graphs algos half_bf16 mem_symbols mem_copies transformer_block) do copy /y "%BIN%\%%F.exe" "%RD%\" >nul
set RD2=%BIN%\run12
mkdir "%RD2%" 2>nul
copy /y "%SHIM%\nvcuda.dll" "%RD2%\" >nul
copy /y "%C12%\cudart64_12.dll" "%RD2%\" >nul
copy /y "%SHIM%\cublas64_12.dll" "%RD2%\" >nul
copy /y "%SHIM%\cublas64_12.dll" "%RD2%\cublasLt64_12.dll" >nul
for %%F in (cublas_gemm cublaslt_bias) do copy /y "%BIN%\%%F.exe" "%RD2%\" >nul
for %%F in (basic_kernels atomics_warp streams_graphs algos half_bf16 mem_symbols mem_copies transformer_block) do call :one "%RD%" %%F
for %%F in (cublas_gemm cublaslt_bias) do call :one "%RD2%" %%F
rmdir %USERPROFILE%\cfi-work\gpu.lock
exit /b 0
:one
echo ===== %2
pushd %1
%2.exe 2>&1
echo EXIT=%errorlevel%
popd
exit /b 0
