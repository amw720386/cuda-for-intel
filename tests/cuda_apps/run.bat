@echo off
rem Run all tests under the GPU lock protocol. Needs SHIM_DIR with the shim nvcuda.dll (+ cublas64_12.dll).
rem Usage: run.bat bindir shimdir cudart13dir cuda12bindir
set BIN=%~1& set SHIM=%~2& set C13=%~3& set C12=%~4
mkdir C:\Users\USER\cfi-work\gpu.lock 2>nul || (echo GPU LOCK HELD & exit /b 2)
set RD=%BIN%\run13& mkdir "%RD%" 2>nul
copy /y "%SHIM%\nvcuda.dll" "%RD%\" >nul& copy /y "%C13%\cudart64_13.dll" "%RD%\" >nul
for %%F in (basic_kernels atomics_warp streams_graphs) do copy /y "%BIN%\%%F.exe" "%RD%\" >nul
set RD2=%BIN%\run12& mkdir "%RD2%" 2>nul
copy /y "%SHIM%\nvcuda.dll" "%RD2%\" >nul& copy /y "%C12%\cudart64_12.dll" "%RD2%\" >nul
copy /y "%SHIM%\cublas64_12.dll" "%RD2%\" >nul& copy /y "%SHIM%\cublas64_12.dll" "%RD2%\cublasLt64_12.dll" >nul
for %%F in (cublas_gemm cublaslt_bias) do copy /y "%BIN%\%%F.exe" "%RD2%\" >nul
for %%F in (basic_kernels atomics_warp streams_graphs) do ( echo ===== %%F & pushd "%RD%" & "%RD%\%%F.exe" 2>&1 & echo EXIT=%errorlevel% & popd )
for %%F in (cublas_gemm cublaslt_bias) do ( echo ===== %%F & pushd "%RD2%" & "%RD2%\%%F.exe" 2>&1 & echo EXIT=%errorlevel% & popd )
rmdir C:\Users\USER\cfi-work\gpu.lock
