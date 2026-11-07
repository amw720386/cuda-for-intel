@echo off
rem Builds every test with PTX-only embedding (compute_75). Usage: build.bat [outdir]
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
if "%CUDA%"=="" set CUDA=%USERPROFILE%\cfi-work\torchbuild\cuda
set OUT=%1
if "%OUT%"=="" set OUT=%~dp0bin
mkdir "%OUT%" 2>nul
for %%F in (basic_kernels atomics_warp streams_graphs) do (
  "%CUDA%\bin\nvcc.exe" -allow-unsupported-compiler -gencode arch=compute_75,code=compute_75 -cudart shared -Wno-deprecated-gpu-targets -I"%CUDA%\include" -L"%CUDA%\lib\x64" -o "%OUT%\%%F.exe" "%~dp0%%F.cu" || echo BUILD_FAIL %%F
)
rem cuBLAS/cuBLASLt tests are host-only: build with cl against the CUDA 12.x headers/libs (the shim provides cublas64_12.dll).
rem (nvcc 12.6's cudafe++ crashes on MSVC 14.51 STL headers, so kernel tests use CUDA 13.x nvcc, see README.)
if "%CUDA12%"=="" set CUDA12=%USERPROFILE%\cfi-work\torchbuild\cuda
for %%F in (cublas_gemm cublaslt_bias) do (
  cl /nologo /EHsc /O2 /TP /I"%CUDA12%\include" /Fe:"%OUT%\%%F.exe" /Fo:"%OUT%\%%F.obj" "%~dp0%%F.cu" /link /LIBPATH:"%CUDA12%\lib\x64" cudart.lib cublas.lib cublasLt.lib || echo BUILD_FAIL %%F
)
