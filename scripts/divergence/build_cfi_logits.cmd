@echo off
rem Builds cfi_logits.exe against an existing llama.cpp CMake build (needs its src\llama.lib and ggml libs).
rem usage: build_cfi_logits.cmd <llama.cpp source dir> <llama.cpp build dir> <output exe>
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
cl /nologo /O2 /std:c++17 /EHsc /I%1\include /I%1\ggml\include %~dp0cfi_logits.cpp /Fe:%3 /link %2\src\llama.lib %2\ggml\src\ggml.lib %2\ggml\src\ggml-base.lib
