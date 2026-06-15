@echo off
REM Build a CUDA source with the pinned MSVC 14.38 toolset.
REM 14.38 is the newest installed x64 toolset whose STL still accepts CUDA 12.2
REM (14.40+ adds the STL1002 "needs CUDA 12.4+" static_assert).
REM Usage: build.bat <source.cu> <output.exe>
call "D:\Applications\VisualStudio\VS\VC\Auxiliary\Build\vcvars64.bat" -vcvars_ver=14.38 >nul
nvcc %1 -o %2 -O2 -arch=sm_89 -ccbin "%VCToolsInstallDir%bin\HostX64\x64" %3 %4 %5 %6 %7 %8
