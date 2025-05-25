@echo off
SETLOCAL ENABLEDELAYEDEXPANSION

echo.
echo ======================================================================================================= threads = 1
@REM for %%p in (4341,5908,7797,10880,14179,16464,19383,23200,28330,35464,40128,45810) do (
for %%p in (4341,5908) do (
    echo.
    echo ============== p = %%p
    for /L %%i in (1,1,20) do (
        echo.
        echo # %%i/20
        .\HPC_project_2024 -p none -o videos\serial.mp4 -R FHD -D 2 -L p%%p.txt fib
    )
)

set OMP_NUM_THREADS=6
echo.
echo ======================================================================================================= threads = 6
for %%p in (4341,5908,7797,10880,14179,16464,19383,23200,28330,35464,40128,45810) do (
    echo.
    echo ============== p = %%p
    for /L %%i in (1,1,20) do (
        echo.
        echo # %%i/20
        .\HPC_project_2024 -p omp -o videos\omp6.mp4 -R FHD -D 2 -L p%%p.txt fib
    )
)

set OMP_NUM_THREADS=12
echo.
echo ======================================================================================================= threads = 12
for %%p in (4341,5908,7797,10880,14179,16464,19383,23200,28330,35464,40128,45810) do (
    echo.
    echo ============== p = %%p
    for /L %%i in (1,1,20) do (
        echo.
        echo # %%i/20
        .\HPC_project_2024 -p omp -o videos\omp12.mp4 -R FHD -D 2 -L p%%p.txt fib
    )
)

set OMP_NUM_THREADS=12
echo.
echo ======================================================================================================= threads = CUDA
for %%p in (4341,5908,7797,10880,14179,16464,19383,23200,28330,35464,40128,45810) do (
    echo.
    echo ============== p = %%p
    for /L %%i in (1,1,20) do (
        echo.
        echo # %%i/20
        .\HPC_project_2024 -p gpu -o videos\cuda.mp4 -R FHD -D 2 -L p%%p.txt fib
    )
)

ENDLOCAL