@echo off
SETLOCAL ENABLEDELAYEDEXPANSION

for %%n in (3, 6, 12) do (
    set OMP_NUM_THREADS=%%n

    echo.
    echo ================================================================================ threads = %%n

    echo.
    echo =========================================================== Fibonacci^(z^)
    for /L %%i in (1,1,20) do (
      echo.
      echo # %%i/20
      .\HPC_project_2024 -p omp -o videos\omp-fib.mp4 -R qHD -D 2 -L 5 fib
    )

    echo.
    echo =========================================================== exp^(z^^3^)
    for /L %%i in (1,1,20) do (
      echo.
      echo # %%i/20
      .\HPC_project_2024 -p omp -o videos\omp-cubic.mp4 -R qHD -D 2 -n 3 -L 5 "exp^n"
    )

    echo.
    echo =========================================================== Gamma^(z^)
    for /L %%i in (1,1,20) do (
      echo.
      echo # %%i/20
      .\HPC_project_2024 -p omp -o videos\omp-gamma.mp4 -R qHD -D 2 -L 5 --speed 0.1 gamma
    )
)

ENDLOCAL