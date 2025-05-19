@echo off
SETLOCAL ENABLEDELAYEDEXPANSION

set OMP_NUM_THREADS=12
echo.
    echo ================================================================================ threads = 12

    echo.
    echo =========================================================== Fibonacci^(z^)
      echo.
      .\HPC_project_2024 -p omp -R FHD -D 1 -f 20 -L 5 -d 12 fib
    )

ENDLOCAL