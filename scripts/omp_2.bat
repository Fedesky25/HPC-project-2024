@echo off
SETLOCAL ENABLEDELAYEDEXPANSION


    set OMP_NUM_THREADS= 6
    echo.
    echo ================================================================================ threads = 6

    for %%n in (39, 28, 23, 20, 18, 16, 15, 14) do (

        echo.
        echo =========================================================== d = %%n

        for /L %%i in (1,1,20) do (
            echo.
            echo # %%i/20
            .\HPC_project_2024 -p omp -R FHD -D 1 -f 20 -L 5 -d %%n fib
        )

    )

ENDLOCAL