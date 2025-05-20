@echo off

for %%n in (39, 23, 18, 15, 13, 12, 11, 10) do (

    echo.
    echo =========================================================== d = %%n

    for /L %%i in (1,1,20) do (
        echo.
        echo # %%i/20
        .\HPC_project_2024 -p gpu -R FHD -D 1 -f 20 -L 5 -d %%n fib
    )

)