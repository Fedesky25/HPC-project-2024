@echo off

@REM sofi (39, 23, 18, 15, 13, 12, 11, 10)
@REM fede (39, 19, 15, 12, 11, 10, 9, 8)

for %%n in (39, 19, 15, 12, 11, 10, 9, 8) do (
    echo.
    echo =========================================================== d = %%n
    for /L %%i in (1,1,20) do (
        echo.
        echo # %%i/20
        .\HPC_project_2024 -p gpu -R FHD -D 0 -L 5 -d %%n
    )
)