@echo off

@REM sofi (39, 33, 29, 26, 24, 23, 21, 20)
@REM fede (39, 37, 34, 32, 31, 29, 28, 27)

for %%n in (39, 37, 34, 32, 31, 29, 28, 27) do (
    echo.
    echo =========================================================== d = %%n
    for /L %%i in (1,1,20) do (
      echo.
      echo # %%i/20
      .\HPC_project_2024 -p none -o videos\serial-fib.mp4 -R FHD -D 1 -f 20 -L 5 -d %%n fib
    )
)