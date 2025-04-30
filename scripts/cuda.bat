@echo off

echo.
echo =========================================================== Fibonacci(z)
for /L %%i in (1,1,20) do (
  echo.
  echo # %%i/20
  .\HPC_project_2024 -p gpu -o videos\cuda-fib.mp4 -R qHD -D 2 -L 5 fib
)

echo.
echo =========================================================== exp(z^3)
for /L %%i in (1,1,20) do (
  echo.
  echo # %%i/20
  .\HPC_project_2024 -p gpu -o videos\cuda-cubic.mp4 -R qHD -D 2 -n 3 -L 5 "exp^n"
)

echo.
echo =========================================================== Gamma(z)
for /L %%i in (1,1,20) do (
  echo.
  echo # %%i/20
  .\HPC_project_2024 -p gpu -o videos\cuda-gamma.mp4 -R qHD -D 2 -L 5 --speed 0.1 gamma
)