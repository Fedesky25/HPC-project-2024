echo ""
echo "======================================================================================================= threads = 1"
for p in 4341 5908 7797 10880 14179 16464 19383 23200 28330 35464 40128 45810; do
  echo ""
  echo "============== p = $p"
  for i in {1..20}; do
    echo ""
    echo "# $i/20"
    ./build/HPC_project_2024 -p none -o videos\serial.mp4 -R FHD -D 2 -L ./particles/p$p.txt fib
  done
done


for n in 4 8 16 32 64 112; do
  export OMP_NUM_THREADS=$n
  echo ""
  echo "======================================================================================================= threads = 1"
  for p in 4341 5908 7797 10880 14179 16464 19383 23200 28330 35464 40128 45810; do
    echo ""
    echo "============== p = $p"
    for i in {1..20}; do
      echo ""
      echo "# $i/20"
      ./build/HPC_project_2024 -p omp -o videos\omp$n.mp4 -R FHD -D 2 -L ./particles/p$p.txt fib
    done
  done
done


echo ""
echo "======================================================================================================= threads = 1"
for p in 4341 5908 7797 10880 14179 16464 19383 23200 28330 35464 40128 45810; do
  echo ""
  echo "============== p = $p"
  for i in {1..20}; do
    echo ""
    echo "# $i/20"
    ./build/HPC_project_2024 -p gpu -o videos\cuda.mp4 -R FHD -D 2 -L ./particles/p$p.txt fib
  done
done