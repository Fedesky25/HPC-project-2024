#!/bin/bash +x
#SBATCH --job-name="CFS (Serial)"
#SBATCH --time=10:00
#SBATCH --mem=5GB
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=global
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s328789@studenti.polito.it
#SBATCH --output=logs/serial-%j.out

echo $(date +"%D %T")
echo ""

if [[ ! -e "./build/HPC_project_2024" ]]
  echo "Could not find executable!"
  exit
fi

module load ffmpeg/4.3.4
cd ./build

echo ""
echo "=========================================================== Fibonacci(z)"
for i in {1..20} do
  echo "# ${i}/20"
  ./HPC_project_2024 -p none -o videos/serial-fib.mp4 -R qHD -D 2 -L 5 fib
done

echo ""
echo "=========================================================== exp(z^3)"
for i in {1..20} do
  echo "# ${i}/20"
  ./HPC_project_2024 -p none -o videos/serial-cubic.mp4 -R qHD -D 2 -L 5 -n 3 exp^n
done

echo ""
echo "=========================================================== Gamma(z)"
for i in {1..20} do
  echo "# ${i}/20"
  ./HPC_project_2024 -p none -o videos/serial-gamma.mp4 -R qHD -D 2 -L 5 -v 0.1 gamma
done
