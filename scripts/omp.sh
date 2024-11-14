#!/bin/bash +x
#SBATCH --job-name="CFS (OpenMP)"
#SBATCH --time=30:00
#SBATCH --mem=5GB
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --partition=global
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s328789@studenti.polito.it
#SBATCH --output=logs/omp-%j.out

echo $(date +"%D %T")
echo ""

if [[ ! -e "./build/HPC_project_2024" ]]
  echo "Could not find executable!"
  exit
fi

# module load ffmpeg/4.3.4
cd ./build

num_threads=(4,8,16,32,64)
for num in "${num_threads[@]}"; do
  export OMP_NUM_THREADS $num
  echo "================================================================================ threads = $num"
  for i in $(seq 1 100); do
    echo ">> iteration $i/100"
    ./HPC_project_2024 -p omp -R FHD -D 10 -f 60 -s 4u/h -d 12 poly1
  done
done
