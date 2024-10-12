#!/bin/bash +x
#SBATCH --job-name="Build CFS"
#SBATCH --time=10:00
#SBATCH --mem=5GB
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=global
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s328789@studenti.polito.it
#SBATCH --output=logs/%j.out

echo $(date +"%D %T")
echo ""

source_dir="./source"
if [[ -e "./CMakeLists.txt" ]]; then
  source_dir=$(pwd)
  cd ..
elif [[ ! -e "./source/CMakeLists.txt" ]]; then
  echo "Expected source directory with CMakeLists.txt file inside"
  exit 1
fi

mkdir -p build

echo "Loading modules..."
module load nvidia/cudasdk/10.1
module load cmake/3.14.3

# Tesla K40: Kepler micro-architecture, compute capability 3.5  ->  sm_35
export GPU_ARCHITECTURE=sm_35

echo "Compiling with $SLURM_CPUS_PER_TASK threads..."
cmake -DCMAKE_BUILD_TYPE=Release -S $source_dir -B ./build
if [[ $SLURM_CPUS_PER_TASK -gt 1 ]]; then
  cmake --build ./build --target HPC_project_2024 -j $SLURM_CPUS_PER_TASK
else
  cmake --build ./build --target HPC_project_2024
fi