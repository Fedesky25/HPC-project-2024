#!/bin/bash +x
#SBATCH --job-name="Function Streamplotter"
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --mem=5GB
#SBATCH --gres=gpu:1
#SBATCH --partition=cuda
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
module load nvidia/cudasdk/11.8
module load cmake/3.14.3
module load ffmpeg/4.3.4

echo "Compiling..."
cmake -DCMAKE_BUILD_TYPE=Release $source_dir
cmake --build ./build --target HPC_project_2024 -j 10

echo -e "\n============================================================== Start executions\n"

./build/HPC_project_2024 -d 40 frac