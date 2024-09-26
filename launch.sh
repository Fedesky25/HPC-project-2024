#!/bin/bash +x
#SBATCH --job-name="Function Streamplotter"
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --mem=5GB
#SBATCH --gres=gpu:1
#SBATCH --partition=cuda
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s328789@studenti.polito.it

echo $(date +"%D %T")
echo ""

echo "Loading modules..."
module load nvidia/cudasdk/11.8
module load cmake/3.14.3
module load ffmpeg/4.3.4

echo "Handle directories..."
mkdir -p build
if [ -d "./source" ]; then rm -r ./source; fi

git clone https://github.com/Fedesky25/HPC-project-2024.git source
cmake -DCMAKE_BUILD_TYPE=Release ./source
cmake --build ./build --target HPC_project_2024 -j 10

echo -e "\n============================================================== Start executions\n"

./build/HPC_project_2024 -d 40 frac