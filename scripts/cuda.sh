#!/bin/bash +x
#SBATCH --job-name="CFS (CUDA)"
#SBATCH --time=5:00
#SBATCH --mem=5GB
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --partition=cuda
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s328789@studenti.polito.it
#SBATCH --output=logs/cuda-%j.out

echo $(date +"%D %T")
echo ""

if [[ ! -e "./build/HPC_project_2024" ]]
  echo "Could not find executable!"
  exit
fi

module load ffmpeg/4.3.4
cd ./build

./HPC_project_2024 -d 3 -o sin.raw sin
./HPC_project_2024 -d 3 -o pow.raw -n 4 ^n
