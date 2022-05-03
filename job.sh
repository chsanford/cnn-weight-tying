#!/bin/sh
#
#SBATCH -A astro # The account name for the job.
#SBATCH --job-name=Rnorm   # The job name.
#SBATCH --error=gpu.err    # error file
#SBATCH --nodes=1          # 1 GPU node
#SBATCH --cpus-per-task=4  # 4 CPU core to drive GPU
#SBATCH --time=10:00:00    # The time the job will take to run.
#SBATCH --mem-per-cpu=16gb # The memory the job will use per cpu core.
#SBATCH --gres=gpu:1
#SBATCH -o log.log


module load anaconda
module load cuda11.1/toolkit
source activate nn
unset XDG_RUNTIME_DIR

CNNlr=${1:0.1}
LClr=${2:0.01}
epochs=${3:5}
nsplit=${4:2}

echo Starting CNNLR ${CNNlr} LClr ${LClr} host:$(hostname) date:$(date)

python experiments.py --epochs ${epochs} --n-splits ${nsplit} --cnn-lr ${CNNlr} --lc-lr ${LClr}
