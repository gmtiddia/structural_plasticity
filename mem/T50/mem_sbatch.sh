#!/bin/bash -x
#SBATCH --account=icei-hbp-2023-0002
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --output=/p/project/icei-hbp-2023-0002/tiddia2/structural_plasticity/mem/logfiles/T50_%j.out
#SBATCH --error=/p/project/icei-hbp-2023-0002/tiddia2/structural_plasticity/mem/logfiles/T50_%j.err
#SBATCH --time=10:00:00
#SBATCH --partition=gpus

# *** start of job script ***
# Note: The current working directory at this point is
# the directory where sbatch was executed.

if [ "$#" -ne 1 ]; then
    seed=0
else
    seed=$1
fi

srun memory_capacity $seed
mv *$seed.dat data$seed/.
