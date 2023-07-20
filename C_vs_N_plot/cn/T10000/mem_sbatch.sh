#!/bin/bash -x
#SBATCH --account=icei_ASDSNN
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --output=/g100_work/icei_ASDSNN/lsergi00/structural_plasticity/mem/logfiles/mem_%j.out
#SBATCH --error=/g100_work/icei_ASDSNN/lsergi00/structural_plasticity/mem/logfiles/mem_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=g100_usr_prod

# *** start of job script ***
# Note: The current working directory at this point is
# the directory where sbatch was executed.

if [ "$#" -ne 1 ]; then
    seed=0
else
    seed=$1
fi

srun continous $seed
