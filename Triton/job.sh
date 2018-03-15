#!/bin/bash -l
#SBATCH --time=6:00:00
#SBATCH --mem=5G    
#SBATCH --array=0-399



cd $WRKDIR/mono-gaussian-processes/$1

module load anaconda3

source activate $WRKDIR/pystan-env

echo "Parameters $2 $SLURM_ARRAY_TASK_ID"
srun python $WRKDIR/mono-gaussian-processes/err_curve.py $2 $SLURM_ARRAY_TASK_ID $3
source deactivate
module unload anaconda3

