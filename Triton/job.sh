#!/bin/bash -l
#SBATCH -c 2
#SBATCH --time=4:00:00
#SBATCH --mem=5G    
#SBATCH --array=0-399

cd $WRKDIR/mono-gaussian-processes/$1

module load anaconda3

source activate $WRKDIR/pystan-env

echo "Parameters $2 $SLURM_ARRAY_TASK_ID"
srun -c 2 python $WRKDIR/mono-gaussian-processes/err_curve.py $2 $SLURM_ARRAY_TASK_ID $3
source deactivate
module unload anaconda3

