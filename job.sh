#!/bin/bash -l
#SBATCH --time=0-10:00:00    
#SBATCH --mem=4G    

cd $WRKDIR/mono-gaussian-processes

module load anaconda3

source activate ../pystan-env

srun python GP.py

source deactivate
module unload anaconda3

