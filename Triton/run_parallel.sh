#!/bin/bash -l


# input: 
#		folder of output
# 		number of data points to evaluate upto

RANDOM=$$
if [ ! -d "$1" ]
then
    echo "File doesn't exist. Creating now"
    mkdir ./$1
    echo "File created"
else
    echo "File exists"
fi

seed=$RANDOM
for ((i=1;i<=$2;i++));
do 
	sbatch ./job.sh $1 $i $seed
done 


