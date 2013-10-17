if [ "$#" -ne 6 ]; then
 echo "Usage is ./run.sh <job-name> <outputfilename> <binary-name> <graph-input> <sssp-output>"
 exit 1
fi
echo \#!/bin/sh
echo \#SBATCH -J $1          # job name
echo \#SBATCH -o $2      # output and error file name (%j expands to jobID)
echo \#SBATCH -n 1              # total number of mpi tasks requested
echo \#SBATCH -p development     # queue (partition) -- normal, development, etc.
echo \#SBATCH -t 01:30:00        # run time (hh:mm:ss) - 1.5 hours
echo ibrun tacc_affinity $3 -v -f $4 -o $5 -algorithm $6


