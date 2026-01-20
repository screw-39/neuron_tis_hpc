#!/bin/bash
#SBATCH --job-name=waychia            # (-J) Job name
#SBATCH --partition=jhub            # (-p) Slurm partition
#SBATCH --nodes=1                   # (-N) Maximum number of nodes to be allocated
#SBATCH --cpus-per-task=1           # (-c) Number of cores per MPI task
#SBATCH --ntasks-per-node=6        # Maximum number of tasks on each node
#SBATCH --output=job-%j.out         # (-o) Path to the standard output file

ml openmpi/5.0.4

# create databases
uv run create_db.py
# set test parameter with test id
uv run set_test_parameter.py
# run simulation parallelly with test id
mpirun --map-by ppr:1:node ./job_launcher 46657