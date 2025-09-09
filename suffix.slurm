#!/bin/bash

#SBATCH -c 40        # Number of cores (-c)
#SBATCH -t 0-01:00     # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p seas_gpu,gpu  # Partition to submit to
#SBATCH --mem=80000
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SBATCH -J npd #job_name
#SBATCH -o ./%j.out # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e ./%j.err # File to which STDERR will be written, %j inserts jobid

source activate sparse_coding

python neuronpedia_example_2.py