#!/bin/bash

#SBATCH -c 40        # Number of cores (-c)
#SBATCH -t 1-10:00     # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p seas_gpu,gpu  # Partition to submit to
#SBATCH --mem=100000
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SBATCH -J targeted_population_suffix #job_name
#SBATCH -o ./%j.out # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e ./%j.err # File to which STDERR will be written, %j inserts jobid

source activate sparse_coding

for i in {0..99}
do
  echo "Running sample_idx=$((i+1))"
  python main.py --targeted --level population --mode suffix --layer_num 20 --data_file two_class_generated.csv --sample_idx $i --batch_size 800 --suffix_len 3 --num_iters 50 --m 400 --k 192 --model_type llama3-8b --log 
done