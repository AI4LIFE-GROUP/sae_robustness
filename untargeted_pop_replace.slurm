#!/bin/bash

#SBATCH -c 40        # Number of cores (-c)
#SBATCH -t 0-05:00     # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p seas_gpu,gpu,gpu_requeue  # Partition to submit to
#SBATCH --mem=80000
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SBATCH -J u_p_s #job_name
#SBATCH -o ./%j.out # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e ./%j.err # File to which STDERR will be written, %j inserts jobid

source activate sparse_coding

for i in {75..95}
do
  echo "Running sample_idx=$((i+1))"
  python main.py --level population --mode replacement --layer_num 20 --data_file art_science.csv --sample_idx $i --batch_size 200 --suffix_len 1 --num_iters 15 --m 300 --k 192 --model_type llama3-8b --log 
done

