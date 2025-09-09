#!/bin/bash

#SBATCH -c 40        # Number of cores (-c)
#SBATCH -t 0-12:00     # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p seas_gpu,gpu  # Partition to submit to
#SBATCH --mem=80000
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SBATCH -J u_i #job_name
#SBATCH -o ./%j.out # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e ./%j.err # File to which STDERR will be written, %j inserts jobid

source activate sparse_coding

for i in {130..140}
do
  echo "Running sample_idx=$((i+1))"
  python main.py --level individual --mode suffix --layer_num 20 --data_file ag_news.csv --sample_idx $i --batch_size 100 --suffix_len 1 --num_iters 10 --m 300 --k 192 --model_type llama3-8b --log
done

for i in {230..240}
do
  echo "Running sample_idx=$((i+1))"
  python main.py --level individual --mode replacement --layer_num 20 --data_file ag_news.csv --sample_idx $i --batch_size 100 --suffix_len 1 --num_iters 10 --m 300 --k 192 --model_type llama3-8b --log
done