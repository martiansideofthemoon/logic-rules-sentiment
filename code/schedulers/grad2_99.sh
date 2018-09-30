#!/bin/bash
#SBATCH --time=360  # wall-clock time limit in minutes
#SBATCH --gres=gpu:3,gpu_mem:500M  # number of GPUs and memory limit
#SBATCH --output=slurm-logs/grad2_99_0.sh  # output file

l1=99
cp config/nonstatic.yml config/grad2_$l1.yml
sed -i -e 's/l1_val: 0.7/l1_val: 0.99/g' config/grad2_$l1.yml
sed -i -e 's/gradient_type: sum_abs_prob/gradient_type: sum_sq_log/g' config/grad2_$l1.yml

for i in {39..100}
do
	python main.py --data_dir data/sst2/ --config config/grad2_$l1.yml --no-cache --seed $i --job_id grad2_${l1}_seed_$i 2>&1 | tee logs/grad2_${l1}_seed_$i.log
done
