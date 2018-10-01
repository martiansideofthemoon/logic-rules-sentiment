#!/bin/bash

# mode can be default, elmo or logicnn
mode=elmo

for i in {1..100}
do
	python main.py --data_dir data/sst2-sentence/ --config config/${mode}.yml --no-cache --seed $i --job_id ${mode}_seed_${i}
done
