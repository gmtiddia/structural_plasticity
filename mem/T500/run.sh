#!/bin/bash

for i in {0..9}; do
    mkdir data$i
    sbatch mem_sbatch.sh ${i}
done
