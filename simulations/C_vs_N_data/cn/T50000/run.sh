#!/bin/bash

for i in {0..0}; do
    sbatch mem_sbatch.sh ${i}
done
