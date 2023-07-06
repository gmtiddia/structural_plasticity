#!/bin/bash
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 seed0 seed1 (seed0: first seed offset, seed1: last seed offset)"
    echo "e.g.:"
    echo "$0 0 9"
    echo "for 10 simulations with different seeds for random number generation."
else
    cat run_sbatch.templ | sed "s;__PATH__;$(pwd);" > run_sbatch.sh
    for i in $(seq $1 $2); do
        sbatch run_sbatch.sh ${i}
    done
fi
