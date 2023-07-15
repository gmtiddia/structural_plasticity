t=(0 1 5 10 25 50 100 200 500 1000 2000 5000 10000)
for j in ${t[@]}; do
    cp -r template $j
    cd $j
    sed "s/__RECOMB__/$j/" params.templ > params.dat
    
    #net=$(printf "network_000*_%04d.dat" $(($i-1)))
    #ln -s /p/scratch/icei-hbp-2023-0002/pi_step100_lognorm_C5000_T100000_noise1.0/$net .

    . make_structural_plasticity.sh
    
    cd ..    
done
