for i in {1..20}; do
    j=$(($i*5000))
    cp -r template $j
    cd $j
    sed "s/__PATTERNS__/$j/" params.templ > params.dat
    
    net=$(printf "network_000*_%04d.dat" $(($i-1)))
    ln -s /p/scratch/icei-hbp-2023-0002/100K/$net .

    . make_structural_plasticity.sh
    
    cd ..    
done
