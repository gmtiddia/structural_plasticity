t=(0 1 5 10 25 50 100 200 500 1000 2000 5000 10000)
for j in ${t[@]}; do
    rm -rf $j
    cp -r template $j
    cd $j
    sed "s/__RECOMB__/$j/" params.templ > params.dat
    
    . make_structural_plasticity.sh
    
    cd ..    
done
