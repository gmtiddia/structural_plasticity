t=(0 1 5 10 25 50 100 200 500 1000 2000 5000 10000)
for j in ${t[@]}; do
    cd $j
    rm -rf network_000*.dat
    for i in {0..9}; do
	net=$(printf "network_000*_%04d.dat" $(($i-1)))
	ln -s /p/scratch/icei-hbp-2023-0002/t_study_w_noise/$j/$net .
    done
    cd ..    
done
