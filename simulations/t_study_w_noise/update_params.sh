t=(0 1 5 10 25 50 100 200 500 1000 2000 5000 10000)
for j in ${t[@]}; do
    sed "s/__RECOMB__/$j/" params.dat > $j/params.dat
    cd $j
    #    for i in {0..9}; do
    #cp -f ../status_dum.dat status_000$i.dat
    #done
    cd ..
done
