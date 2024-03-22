t=(0 1 5 10 25 50 100 200 500 1000 2000 5000 10000)
for j in ${t[@]}; do
    cd $j
    . run.sh 0 9    
    cd ..    
done
