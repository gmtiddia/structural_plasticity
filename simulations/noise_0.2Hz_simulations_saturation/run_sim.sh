for i in {11..19}; do
    j=$(($i*5000))
    cd $j
    . run.sh 0 9    
    cd ..    
done
