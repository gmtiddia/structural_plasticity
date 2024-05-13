for i in {19..20}; do
    j=$(($i*5000))
    cd $j
    . run.sh 0 9    
    cd ..    
done
