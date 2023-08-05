for i in {1..20}; do
    j=$(($i*5000))
    cd $j
    sed -i /#/d params.dat
    cd ..
done

#cd 500
#sed -i /#/d params.dat
#cd ../1000
#sed -i /#/d params.dat
#cd ..
