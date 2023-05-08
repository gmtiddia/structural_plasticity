for i in $(ls -d */); do cd $i/data0; cat mem_head_0000.dat | awk '{print $3}' | head -6 | tail -5 > ../dum_teor.txt; cd ../..; done
