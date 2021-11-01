#!/bin/bash
# Generate PCA data and figure 9(a)
# 

mkdir -p results

tc=1
prefx=results/RBC_r28L_T100R100
rm -f ${prefix}*.log

for n in 3 5 7 32
do 
    python rbc_pca.py -tc 1 $n | tee ${prefx}_pca${n}.log
done

rm -f rm ${prefx}_pca[357]_var.txt
