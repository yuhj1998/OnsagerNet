#!/bin/bash
# test nearest neighbor for RBC problem using PCA data (used in Table II in reference paper)

seed=0
epoch=1000
net=ons

if [ $# -gt 0 ]; then
   epoch=$1;
fi

mkdir -p results

##----------------------------------- ODE for r=28 -------------------------------------------
tc=1
L=2
log=results/RBC_r28L_T100R100_nn_${net}.log
rm -f $log

for n in 3 5 7
do
     python -O rbc_ode.py -tc $tc $n --onet $net --nL $L -e $epoch --nseeds 1 --seed $seed | tee -a ${log}
     python -O test_ode_nn.py -tc $tc --onet ons --nL $L $n --seed 0 --calc_traj_error | tee -a $log
done
