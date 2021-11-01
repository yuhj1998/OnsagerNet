#!/bin/bash
# Train OnsagerNet for RBC problem using PCA data

L=2
nseed=3
epoch=1000
net=ons

if [ $# -gt 0 ]; then
	epoch=$1;
fi

mkdir -p results

##----------------------------------- ODE for r=28 -------------------------------------------
echo ">>Learning ODE on `hostname` for r=28 L=$L with epoch=$epoch and nseed=$nseed"
tc=1
prefx=results/RBC_r28L_T100R100_$net
rm -f ${prefix}*.log
tpred=99

for n in 3 5 7
do
   for (( seed=0; seed<$nseed; seed++ ))  
   do
       echo "Start working on nPC=$n seed=$seed "
       python -O rbc_ode.py -tc $tc $n --onet $net --nL $L -e $epoch --nseeds 1 --seed $seed | tee -a ${prefx}.log
       python -O test_ode_RBC.py -tc $tc --onet $net --nL $L $n --seed $seed --calc_traj_error --draw_traj | tee -a ${prefx}.log
       #python -O test_ode_RBC.py -tc $tc --onet $net --nL $L $n --seed $seed --calc_traj_error --draw_traj --draw_structure | tee ${prefx}_pca${n}_${net}_test.log
       echo "done for nPC=$n seed=$seed";
   done
done
