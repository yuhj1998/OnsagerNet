#!/bin/bash
# Train OnsagerNet for RBC problem using PCA data

L=2
nseed=1
epoch=1000
nets="ons ode"
n=7    # nPC

if [ $# -gt 0 ]; then
  epoch=$1;
fi

mkdir -p results

##----------------------------------- ODE for r=28 -------------------------------------------
echo ">>Learning ODE on `hostname` for r=28 L=$L with epoch=$epoch and nseed=$nseed"
tc=1
prefx=results/paper_fig10
rm -f ${prefix}*.log
tpred=99

for net in $nets
do
   for (( seed=0; seed<$nseed; seed++ ))  
   do
       echo "Start working on nPC=$n onet=$net seed=$seed "
       python -O rbc_ode.py -tc $tc $n --onet $net --nL $L -e $epoch --nseeds 1 --seed $seed | tee -a ${prefx}.log
       #python -O test_ode_RBC.py -tc $tc --onet $net --nL $L $n --seed $seed --calc_traj_error --draw_traj | tee -a ${prefx}.log
   done
done
