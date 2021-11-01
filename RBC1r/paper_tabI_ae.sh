#!/bin/bash
# Train autoencoder and OnsagerNet together for RBC problem
# Note: need long time to run the file once

L=2
nseed=5
epoch=1000
enNet=PCAResNet
#enNet=SAE      # uncommet this to use standard AE
b_cae=0.0
#b_cae=1e-3     # uncomment to use contractive AE with default penalty parameter

b_ae=1.0
a_isom=0.8
b_isom=1.0
e2e=0           # e2e loss coefficient, not included in the paper 
#e2e=100         # uncomment to use e2e loss with b_ae=b_isom=0

tunepar="--a_isom $a_isom --b_isom $b_isom --b_ae $b_ae --b_cae $b_cae --e2e $e2e --enNet $enNet"

if [ $# -gt 0 ]; then
   epoch=$1;
fi

if (( $epoch <= 100 ))  # for testing
then
  nseed=1
  chkenv=False
fi

mkdir -p results/src

##----------------------------------- AE for r=28 -------------------------------------------
logf=results/runae.log
rm -f $logf

echo ">>Learning AE ODE on `hostname` at `date "+%D %T"` for r=28 L=$L with epoch=$epoch and nseed=$nseed" | tee -a $logf
echo "Envirement Setting: OMP_NUM_THREADS=$OMP_NUM_THREADS  MKL_CBWR=$MKL_CBWR" | tee -a $logf
echo "Running command: $0 $*" | tee -a $logf
echo "pwd:`pwd`"  | tee -a $logf
if [ x$chkenv != x'False' ]
 then 
   echo "Checking conda envirement" | tee -a $logf
   ./check_env.sh | tee -a $logf
 fi
echo "------------------------------------------------------------------------------------------------------"
echo " "

tc=1
tpred=99

for n in 3 5 7
do
   for (( seed=0; seed<$nseed; seed++ ))  
   do
       echo "Start working on nPC=$n seed=$seed " | tee -a $logf
       python -O rbc_ae_ode.py -tc $tc $n $tunepar --nL $L -e $epoch --nseeds 1 --seed $seed | tee -a ${logf}
       if (( $epoch >= 50 ))
       then
           python -O test_ode_RBC.py -tc $tc --nL $L $n --seed $seed --method ae --calc_traj_error --draw_structure | tee -a ${logf}
       fi
       echo "done for nPC=$n seed=$seed" | tee -a $logf
   done
done
