#!/bin/bash

bs=200
lr=0.0128
nseed=3

if [ x$1 != x ]
then
  e=$1
  pat=$(($e/30+5))
else
  e=600
  pat=25
  echo "running with epoch=$e, pat=$pat, may take more than 2 hours"
fi

mkdir -p results

## 1. Generate Fig 7.  the accuracy comparison
logfile=results/paper_fig7.log
errfile=results/paper_fig7.txt
rm -f $logfile
echo "testing using $e epochs on host `hostname` at `date "+%D %T"`" | tee -a $logfile
./check_env.sh | tee -a $logfile
for net in ons ode
 do
     if [ $net == 'ons' ]
     then
 	nH=20
     elif [ $net == 'ode' ]
     then
 	nH=16
     fi
     cmn_opts="-n $nH --nseeds $nseed --seed 0 --epochs $e --patience $pat -bs $bs -lr $lr "
     for r in 16 28
     do
         for fid in 0 2
         do
            echo ">>>>>> Testing r=$r net=$net fid=$fid ..."
            python test_ode_Lorenz.py -r $r -f $fid --onet ${net} $cmn_opts | tee -a $logfile
         done
     done
 done

fgrep ">>" $logfile | cut -d "=" -f 11 > $errfile
python plot_bar_NetF_Lorenz.py --errfile $errfile --nSeeds $nseed
