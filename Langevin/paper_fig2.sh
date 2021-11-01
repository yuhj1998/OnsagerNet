#!/bin/bash
# 
# This is the script to generate Figure 2 in the PRF paper
e=600
pat=25
nseed=3
bs=200
lr=0.0256
nL=1
nHons=12
nHode=9
nHsym=17

mkdir -p results

#######################
#  1. prepare figure 2a
#######################
if [ x$1 != x ]
then
  e=$1
  pat=$(($e/40+10))
fi

for tc in 2 
do
    if [ $tc == '1' ]  # linear Hookean
    then
        kid=0
        gid=0
        gamma=3
       logfile=results/paper_Langevin_tc1.log
       errfile=results/paper_Langevin_tc1.txt
    else
        if [ $tc == '2' ]  # Nonlinear pendulum
        then
            kid=1
            gid=1
            gamma=3
           logfile=results/paper_Langevin_tc2.log
           errfile=results/paper_Langevin_tc2.txt
        else       # tc=3: double_well
            kid=2
            gid=0
            gamma=3
           logfile=results/paper_Langevin_tc3.log
           errfile=results/paper_Langevin_tc3.txt
        fi
    fi
    
    rm -f $logfile
    echo "testing case $tc using $e epochs on host `hostname` at `date "+%D %T"`" | tee -a $logfile
    ./check_env.sh | tee -a $logfile
    for net in ons ode sym
    do
        if [ $net == 'ons' ]
        then
    		nH=$nHons
        elif [ $net == 'ode' ]
        then
    		nH=$nHode
        elif [ $net == 'sym' ]
        then
    		nH=$nHsym
        fi
        cmn_opt="-kid $kid -gid $gid -gamma $gamma --onet ${net} -n $nH --nL $nL"
        trn_opt="--seed 0 --nseeds $nseed --epochs $e --patience $pat -bs $bs -lr $lr"
        for fid in 0 2 6 8 9  # ReQUr, ReQU, softplus, sigmoid, tanh
        do
            echo "====Testing kid=$kid gid=$gid gamma=$gamma net=$net fid=$fid ====" | tee -a $logfile
            python test_ode_Langevin.py -f $fid $cmn_opt $trn_opt | tee -a $logfile
        done
    done
    
    fgrep ">>" $logfile | cut -d "=" -f 13 > $errfile
    python plot_errbar_NetF.py --errfile $errfile --nSeeds $nseed
done

#######################
#  2. prepare figure 2b
#######################
fid=0
for seed in 0 1 2
do
   python test_ode_error.py -o ons -f $fid --seed $seed
   python test_ode_error.py -o ode -f $fid --nHnodes 9 --seed $seed
   python test_ode_error.py -o sym -f $fid --nHnodes 17 --seed $seed
   python plot_errbar_Tpred.py -f $fid --seed $seed
   echo "The results is generated to: >> results/Langevin_k1_4_g1_3_f${fid}_s${seed}_err_meanstd.pdf"
done
