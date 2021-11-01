#!/bin/bash
e=1000	  	  # epoch
pat=30    	  # patience in Adam
seed=0  	  # initial seed
nseed=3
bs=100
lr=0.0128
nets="ons ode"
fids="0 9"  		# ReQUr, tanh
rs="28"              # change to "16 28" to generate a bar plot(see the end of this file)
nL=3

declare -a nHons=( 0 20 100 100 100 )
declare -a nHode=( 0 16  74  84  88 )

if [ x$1 != x ]
then
  nL=$1
fi
if [ x$2 != x ]
then
  e=$2
  pat=$(($e/40+10))
fi

if (( $e <= 10 ))  # for testing
then
  nseed=1
  fids="0"
  chkenv=False
fi

mkdir -p results

logfile=results/paper_fig8.log
errfile=results/paper_fig8.txt
rm -f $logfile

for net in $nets
  do
    if [ $net == 'ons' ]
    then
      nH=${nHons[$nL]}
    elif [ $net == 'ode' ]
    then
      nH=${nHode[$nL]}
    fi
    
    echo "Testing using $e epochs on host `hostname` at `date "+%D %T"`" | tee -a $logfile
    echo "Running command: $0 $*" | tee -a $logfile
    echo "pwd:`pwd`"  | tee -a $logfile
    if [ x$chkenv != x'False' ]
    then 
      echo "Checking conda envirement"
      ./check_env.sh | tee -a $logfile
    fi
    cmn_opt=" --onet ${net} -n $nH --nL $nL -ig 0.1 "
    trn_opt="--seed $seed --nseeds $nseed --epochs $e --patience $pat -bs $bs -lr $lr"
    for r in $rs
    do
      for fid in $fids
      do
        echo ">>>>>> Testing r=$r net=$net fid=$fid ..." | tee -a $logfile
        python test_ode_Lorenz.py -r $r -f $fid $cmn_opt $trn_opt | tee -a $logfile
      done
    done
  done

fgrep ">>" $logfile | cut -d "=" -f 11 > $errfile
python plot_bar_NetF_Lorenz.py --errfile $errfile --nSeeds $nseed

