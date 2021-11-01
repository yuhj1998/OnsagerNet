#!/bin/bash
if [ x$1 != x ]
then
  e=$1
  pat=$(($e/40+10))
else
  e=600
  pat=25
fi

mkdir -p results
bs=50
seed=0

# 1. Learning by using OnsagerNet
python test_ode_Lorenz.py -r 28 -f 2 --nseeds 1 --seed $seed --epochs $e --patience $pat -bs $bs -lr 0.0128 -v 2
# 2. Show the structure learned by OnsagerNet
python ode_analyzer.py -r 28 -f 2 -n 20 --seed $seed --niter 100
