#!/bin/bash
# 
e=600
pat=25
mkdir -p results

cmn_opt="-gamma 3 --nseeds 1 --epochs $e --patience $pat -bs 200 -lr 0.0256 -p"
python test_ode_Langevin.py -kid 0 -gid 0 -f 0 -ig 0.1 --seed 0 $cmn_opt | tee -a rpot.log
python test_ode_Langevin.py -kid 1 -gid 1 -f 2 -ig 0.1 --seed 2 $cmn_opt | tee -a rpot.log
python test_ode_Langevin.py -kid 2 -gid 0 -f 2 -ig 0.1 --seed 2 $cmn_opt | tee -a rpot.log

