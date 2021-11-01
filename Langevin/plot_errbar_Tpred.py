#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import argparse
plt.switch_backend('agg')

parser = argparse.ArgumentParser(
    description='Test learned ODEs for Langevin system')
parser.add_argument('-gamma', type=float, default=3.0, metavar='g',
                    help='input diffusion constant gamma (default: 1.0)')
parser.add_argument('-kappa', type=float, default=4, metavar='k',
                    help='input elastic constant kappa (default: 4.0)')
parser.add_argument('-gid', type=int, default=1, metavar='gid',
                    help='input diffusion type (0=const, 1=nonlinear)')
parser.add_argument('-kid', type=int, default=1, metavar='kid',
                    help='input potential type (0=Hookean, 1=Pendulum, 2=DoubleWell)')
parser.add_argument('-f', '--fid', type=int, default=0,
                    metavar='FID',
                    help='the id of activation function')
parser.add_argument('--seed', type=int, default=0, metavar='SEED',
                    help='The first SEED to test the performance')
args = parser.parse_args()
print(args)

kid = args.kid
gamma = args.gamma
kappa = args.kappa
gid = args.gid
kid = args.kid
fid = args.fid
seed = args.seed

nets = ['ons', 'ode', 'sym']
labels = ['OnsagerNet', 'MLP-ODEN', 'SymODEN']
fmts = ['-s', '-o', '-_']

fig = plt.figure(figsize=[6, 4])
ax = fig.add_subplot(111)

for i in np.arange(len(nets)):
    errfile = (f'results/Langevin_k{kid}_{int(kappa)}_g{gid}_{int(gamma)}'
               + f'-{nets[i]}_f{args.fid}_s{seed}_err_meanstd.txt')
    errs = np.loadtxt(errfile, delimiter=',')

    ax.errorbar(errs[:, 0], errs[:, 1], yerr=errs[:, 2], fmt=fmts[i], label=labels[i])

ax.set_ylabel(r'average relative $L^2$ error')
ax.set_xlabel('t')
ax.legend(loc='upper left')

fig.tight_layout()
barfile = (f'results/Langevin_k{kid}_{int(kappa)}_g{gid}_{int(gamma)}'
           + f'_f{args.fid}_s{seed}_err_meanstd.pdf')
plt.savefig(barfile, bbox_inches='tight', dpi=288)
plt.close()
