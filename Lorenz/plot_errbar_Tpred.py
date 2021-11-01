#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import argparse
plt.switch_backend('agg')

parser = argparse.ArgumentParser(
    description='Test learned ODEs for Langevin system')
parser.add_argument('-r', type=float, default=16,
                    help='scaled Rayleigh number')
parser.add_argument('-s', '--sigma', type=float, default=10,
                    help='Prandtl number (default: 10)')
parser.add_argument('-f', '--fid', type=int, default=0,
                    metavar='FID',
                    help='the id of activation function')
parser.add_argument('--seed', type=int, default=0, metavar='SEED',
                    help='The first SEED to test the performance')
args = parser.parse_args()
print(args)

r = args.r
fid = args.fid
seed = args.seed

nets = ['ons', 'ode']
labels = ['OnsagerNet', 'MLP-ODEN']
fmts = ['-s', '-o', '--s', '--o']

fig = plt.figure(figsize=[6, 4])
ax = fig.add_subplot(111)
eps=1e-8

for i in np.arange(len(nets)):
    errfile = f'results/Lorenz_r{int(r)}_{nets[i]}_f{args.fid}_s{seed}_err_meanstd.txt'
    errs = np.loadtxt(errfile, delimiter=',')
    ax.errorbar(errs[1:, 0], np.log(errs[1:, 1]+eps), yerr=np.log(1+errs[1:, 2]/(errs[1:,1]+eps)),
                uplims=False, lolims=True, fmt=fmts[i], label=labels[i])

ax.set_ylabel(r'average log of relative $L^2$ error')
ax.set_xlabel('t')
ax.legend(loc=0)

fig.tight_layout()
barfile = (f'results/Lorenz_r{int(r)}_f{args.fid}_s{seed}_err_meanstd.pdf')
plt.savefig(barfile, bbox_inches='tight', dpi=288)
plt.close()
