#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
plt.switch_backend('agg')

#act_funs = ['ReQU', 'ReQUr', 'tanh']
act_funs = ['ReQUr', 'ReQU', 'softplus', 'sigmoid', 'tanh']
nets = ['OnsagerNet', 'MLP-ODEN', 'SymODEN']
nActs = len(act_funs)

msg = 'Plot mean and std of MSE for different nets and activations'
parser = argparse.ArgumentParser(description=msg)
parser.add_argument('--errfile', type=str,
                    default='results/paper_Langevin_tc2.txt',
                    help='file path and name that contains error data')
parser.add_argument('--nSeeds', type=int, default=3,
                    help='Number of seeds for each cases')
parser.add_argument('--nDel', type=int, default=0,
                    help='Number of seeds for each cases')
args = parser.parse_args()
print(args)
errfile = args.errfile
nSeeds = args.nSeeds

errors = np.loadtxt(errfile, delimiter=',')
n = errors.shape[0]
nNets = n//nSeeds//nActs

errors = errors.reshape([nNets, nActs, nSeeds])
errors = np.sort(errors)[:, :, :nSeeds-args.nDel]

mse_mean = np.mean(- np.log10(errors), axis=2)
mse_std = np.std(- np.log10(errors), axis=2)
ltxfile = errfile[:-3] + 'ltx'
fmt = " & %.2e " * nActs
with open(ltxfile, 'w') as fh:
    fh.write(r'\begin{tabular}{r|lllll} \hline'+'\n')
    fh.write('ODE nets & '+' & '.join(act_funs) + r'\\ ' + '\n')
    fh.write('\hline' + '\n')
    for i in np.arange(nNets):
        fh.write(nets[i]+' & ' +
                 " & ".join(f'{x:.2f}' for x in mse_mean[i, :]))
        fh.write(r'\\ ' + '\n')
    fh.write('\hline')
    fh.write(r'\end{tabular}')

x = np.arange(len(act_funs))  # the label locations
width = 0.27  		    # the width of the bars

fig = plt.figure(figsize=[6, 4])
ax = fig.add_subplot(111)
rects1 = ax.bar(x - width, mse_mean[0, :], width, label=nets[0])
rects2 = ax.bar(x + 0, mse_mean[1, :], width, label=nets[1])
if nNets > 2:
    rects3 = ax.bar(x + width, mse_mean[2, :], width, label=nets[2])

rects1 = ax.errorbar(
    x - width, mse_mean[0, :], yerr=mse_std[0, :],  fmt="_", color='red')
rects2 = ax.errorbar(x + 0, mse_mean[1, :],
                     yerr=mse_std[1, :],  fmt="_", color='red')
if nNets > 2:
    rects3 = ax.errorbar(
        x + width, mse_mean[2, :], yerr=mse_std[2, :],  fmt="_", color='red')

ax.set_ylabel(r'Accuracy (Higher = Better)')
ax.set_xticks(x)
ylim = np.max(mse_mean.max()) + 1
ax.set_ylim([0, ylim])
ax.set_xticklabels(act_funs)
ax.legend(loc=0, ncol=3)

fig.tight_layout()
barfile = errfile[:-4:]+'.pdf'
plt.savefig(barfile, bbox_inches='tight', dpi=288)
plt.close()
