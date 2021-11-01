#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import argparse
plt.switch_backend('agg')

nlabels = 4
nNet = 2
labels = ['ReQUr(r=16)', 'tanh(r=16)', 'ReQUr(r=28)', 'tanh(r=28)' ]

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
nseeds = args.nSeeds

errors = np.loadtxt(errfile, delimiter=',')
n = errors.shape[0]
n0 = n//nNet
errors = errors.reshape([nNet, -1])
ons_errs = errors[0, :]
ode_errs = errors[1, :]

ONSNet = -np.log10(np.array(ons_errs).reshape((nlabels,nseeds))).mean(axis=1) + 3.5
MLPODE = -np.log10(np.array(ode_errs).reshape((nlabels,nseeds))).mean(axis=1) + 3.5

ONSstd = -np.log10(np.array(ons_errs).reshape((nlabels,nseeds))).std(axis=1)
MLPstd = -np.log10(np.array(ode_errs).reshape((nlabels,nseeds))).std(axis=1)

x = np.arange(nlabels)  # the label locations
width = 0.27  # the width of the bars

fig = plt.figure(figsize=[6, 4])
ax = fig.add_subplot(111)
rects1 = ax.bar(x - width, ONSNet, width, label='OnsagerNet')
rects2 = ax.bar(x + 0, MLPODE, width, label='MLP-ODEN')

ax.errorbar(x-width, ONSNet, yerr=ONSstd, fmt="_", color='red')
ax.errorbar(x + 0,  MLPODE, yerr=MLPstd, fmt="_", color='red')

ax.set_ylabel(r'Accuracy (Higher = Better)')
ax.set_xticks(x)
ymax = np.max([MLPODE.max(), ONSNet.max()]) + 1.5
ax.set_ylim([0, ymax])
ax.set_xticklabels(labels)
ax.legend(loc='upper left', ncol=2)

fig.tight_layout()
barfile=errfile[:-4:]+'.pdf'
plt.savefig(barfile, bbox_inches='tight', dpi=288)
