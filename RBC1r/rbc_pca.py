#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Do PCA analysis on given RBC data
# @author: Haijun Yu <hyu@lsec.cc.ac.cn>
#  We first do a PCA output the primary compoents (PCs) and then
#  1. scatter plot the first 3 PCs;
#  2. plot figures for choosing representative trajectives
#

import rbctools as rbc
import autoencoders as ae
import config as cfgs
import torch
import numpy as np
import argparse
from torch import manual_seed
from numpy import random
random.seed(0)
manual_seed(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Do PCA analysis on Rayleigh-Bernard convection data')
    parser.add_argument('-tc', type=int, default=cfgs.DEFAULT_CASE_ID,
                        metavar='tc',
                        help='input the if of the test case')
    parser.add_argument('nPC', type=int, nargs='?', default=-1, metavar='nPC',
                        help='input number of PCs')
    parser.add_argument('-ts', type=int, default=-1, metavar='ts',
                        help='number of snapshots to trim from start')
    parser.add_argument('-te', type=int, default=-1, metavar='te',
                        help='number of snapshots to trim from end')
    parser.add_argument('--plot_traj', action='store_true',
                        default=False,
                        help='draw trajectory and phase portrait of first 3 PCA')
    parser.add_argument('--plot_traj_ref', action='store_true',
                        default=False,
                        help='draw phase portrait of first 3 PCA of reference traj')
    parser.add_argument('--draw_flow', action='store_true',
                        default=False,
                        help='draw flow field before and after PCA reduction')
    args = parser.parse_args()
    print(args)

test_id = args.tc
cfg = cfgs.get_test_case(test_id)

ts = args.ts if args.ts >= 0 else cfg.ts
te = args.te if args.te >= 0 else cfg.te
nPC = args.nPC if args.nPC > 0 else cfg.nPC

nTraj = cfg.nTraj
h5fname = cfg.h5fname
outloc = cfg.outloc
np.random.seed(seed=cfg.iseed)
torch.manual_seed(cfg.iseed)

# %% Load data, do PCA, and show pca variance, visualize PCs
print(f'Loading dataset {h5fname}.h5 ...', flush=True)
uvh, coordx, coordy, nx, ny, nf = rbc.load_uvh_data(h5fname+'.h5')
print(f'Downsampling ts={ts}, te={te}, nTraj={nTraj},  ratio=2')
te = (nf//nTraj - te*2)
uvh, coordx, coordy, nx, ny = rbc.downsampling(uvh, coordx, coordy,
                                               ratio=2, ts=ts*2, te=te,
                                               nRun=nTraj)
print(f'PCA for dataset {h5fname}.h5 ...', flush=True)
nf = uvh.shape[0]
nVar = uvh.shape[1]
mu = np.mean(uvh, axis=0) * 0

wt = 1/np.sqrt(nVar/3.0) * cfg.wt_factor
print(f'pca_wt={wt}')
ds = uvh * wt
model = ae.PCA_Encoder(nPC, ds, trainable=False)

var = model.pca.explained_variance_
var_ratio = model.pca.explained_variance_ratio_
var_cumsum = model.pca.explained_variance_ratio_.cumsum()
print(f' var={var}\n var_ratio={var_ratio}\n var_res={1-var_cumsum}')
var_file = h5fname+f'_pca{nPC}_var.txt'
var_out = np.vstack([var, var_ratio, 1-var_cumsum]).T
np.savetxt(var_file, var_out, delimiter=', ', fmt='%.8e')
if nPC > 8:
    rbc.plot_pca_var(var_ratio, var_cumsum, h5fname+f'_pca{nPC}_var')

PCs = model.encode(ds)
PCs = PCs.reshape([nf, nPC])
PCs = PCs.detach().numpy()
enc_file = h5fname+f'_pca{nPC}_enc_data.txt.gz'
np.savetxt(enc_file, PCs, delimiter=', ', fmt='%.8e')
print('  PCA encode data saved to ', enc_file)
dict_file = h5fname+f'_pca{nPC}_model_dict.pth'
torch.save(model.state_dict(), dict_file)
print('  PCA model dict saved to ', dict_file)


nOut = nf//nTraj

if args.plot_traj:
    nPlot = 20
    for i in np.arange((nTraj//nPlot)):
        iP = i*nOut*nPlot + np.arange(nOut//2*nPlot)*2
        for j in np.arange(1):
            jj = 2*j
            rbc.plot_nTraj_t2d(PCs[iP, jj:], outloc +
                               f'_plot{i}_p{jj+1}p{jj+2}', nPlot)
            rbc.plot_nTraj2d_phase(PCs[iP, jj:], outloc +
                                   f'_plot{i}_p{jj+1}p{jj+2}', nPlot)
    print(f'Choose traj from {outloc}_3pc_plot.pdf for better fig')
    nP = min(nTraj, 40) * nOut
    rbc.plot_nTraj3d_scatter(
        PCs[:nP:2, :], outloc+'_3pc_scatter', nP//nOut, ts=0)

if args.plot_traj_ref:
    tPC = PCs.reshape([nTraj, nOut, -1])
    tPC = tPC[cfg.Paths, ::2, :].reshape([-1, nPC])
    rbc.plot_nTraj_t2d(tPC[:, :], outloc+'_3pc_plot_ref', len(cfg.Paths))
    rbc.plot_nTraj2d_phase(tPC[:, :], outloc+'_3pc_plot_ref', len(cfg.Paths))
    rbc.plot_nTraj3d_scatter(
        tPC[:, :], outloc+'_3pc_scatter_ref', len(cfg.Paths), ts=0)

if args.draw_flow:
    for iPC in range(min(6, nPC)):
        pcFlow = model.pca.components_[iPC, :]
        stitle = outloc+f'_Flow_pca{nPC}_PC{iPC+1}'
        rbc.plotflow(coordx, coordy, pcFlow, stitle)

# %% testing the PCA accuracy
nPathLen = ds.shape[0]//nTraj
for i in np.arange(nPathLen):
    orgflow = ds[i, :]
    enc = model.encode(orgflow)
    encflow = model.decode(enc).detach().cpu().numpy()

    orgflow = orgflow.flatten()
    err = np.sqrt(np.sum((encflow-orgflow)**2))
    flow_norm = np.sqrt(np.sum((orgflow)**2))
    rel_err = err / flow_norm
    if i < 10 or i >= nPathLen-10:
        print('Testing PCA', nPC, ' on snapshot ', i,
              f': AbsErr={err:.3e}, RelErr={rel_err:.3e}')
    if args.draw_flow and (i == 0 or i == nPathLen-1):
        rbc.plotflow(coordx, coordy, orgflow/wt, outloc+f'_Flow{i}_Original')
        rbc.plotflow(coordx, coordy, encflow/wt, outloc+f'_Flow{i}_pca{nPC}')

if args.draw_flow:
    aveflow = model.decode(enc*0).detach().cpu().numpy()
    rbc.plotflow(coordx, coordy, aveflow/wt, outloc+'_Flow_Average')
