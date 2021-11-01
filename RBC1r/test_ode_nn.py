#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
@File 	 : test_ode_nn.py
@Time 	 : 2021/04/22
@Author  : Haijn Yu <hyu@lsec.cc.ac.cn>
@Desc	 : test the accuracy of neareast neighbor method for RBC system
'''

# %% 1. import libs
import numpy as np
import matplotlib.pyplot as plt
import rbctools as rbc
from mpl_toolkits.axes_grid1 import make_axes_locatable
import torch
import ode_net as ode
from scipy.special import binom
import config as cfgs
import argparse
import ode_analyzer as oa


def check_traj_error(hvar, ONet, dt, ns, nTraj, base_file=None):
    nf = hvar.shape[0]
    nPC = ONet.nVar
    nOut = nf // (2*nTraj)
    htest = hvar[:, :].reshape([nTraj, 2*nOut, nPC])
    htest = htest[:, 2*ns:, :]
    nOut = nOut - ns
    for T in (1, 10, 20, nOut-1):
        print(f'Evaluating the ODE solution error for T={T} ...')
        print('Start calculating ODE trajectories ... ', flush=True)
        nnR = nOut - int(T)
        test_err = np.zeros((nTraj, nnR))
        it_shift = 1
        if T > 2*dt:
            it_shift = T*2
        it = np.arange(nnR)
        h0 = htest[:, 2*it, :].reshape([-1, nPC])
        hf = htest[:, 2*it+it_shift, :].reshape([-1, nPC])
        with torch.no_grad():
            h0 = torch.tensor(h0, dtype=torch.float)
            if T > 5*dt:
                hf_ode = ONet.ode_rk3(h0, dt*5, int(T/dt/5))
            else:
                hf_ode = ONet.ode_rk3(h0, dt, int(T/dt))
            hf_ode = hf_ode.detach().cpu().numpy()
        terr = np.sqrt(np.sum((hf_ode-hf)**2, axis=1))
        test_err = terr.reshape([nTraj, -1])

        L2Amp = np.max(np.sqrt(np.sum(htest**2, axis=2)), axis=1)
        LinfNorm = np.max(np.sqrt(np.sum(htest**2, axis=2)))
        L2Norm = np.sqrt(np.mean(np.sum(htest**2, axis=2), axis=1))
        L2err_ave = np.mean(np.sqrt(np.mean(test_err**2, axis=1)) / L2Norm)
        for ir in range(nTraj):
            test_err[ir, :] /= L2Amp[ir]
        L2err_ave2 = np.mean(test_err)
        print(f'The Linf amplitude of the trajectories is {LinfNorm:.3e}')
        print(f'>>L2_rel prediction error ' +
              f'r{cfg.rRa} {method}{nPC}(t={T})={L2err_ave:.3e}, {L2err_ave2:.3e}',
              flush=True)
        np.savetxt(outloc+f'_{method}{nPC}_{onet}_testerr_T{T}.txt',
                   test_err, delimiter=', ', fmt='%15.6e')
        if T < nOut-1:
            fig = plt.figure(figsize=[9, 4])
            ax1 = fig.add_subplot(111)
            im = plt.imshow(test_err, cmap='viridis')
            ax1.set_xlabel('Time snapshots')
            ax1.set_ylabel('Path index')
            divider = make_axes_locatable(ax1)
            cax = divider.append_axes("right", size="5%", pad=0.08)
            plt.colorbar(im, cax=cax)
        elif T == nOut-1:
            fig = plt.figure(figsize=[8, 2])
            ax1 = fig.add_subplot(111)
            plt.plot(test_err, '+-')
            print(f'>>N_fail r{cfg.rRa} {method}{nPC} ={np.sum(test_err>0.3)}')
        plt.tight_layout()
        if base_file is not None:
            figName = base_file + f'_testerr_T{T}.pdf'
            plt.savefig(figName, bbox_inches='tight', dpi=200)
        else:
            plt.draw()


def check_traj_error_nn(dtest, dtrain, dt, ns, nTraj, base_file=None):
    nf = dtest.shape[0]
    nPC = dtest.shape[1]
    nOut = nf // nTraj
    htest = dtest.reshape([nTraj, nOut, nPC])[:, ns:, :]
    htrain = dtrain.reshape([-1, nOut, nPC])[:, ns:, :]
    nOut = nOut - ns
    for T in (1, 10, 20, nOut-1):
        print(f'Evaluating the ODE solution error for T={T} ...')
        print('Start calculating ODE trajectories ... ')
        nnR = nOut - T
        #test_err = np.zeros((nTraj, nnR))
        hf = htest[:, T:, :]
        hf_nn = np.zeros_like(hf)
        for j in np.arange(nnR):
            for i in np.arange(nTraj):
                herr = htest[i, j, :]-htrain[:, j, :]
                mi = np.argmin(np.linalg.norm(herr.squeeze(), axis=1))
                hf_nn[i, j, :] = htrain[mi, T+j, :]
        test_err = np.sqrt(np.sum((hf_nn-hf)**2, axis=2))
        
        L2Amp = np.max(np.sqrt(np.sum(htest**2, axis=2)), axis=1)
        LinfNorm = np.max(np.sqrt(np.sum(htest**2, axis=2)))
        L2Norm = np.sqrt(np.mean(np.sum(htest**2, axis=2), axis=1))
        L2err_ave = np.mean(np.sqrt(np.mean(test_err**2, axis=1)) / L2Norm)
        for ir in range(nTraj):
            test_err[ir, :] /= L2Amp[ir]
        L2err_ave2 = np.mean(test_err)
        print(f'The Linf amplitude of the trajectories is {LinfNorm:.3e}')
        print(f'>>L2_rel 1nn prediction error ' +
              f'r{cfg.rRa} {method}{nPC}(t={T})={L2err_ave:.3e}, {L2err_ave2:.3e}',
              flush=True)
        np.savetxt(outloc+f'_{method}{nPC}_{onet}_testerr_T{T}.txt',
                   test_err, delimiter=', ', fmt='%15.6e')
        if T < nOut-1:
            fig = plt.figure(figsize=[9, 4])
            ax1 = fig.add_subplot(111)
            im = plt.imshow(test_err, cmap='viridis')
            ax1.set_xlabel('Time snapshots')
            ax1.set_ylabel('Path index')
            divider = make_axes_locatable(ax1)
            cax = divider.append_axes("right", size="5%", pad=0.08)
            plt.colorbar(im, cax=cax)
        elif T == nOut-1:
            fig = plt.figure(figsize=[8, 2])
            ax1 = fig.add_subplot(111)
            plt.plot(test_err, '+-')
            print(f'>>N_fail r{cfg.rRa} {method}{nPC} ={np.sum(test_err>0.3)}')
        plt.tight_layout()
        if base_file is not None:
            figName = base_file + f'nn_testerr_T{T}.pdf'
            plt.savefig(figName, bbox_inches='tight', dpi=200)
        else:
            plt.draw()


def plot_ode_trajs(hvar, ONet, nTraj, nOut, nPC, ns, cfg):
    print('Start calculating ODE trajectories ... ', flush=True)
    hvar = hvar[:, :].reshape([nTraj, 2*nOut, nPC])
    hlims = [np.min(hvar[:, 0]), np.max(hvar[:, 0]),
             np.min(hvar[:, 1]), np.max(hvar[:, 1]),
             np.min(hvar[:, 2]), np.max(hvar[:, 2])]

    iPaths = np.arange(nTraj)
    htest = hvar[iPaths, ns*2::2, :]
    h_ode = ONet.ode_run(htest[:, 0, :], dt=dt*5, T=nOut-ns, Tout=0.2)

    PathOuts = list(range(5)) + list(range(nTraj-20, nTraj))
    for i in PathOuts:   # plot results
        iPath = iPaths[i]
        ii = cfg.Paths
        ii.append(iPath)
        print(f'Generating figure for trajectory {iPath}')
        rbc.plot_2Traj3d_tphase(h_ode[iPath, :, :], htest[iPath, :, :],
                                outloc+f'Traj{iPath}_{method}{nPC}')
        ii.pop()


def plot_ode_trajs_nn(dat, dat_nn, nTraj, nOut, nPC, ns, cfg):
    nf = dat.shape[0]
    htest = dat.reshape([-1, nOut, nPC])
    htest_nn = dat_nn.reshape([-1, nOut, nPC])
    nOut = nOut - ns
    nTraj = htest_nn.shape[0]

    print('Start calculating ODE trajectories ... ')
    hvar = dat.reshape([nTraj, nOut, nPC])
    hlims = [np.min(hvar[:, 0]), np.max(hvar[:, 0]),
             np.min(hvar[:, 1]), np.max(hvar[:, 1]),
             np.min(hvar[:, 2]), np.max(hvar[:, 2])]

    iPaths = np.arange(nTraj)
    htest = hvar[iPaths, ns::, :]

    for iPath in np.arange(nTraj):
        print(f'Generating figure for trajectory {iPath}')
        rbc.plot_2Traj3d_tphase(htest[iPath, :, :], htest_nn[iPath, :, :],
                                outloc+f'Traj{iPath}_nn_{method}{nPC}')


def find_nn(ds1, dt1, ntOut, tst=0):
    nf = ds1.shape[0]
    nft = dt1.shape[0]
    nS_train = nf//ntOut
    nS_test = nft//ntOut
    ds1 = ds1.reshape([nS_train, ntOut, -1])
    dt1 = dt1.reshape([nS_test, ntOut, -1])
    dt1nn = np.zeros_like(dt1)
    for i in np.arange(nS_test):
        mi = np.argmin(np.linalg.norm(ds1[:, tst, :]-dt1[i, tst, :], axis=1))
        dt1nn[i, tst:, :] = ds1[mi, tst:, :]
    return dt1nn


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Test learned ODEs for Rayleigh-Bernard convection data')
    parser.add_argument('-tc', type=int, default=1,
                        metavar='tc',
                        help='the id of test case')
    parser.add_argument('nPC', type=int, nargs='?', default=9, metavar='nPC',
                        help='number of hidden variables')
    parser.add_argument('-m', '--method', type=str, choices=['pca', 'ae'],
                        default='pca', metavar='METHOD',
                        help='method of dim reduction (default pca)')
    parser.add_argument('-o', '--onet',  type=str,
                        choices=['ons', 'ode', 'res'],
                        default='ons', metavar='onet',
                        help='type of the ODE net (default ons)')
    parser.add_argument('-f', '--fid', type=int, default=-1,
                        metavar='FID',
                        help='the id of activation function')
    parser.add_argument('--nHnode',  type=int,
                        default=-1, metavar='nHnode',
                        help='number of nodes in each hidden layers')
    parser.add_argument('--nL',  type=int,
                        default=1, metavar='nHiddenLayers',
                        help='number of hidden layers')
    parser.add_argument('--seed', type=int, default=0, metavar='SEED',
                        help='The first SEED to test the performance')
    parser.add_argument('-ts',  type=int,
                        default=0, metavar='TS',
                        help='number of first several time steps to discard')
    parser.add_argument('-te',  type=int,
                        default=0, metavar='TE',
                        help='number of last several time steps to discard')
    parser.add_argument('--draw_traj', action='store_true',
                        default=True,
                        help='flag to draw trajectories of learned ODE net')
    parser.add_argument('--calc_traj_error', action='store_true',
                        default=True,
                        help='flag to calc trajectory error of learned ODE')

    args = parser.parse_args()
    print(args)

test_id = args.tc
cfg = cfgs.get_test_case(test_id)
outloc = cfg.outloc
h5fname = cfg.h5fname
nTraj = cfg.nTraj
dt = cfg.dt

method = args.method
nPC = args.nPC if args.nPC > 0 else cfg.nPC
onet = args.onet
nL = args.nL
nHnode = args.nHnode if args.nHnode > 0 else int(cfg.iNodeC * binom(nPC+2, 2))
ns = args.ts
te = args.te
ode.fid = args.fid if args.fid >= 0 else cfg.ode_fid
seed = args.seed

# %% 2. Load hidden variables and the learned ODE model
ode_nodes = cfg.get_ode_nodes(nPC, nHnode, nL, onet)
ONet = ode.OnsagerNet(ode_nodes, pot_beta=cfg.pot_beta,
                      ons_min_d=cfg.ons_min_d)

ode_dict_file = outloc + \
    f'_{method}{nPC}_{onet}_f{ode.fid}_L{nL}_s{seed}_model_dict.pth'
ONet.load_state_dict(torch.load(ode_dict_file, map_location='cpu'))

# %% 2. Load data and plot the first three compenents
encfile = h5fname+f'_{method}{nPC}_enc_data.txt.gz'
hvar = np.loadtxt(encfile, delimiter=',')
nf = hvar.shape[0]
nOut = nf//nTraj//2
nS_train = int(nf//2 * cfg.tr_ratio)
nS_test = nf//2 - nS_train
ds1 = hvar[0:2*nS_train:2, :]
ds2 = hvar[1:2*nS_train:2, :]
dt1 = hvar[2*nS_train::2, :]
dt2 = hvar[2*nS_train+1::2, :]
nTrajTest = nS_test//nOut

# %% Solve the ODE system and Check the numerical error
hvar = hvar[2*nS_train:, :]
if args.calc_traj_error:
    base_file = outloc + f'_{method}{nPC}_{onet}_s{seed}'
    check_traj_error_nn(dt1, ds1, dt, ns, nTrajTest, base_file)    
    check_traj_error(hvar, ONet, dt, ns, nTrajTest, base_file)

# %% Test the ode trajectory
if args.draw_traj:
    dt1nn = find_nn(ds1, dt1, nOut, tst=0)
    plot_ode_trajs_nn(dt1, dt1nn, nTrajTest, nOut, nPC, ns, cfg)
    plot_ode_trajs(hvar, ONet, nTrajTest, nOut, nPC, ns, cfg)
