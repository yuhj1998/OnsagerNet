#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
@File 	 : test_ode_RBC.py
@Time 	 : 2020/05/11 22:32:34
@Author  : Haijn Yu <hyu@lsec.cc.ac.cn>
@Desc	 : test the accuracy of learned RBC system
           generate the result for paper.
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
import time
import argparse
import ode_analyzer as oa


def plot_dfield(ONet, hvar, sfile):
    xmin = np.max(hvar)
    xmax = np.min(hvar)
    x = np.linspace(xmin, xmax, 30)
    y = np.linspace(xmin, xmax, 30)
    xx, yy = np.meshgrid(x, y)
    xx = xx.flatten()
    yy = yy.flatten()
    nx = xx.size

    plt.figure(figsize=(14, 14))
    nPC = ONet.nVar
    ds = np.zeros([nx, nPC], dtype=np.float32)
    ds[:, 0] = xx
    ds[:, 1] = yy
    dv = ONet(torch.tensor(ds)).detach().numpy()
    ax1 = plt.subplot(221)
    plt.quiver(xx, yy, dv[:, 0], dv[:, 1])
    ax1.set_aspect('equal', 'box')
    plt.xlabel('$h_1$')
    plt.ylabel('$h_2$')

    ds = np.zeros([nx, nPC], dtype=np.float32)
    ds[:, 0] = xx
    ds[:, 2] = yy
    dv = ONet(torch.tensor(ds)).detach().numpy()
    ax2 = plt.subplot(222)
    plt.quiver(xx, yy, dv[:, 0], dv[:, 2])
    ax2.set_aspect('equal', 'box')
    plt.xlabel('$h_1$')
    plt.ylabel('$h_3$')

    ds = np.zeros([nx, nPC], dtype=np.float32)
    ds[:, 1] = xx
    ds[:, 2] = yy
    dv = ONet(torch.tensor(ds)).detach().numpy()
    ax3 = plt.subplot(223)
    plt.quiver(xx, yy, dv[:, 1], dv[:, 2])
    ax3.set_aspect('equal', 'box')
    plt.xlabel('$h_2$')
    plt.ylabel('$h_3$')

    plt.savefig(sfile, bbox_inches='tight')
    plt.draw()
    plt.pause(5)


def check_fitting_error(hvar, ONet, dt, nTraj, nOut, save_file=None):
    nPC = ONet.nVar
    h1 = torch.FloatTensor(hvar[::2, :])
    h2 = torch.FloatTensor(hvar[1::2, :])
    with torch.no_grad():
        f1 = ONet.forward(h1)
        f2 = ONet.forward(h2)
        hp = (h2-h1)/dt
        ff = (f1+f2)/2
        fit_err = (hp - ff)
    print(fit_err.shape)
    fit_err = fit_err.detach().numpy()
    a_err = 1 - torch.sum(hp*ff, dim=1)/(torch.norm(hp, dim=1)
                                         * torch.norm(ff, dim=1) + 1e-7)
    a_filter = 5e-3 * torch.norm(h1, dim=1) < torch.norm(hp, dim=1)
    a_err = a_err.detach().numpy()
    a_filter = a_filter.detach().numpy()
    a_err = np.where(a_filter, a_err, 1e-8)
    a_err = a_err.reshape([nTraj, nOut])

    hp = np.abs(hp.detach().numpy())

    fit_err = fit_err.reshape([nTraj, nOut, nPC])
    fit_err = np.abs(fit_err)
    hp = hp.reshape([nTraj, nOut, nPC])
    for ir in np.arange(nTraj):
        for ip in np.arange(nPC):
            hp_max = np.max(hp[ir, :, ip])
            fit_err[ir, :, ip] = fit_err[ir, :, ip]/hp_max

    fig = plt.figure(figsize=[9, 7])
    ax1 = fig.add_subplot(221)
    im = plt.imshow(np.log10(fit_err[:, :, 0]))
    ax1.set_title('PC1 fitting error')
    ax1.set_xlabel('Snapshot time')
    ax1.set_ylabel('Path index')
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="5%", pad=0.08)
    plt.colorbar(im, cax=cax)

    ax2 = fig.add_subplot(222)
    im = plt.imshow(np.log10(fit_err[:, :, 1]))
    ax2.set_title('PC2 fitting error')
    plt.xlabel('Snapshot time')
    plt.ylabel('Path index')
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="5%", pad=0.08)
    plt.colorbar(im, cax=cax)

    ax3 = fig.add_subplot(223)
    im = plt.imshow(np.log10(fit_err[:, :, 2]))
    ax3.set_title('PC3 fitting error')
    plt.xlabel('Snapshot time')
    plt.ylabel('Path index')
    divider = make_axes_locatable(ax3)
    cax = divider.append_axes("right", size="5%", pad=0.08)
    plt.colorbar(im, cax=cax)

    ax4 = fig.add_subplot(224)
    im = plt.imshow(np.log10(a_err[:, :]))
    ax4.set_title('filtered angle error (log10)')
    plt.xlabel('Snapshot time')
    plt.ylabel('Path index')
    divider = make_axes_locatable(ax4)
    cax = divider.append_axes("right", size="5%", pad=0.08)
    plt.colorbar(im, cax=cax)

    plt.tight_layout()
    if save_file is not None:
        plt.savefig(save_file, dpi=200, bbox_inches='tight')


def check_traj_error(hvar, ONet, dt, ns, nTraj, base_file=None):
    nf = hvar.shape[0]
    nPC = ONet.nVar
    nOut = nf // (2*nTraj)
    htest = hvar[:, :].reshape([nTraj, 2*nOut, nPC])
    htest = htest[:, 2*ns:, :]
    nOut = nOut - ns
    tOuts = (dt, 1, nOut-1)
    tErrs = np.zeros((len(tOuts),))
    nFail = 0
    for iT in np.arange(len(tOuts)):
        T = tOuts[iT]
        print(f'Evaluating the ODE solution error for T={T} ...')
        ts_ode_run = time.time()
        print('Start calculating ODE trajectories ... ')
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
        test_err[:, it] = terr.reshape([nTraj, -1])
        te_ode_run = time.time()
        print(f'\t Done in {te_ode_run-ts_ode_run:.3e} seconds.')

        L2Amp = np.max(np.sqrt(np.sum(htest**2, axis=2)), axis=1)
        LinfNorm = np.max(np.sqrt(np.sum(htest**2, axis=2)))
        L2Norm = np.sqrt(np.mean(np.sum(htest**2, axis=2), axis=1))
        L2err_ave = np.mean(np.sqrt(np.mean(test_err**2, axis=1)) / L2Norm)
        Linf = np.max(np.max(np.abs(test_err), axis=1))
        for ir in range(nTraj):
            test_err[ir, :] /= L2Amp[ir]
        print(f'The Linf amplitude of the trajectories is {LinfNorm:.3e}')
        tErrs[iT] = L2err_ave
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
            nFail = np.sum(test_err>0.3)
            print(f'\t N_fail r{cfg.rRa} {method}{nPC} ={nFail}')
        plt.tight_layout()
        if base_file is not None:
            figName = base_file + f'_testerr_T{T}.pdf'
            plt.savefig(figName, bbox_inches='tight', dpi=200)
        else:
            plt.draw()
    print(f'>> Tpred errors: nPC={nPC}, {tErrs[1]:.3e}(t=1), {tErrs[2]:.3e}(t=99), nFail={nFail}')
     


def plot_enc_data(hvar, nPC, cfg, nPlot=20):
    """ visualize the encode data to have an overview of the vector field
     of the dynamics system.
     hvar: encode data
     nPC:  number of hidden dimension (primary components)
     nTraj: number of trajectory in hvar
     nPlot: the size of group of trajectories to visualize
     cfg: config data structure
      """
    outloc = cfg.outloc
    nTraj = cfg.nTraj
    PCs = hvar.reshape([-1, nPC])
    nf = PCs.shape[0]
    nOut = nf//nTraj
    for i in np.arange((nTraj//nPlot)):
        iP = i*nOut*nPlot + np.arange(nOut//2*nPlot)*2
        rbc.plot_nTraj_t2d(PCs[iP, :], outloc +
                           f'_plot{i}_p{1}p{2}', nPlot)
        rbc.plot_nTraj2d_phase(PCs[iP, :], outloc +
                               f'_plot{i}_p{1}p{2}', nPlot)
    print(f'Choose traj from {outloc}_3pc_plot.pdf for better fig')
    nP = min(nTraj, 40) * nOut
    rbc.plot_nTraj3d_scatter(
        PCs[:nP:2, :], outloc+'_3pc_scatter', nP//nOut, ts=0)

    tPC = PCs.reshape([nTraj, nOut, -1])
    tPC = tPC[cfg.Paths, ::2, :].reshape([-1, nPC])
    rbc.plot_nTraj_t2d(tPC[:, :], outloc+'_3pc_plot_ref', len(cfg.Paths))
    rbc.plot_nTraj2d_phase(tPC[:, :], outloc+'_3pc_plot_ref', len(cfg.Paths))
    rbc.plot_nTraj3d_scatter(
        tPC[:, :], outloc+'_3pc_scatter_ref', len(cfg.Paths), ts=0)


def plot_ode_trajs(hvar, ONet, nTraj, nOut, nPC, ns, cfg):
    print('Start calculating ODE trajectories ... ')
    hvar = hvar[:, :].reshape([nTraj, 2*nOut, nPC])
    hlims = [np.min(hvar[:, 0]), np.max(hvar[:, 0]),
             np.min(hvar[:, 1]), np.max(hvar[:, 1]),
             np.min(hvar[:, 2]), np.max(hvar[:, 2])]

    iPaths = np.arange(nTraj)
    ts_ode_run = time.time()
    htest = hvar[iPaths, ns*2::2, :]
    h_ode = ONet.ode_run(htest[:, 0, :], dt=dt*5, T=nOut-ns, Tout=0.2)
    te_ode_run = time.time()
    print(f'\t Done in {te_ode_run-ts_ode_run:.2e} seconds.')

    PathOuts = list(range(5)) + list(range(nTraj-20, nTraj))
    for i in PathOuts:   # plot results
        iPath = iPaths[i]
        ii = cfg.Paths
        ii.append(iPath)
        print(f'Generating figure for trajectory {iPath}')
        rbc.plot_2Traj3d_tphase(h_ode[iPath, :, :], htest[iPath, :, :],
                                outloc+f'Traj{iPath}_{method}{nPC}')
        ii.pop()


def analyze_RBC_structure(ONet, r, lr=0.0128, niter=100, x0init=[]):
    n = ONet.nVar
    m = 4  # number of critical points
    x0 = np.zeros((m, n), dtype=np.float)   # initial guess
    if x0init is not None:
        x0 = x0init
    elif np.abs(r-28) < 1e-2:  # input initial by hand
        print(f'evaluating r={r} ...')
        x0[0, 0:3] = [-18, -1, -1]
        x0[1, 0:3] = [21, -1, -1]
        x0[2, 0:3] = [5, -12, 4]
        x0[3, 0:3] = [-4, 12, 0]
    elif np.abs(r-84) < 1e-2:
        print(f'evaluating r={r} ...')
        x0[0, 0:3] = [-15.5, 0, 0]
        x0[1, 0:3] = [16, 0, 0]
        x0[2, 0:3] = [0.5, -8.5, 1]
        x0[3, 0:3] = [0.5, 9, 1]

    x = np.zeros((m, n), dtype=np.float)    # found
    iflag = [False, ]*m        # flag of initial point
    x0 = torch.tensor(x0).float()

    for i in np.arange(m):
        xres = oa.find_fixed_pt(ONet, x0[i])
        x[i] = xres[0]
        # the first item in the tuple is fixed pt
        print(f'Initial guess: x0{i} = {x0[i]}')
        print(f'The found fixed point is x{i}={x[i]}')
        # check the solution
        xt = torch.tensor(x[i]).float()
        ft = ONet(xt).detach().numpy()
        if np.linalg.norm(ft, ord=2) < 1e-3:
            iflag[i] = True
        print(f'Check the fixed point found: f(x[{i}]) = {ft}\n')
    fixed_pts = x[iflag]

    # %% check the eigenvalues of the fix points
    nf = fixed_pts.shape[0]
    with torch.no_grad():
        for i in np.arange(nf):
            matA = ONet.calc_Jacobi(fixed_pts[i])
            if matA.shape[0] == 1:
                matA = torch.squeeze(torch.mean(matA, dim=0))
            eigval = torch.eig(matA, eigenvectors=False).eigenvalues
            ind = torch.argsort(eigval[:, 0], descending=True)
            eigval = eigval[ind, :]
            print(f'The eigenvalues for the Jacobi at x{i} is:')
            print(eigval)

    if r > 40:
        nLC = 4  # fixed_pts.shape[0]
        try:
            lcs = np.loadtxt(outloc+f'_{method}{nPC}_{onet}_lcs.txt', delimiter=',')
            x0 = lcs[:, 0:nPC]
            T0 = lcs[:, nPC]
        except IOError:
            print('No existing inital data for limit cycles, calculate them ...')
            x0 = np.zeros((nLC, ONet.nVar))
            T0 = np.zeros((nLC,))
            xtmp = [-14.7150,  -0.3808,   1.4176,   0.5031,  -0.7849,   0.7051,
                    -0.1160,  0.4597,  -0.0593,   0.5284,  -0.0966]
            x0[0] = xtmp[:ONet.nVar]
            T0[0] = 5.379021644592285
            xtmp = [15.7091,  0.1504,  1.3657, -0.6149, -0.6824,  0.7195,
                    0.5850,  0.0208, 0.0950, -0.6287, -0.0710]
            x0[1] = xtmp[:ONet.nVar]
            T0[1] = 5.373147964477539
            xtmp = [6.827191e-01, -8.861500e+00, 1.922136e+00, -3.808442e-01,
                    9.773142e-01, -1.093930e+00, 6.303264e-01, -1.027560e+00,
                    -1.969296e-01, 8.459229e-01, 6.384754e-01]
            x0[2] = xtmp[:ONet.nVar]
            T0[2] = 24.005834579467773
            xtmp = [1.240431e+00, 8.000000e+00, 2.461909e+00, 7.033594e-02,
                    7.484778e-01, -1.706734e+00, -6.783456e-01, -3.501478e-01,
                    -4.248601e-01, -2.328687e-01, 6.134876e-02]
            x0[3] = xtmp[:ONet.nVar]
            T0[3] = 21.956218719482422
    else:
        nLC = 0

    lcs = np.zeros((nLC, nPC+1), dtype=float)
    print('Start to calculate the limit cycles ...')
    for i in np.arange(nLC):
        lcs[i, 0:nPC], lcs[i, nPC] = oa.find_limit_cycle(ONet, x0[i], T0[i],
                                                         niter=niter, lr=lr)
        print('The start point and poerid of limit cycle is\n',
              f' xlc[{i}]={lcs[i,0:nPC]}\n',
              f' Tlc[{i}]={lcs[i, nPC]}')
    return fixed_pts, lcs


def long_run_cmp_RBC(hvar, onet: ode.ODENet, T, fixed_pts, lcs=None,
                     dt=0.005, nOut=200, savefile='run_cmp'):
    ''' compare the learned RBC model by comparing to sample data
        with fixed points and limit cycles added
        @hvar : the encoded data,
        @onet2 : learned ODE
    '''
    nPC = onet.nVar
    nTraj = hvar.shape[0]
    p1 = torch.tensor(hvar.reshape([nTraj, -1, nPC])).float()
    p2 = torch.zeros(nTraj, nOut, nPC)
    p2[:, 0, 0:nPC] = p1[:, 0, 0:nPC]
    with torch.no_grad():
        print('Calculating evaluation data ...', end=' ')
        for i in range(nOut-1):
            nt = int(T/nOut/dt)
            p2[:, i+1, :] = onet.ode_rk3(p2[:, i, :], dt, nt)
        print('done.')

    n = onet.nVar
    if lcs is not None:
        nLC = lcs.shape[0]
        pLC = torch.zeros(nLC, nOut, n)
        pLC[:, 0, 0:n] = torch.tensor(lcs[:, 0:n]).float()
        nt = 2
        with torch.no_grad():
            print('Calculating limit cycle data ...', end=' ')
            for ip in range(nLC):
                Tlc = lcs[ip, n]
                for i in range(nOut-1):
                    dt = Tlc/(nOut-1)/nt
                    pLC[ip, i+1, :] = onet.ode_rk3(pLC[ip, i, :], dt, nt)
            print('done.')
    else:
        nLC = 0

    f = plt.figure(figsize=[8.5, 4], dpi=288)

    ax = f.add_subplot(121)  # plot multiple paths
    for ip in np.arange(nTraj):
        plt.plot(p1[ip, :, 0], p1[ip, :, 1], '.', color='C3',
                 markersize=3, alpha=0.9, zorder=1)
        plt.plot(p2[ip, :, 0], p2[ip, :, 1], color='C0',
                 linewidth=1, alpha=0.8, zorder=3)
    for ip in np.arange(nLC):
        plt.plot(pLC[ip, :, 0], pLC[ip, :, 1], color='black',
                 linewidth=1.5, alpha=1, zorder=4)
    ax.scatter(fixed_pts[:, 0], fixed_pts[:, 1], color='red',
               marker='+', alpha=1, edgecolors=None,
               zorder=5)
    for ip in np.arange(fixed_pts.shape[0]):
        ax.text(fixed_pts[ip, 0]+1, fixed_pts[ip, 1]+0.5, 'ABCD'[ip],
                color='black', fontsize=14, zorder=5)
    if nLC > 2:   # r= 84
        ax.set_yticks([-10, -5, 0, 5, 10])
    else:
        ax.set_yticks([-16, -8, 0, 8, 16])
    plt.xlabel('$h_1$')
    plt.ylabel('$h_2$')
    plt.grid(True, which='major')

    ax = f.add_subplot(122)
    if nLC < 4:
        for ip in np.arange(nTraj):
            plt.plot(p1[ip, :, 0], -p1[ip, :, 2], '.', color='C3',
                     markersize=3, alpha=0.9, zorder=1)
            plt.plot(p2[ip, :, 0], -p2[ip, :, 2], color='C0',
                     linewidth=1, alpha=0.8, zorder=3)
        for ip in np.arange(nLC):
            plt.plot(pLC[ip, :, 0], -pLC[ip, :, 2], color='black',
                     linewidth=1.5, alpha=1, zorder=4)
        ax.scatter(fixed_pts[:, 0], -fixed_pts[:, 2], color='red',
                   marker='+', alpha=1, edgecolors=None,
                   zorder=5)
        for ip in np.arange(fixed_pts.shape[0]):
            ax.text(fixed_pts[ip, 0]+1, -fixed_pts[ip, 2]+1, 'ABCD'[ip],
                    color='black', fontsize=14, zorder=5)
        plt.xlabel('$h_1$')
        plt.ylabel('$h_3$')
        if nLC >= 4:
            ax.set_yticks([-10, -5, 0, 5, 10])
    else:  # the r=84 case
        for ip in np.arange(nTraj):
            plt.plot(p1[ip, :, 0]/2+p1[ip, :, 2], p1[ip, :, 1], '.', color='C3',
                     markersize=3, alpha=0.9, zorder=1)
            plt.plot(p2[ip, :, 0]/2+p2[ip, :, 2], p2[ip, :, 1], color='C0',
                     linewidth=1, alpha=0.8, zorder=3)
        for ip in np.arange(nLC):
            plt.plot(pLC[ip, :, 0]/2+pLC[ip, :, 2], pLC[ip, :, 1], color='black',
                     linewidth=1.5, alpha=1, zorder=4)
        ax.scatter(fixed_pts[:, 0]/2+fixed_pts[:, 2], fixed_pts[:, 1], color='red',
                   marker='+', alpha=1, edgecolors=None,
                   zorder=5)
        for ip in np.arange(fixed_pts.shape[0]):
            ax.text(fixed_pts[ip, 0]/2+fixed_pts[ip, 2]-0.,
                    fixed_pts[ip, 1]-1.2, 'ABCD'[ip],
                    color='black', fontsize=14, zorder=5)
        plt.xlabel('$h_1$/2+$h_3$')
        plt.ylabel('$h_2$')
        ax.set_yticks([-8, -4, 0, 4, 8])
    plt.grid(True, which='major')

    plt.savefig(savefile+'.pdf', bbox_inches='tight', dpi=288)


def calc_Lyapunov_exp1(ONet, nS=1, T0=5, nOut=5000, dt=0.01, dt_out=0.1,
                       region=[-25., 25, -25, 25, -5, 30],
                       fname='RBC_ode_samples.txt'):
    n = ONet.nVar
    T = int(dt_out*nOut)
    nOut_tot = int((T+T0)/dt_out)
    print(f'T={T}, T0={T0}, nOut={nOut}, dt_out={dt_out}, dt={dt}')
    print('Calculate trajectories from learned ODE system ...')
    paths = torch.zeros(nS, nOut_tot, n)
    for i in np.arange(min(n, 3)):
        paths[:, 0,  i] = torch.Tensor(nS).uniform_(region[2*i], region[2*i+1])
    with torch.no_grad():
        nt = int(dt_out/dt)
        for i in range(nOut_tot-1):
            paths[:, i+1, :] = ONet.ode_rk3(paths[:, i, :], dt, nt)

    LyaInd = np.zeros((nS,), dtype=np.float)
    for i in np.arange(nS):
        data = paths[i, nOut_tot-nOut:, 0].numpy()  # only need first component
        print(f'Estimate the Largest Lyapunov index for Traj {i}...')
        x = np.arange(nOut) * dt_out
        Tmean = oa.plot_fft(x, data)
        K = int(20/dt_out)  # need to check here
        P = nOut//15
        print('Tmean=', Tmean)
        LyaInd[i], yy = oa.estimate_Lyapunov_exp1(data, dt_out,
                                                  P=P, J=11, m=n, K=K)


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
    parser.add_argument('-p', '--print_net', action='store_true',
                        help='flag to print the coefficients of ODE net')
    parser.add_argument('--draw_traj', action='store_true',
                        default=False,
                        help='flag to draw trajectories of learned ODE net')
    parser.add_argument('--dfield',  action='store_true',
                        default=False,
                        help='flag to draw the vector field learned')
    parser.add_argument('--calc_fit_error', action='store_true',
                        default=False,
                        help='flag to draw the fitting error in learning ODE')
    parser.add_argument('--calc_traj_error', action='store_true',
                        default=False,
                        help='flag to calc trajectory error of learned ODE')
    parser.add_argument('--plot_enc', action='store_true',
                        default=False,
                        help='flag to plot the encoded sample data')
    parser.add_argument('--draw_structure', action='store_true',
                        default=False,
                        help='flag to draw structure of learned ODE')
    parser.add_argument('--calc_Lindex', action='store_true',
                        default=False,
                        help='flag to calc the Lyapunov index of learned ODE')
    parser.add_argument('-e', '--epochs', type=int, default=50,
                        metavar='epochs',
                        help='input number of epochs')
    parser.add_argument('-lr', type=float, default=0.0016, metavar='LR',
                        help='learning rate')

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
ts_loaddata = time.time()

ode_nodes = cfg.get_ode_nodes(nPC, nHnode, nL, onet)
if onet == 'ode':
    ONet = ode.ODENet(ode_nodes)
elif onet == 'res':
    ONet = ode.ResODENet(ode_nodes)
else:
    ONet = ode.OnsagerNet(ode_nodes, pot_beta=cfg.pot_beta,
                          ons_min_d=cfg.ons_min_d)

if method == 'pca':
    savefile = outloc + f'_{method}{nPC}_{onet}_f{ode.fid}_L{nL}_s{seed}'
    ode_dict_file = savefile+'_model_dict.pth'
    ONet.load_state_dict(torch.load(ode_dict_file, map_location='cpu'))
else:
    ode_dict_file = outloc+f'_ae{nPC}_{onet}_model_dict_L{nL}_s{seed}.pth'
    ONet.load_state_dict(torch.load(ode_dict_file, map_location='cpu'))
te_loaddata = time.time()
print(f'Loading {ode_dict_file} in {te_loaddata-ts_loaddata:.3e} seconds.')

# %% 2. Load data and plot the first three compenents
if method == 'pca':
    encfile = h5fname+f'_{method}{nPC}_enc_data.txt.gz'
else:
    encfile = outloc+f'_{method}{nPC}_{onet}_enc_data.txt.gz'
hvar = np.loadtxt(encfile, delimiter=',')
nf = hvar.shape[0]
nOut = nf//nTraj//2

# %% 3. draw the learned rhs function
if args.dfield:
    dFieldFile = h5fname+f'_all_{method}{nPC}_dfield_learned.pdf'
    plot_dfield(ONet, hvar, dFieldFile)

# %% output ONet's coefficient
if args.print_net:
    ONet.print()

# %% plot the numerical fitting error
if args.calc_fit_error:
    fiterr_file = outloc+f'_{method}{nPC}_{onet}_fiterr.pdf'
    check_fitting_error(hvar, ONet, dt, nTraj, nOut, fiterr_file)

# %% Solve the ODE system and Check the numerical error
if args.calc_traj_error:
    savefile = outloc + f'_{method}{nPC}_{onet}_f{ode.fid}_L{nL}_s{seed}'
    check_traj_error(hvar, ONet, dt, ns, nTraj, savefile)

# %% visualize the encoder data
if args.plot_enc:
    plot_enc_data(hvar, nPC, cfg, nPlot=20)

# %% Test the ode trajectory
if args.draw_traj:
    plot_ode_trajs(hvar, ONet, nTraj, nOut, nPC, ns, cfg)

# %% find the critical points
if args.draw_structure:
    r = cfg.rRa
    ns = 0
    iPaths = cfg.Paths
    print(f'Calculate and draw structure for r={r}, T={nOut-ns} ...')
    PCs = hvar[::2, :].reshape([nTraj, nOut, nPC])
    PCs = PCs[iPaths, ns:, :].reshape([len(iPaths), -1])
    x0init = np.mean(PCs.reshape([len(iPaths), -1, nPC]), axis=1)

    fixed_pts, lcs = analyze_RBC_structure(ONet, r, lr=args.lr,
                                           niter=args.epochs,
                                           x0init=x0init)
    np.savetxt(outloc+f'_{method}{nPC}_{onet}_fixpts.txt',
               fixed_pts, delimiter=', ', fmt='%.8e')
    np.savetxt(outloc+f'_{method}{nPC}_{onet}_lcs.txt',
               lcs, delimiter=', ', fmt='%.8e')

    np.savetxt(outloc+f'_{method}{nPC}_{onet}_lcs.txt',
               lcs, delimiter=', ', fmt='%.8e')
    #base_file = outloc + f'_{method}{nPC}_{onet}'
    savefile = outloc + f'_{method}{nPC}_{onet}_f{ode.fid}_L{nL}_s{seed}'

    PCs = hvar[::2, :]
    PCs = PCs.reshape([nTraj, -1, nPC])[cfg.Paths, ...]
    long_run_cmp_RBC(PCs, ONet, T=nOut,
                     fixed_pts=fixed_pts, lcs=lcs,
                     dt=0.01, nOut=1000,
                     savefile=savefile+'_structure')

# %% calculate the Lyapunov index for randomly sampled pathes
if args.calc_Lindex:
    region = [hvar[:, 0].min(), hvar[:, 0].max(),
              hvar[:, 1].min(), hvar[:, 1].max(),
              hvar[:, 2].min(), hvar[:, 2].max()]
    if cfg.rRa > 30:
        calc_Lyapunov_exp1(ONet, nS=50, T0=40, nOut=8000, dt=0.02,
                           dt_out=0.1, region=region)
    else:
        calc_Lyapunov_exp1(ONet, nS=50, T0=25, nOut=2000, dt=0.05,
                           dt_out=0.1, region=region)
