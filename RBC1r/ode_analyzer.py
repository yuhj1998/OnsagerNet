#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
@File 	 : ode_analyzer.py
@Time 	 : 2020/07/21
@Author :  Haijn Yu <hyu@lsec.cc.ac.cn>
@Desc	 : Analyze the learned OnsagerNet for Lorenz system
          see more descriptions and update on [GitHub](https://github.com/yuhj1998/ode-analyzer)
'''

# %% 1. import library and set parameters
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import ode_net as onet
import argparse
import scipy.optimize as pyopt
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rc('lines', linewidth=1, markersize=2)
float_formatter = "{:.6e}".format
np.set_printoptions(formatter={'float_kind': float_formatter})


def long_run_cmp_Lorenz(onet1: onet.ODENet, onet2: onet.ODENet,
                        T, fixed_pts, lcs=None,
                        dt=0.001, nOut=100,
                        region=[-10.0, 10, -10, 10, 0, 1],
                        savefile='run_cmp'):
    ''' based on long_run_cmp function in test_ode_Lorenz.py
        with fixed points and limit cycles added
        @onet1 : underlying ODE,
        @onet2 : learned ODE
    '''
    nS = 4
    n = onet1.nVar
    if T/nOut > 0.025:
        nOut = int(T/0.025)
    p1 = torch.zeros(nOut, nS, n)
    p2 = torch.zeros(nOut, nS, n)
    p1[0, :, 0] = torch.tensor([region[0], -1.0, 1.0, region[1]]).float()
    p1[0, :, 1] = torch.tensor([region[2], -1.0, 1.0, region[3]]).float()
    p1[0, :, 2] = torch.tensor(region[4]).float()
    p2[0, :, 0:n] = p1[0, :, 0:n]
    with torch.no_grad():
        print('Calculating evaluation data ...', end=' ')
        for i in range(nOut-1):
            nt = int(T/nOut/dt)
            p1[i+1, ...] = onet1.ode_rk3(p1[i, ...], dt, nt)
            p2[i+1, ...] = onet2.ode_rk3(p2[i, ...], dt, nt)
        print('done.')

    L2err_pts = torch.sum((p1-p2)**2, dim=2).sqrt()
    Linf = torch.max(L2err_pts)
    L2nrm_pth = torch.sqrt(torch.sum(p1**2, dim=[0, 2]) * T/nOut)
    L2err_pth = torch.sqrt(torch.sum(L2err_pts**2, dim=0)*T/nOut)
    L2err_rel = torch.sqrt(torch.sum((L2err_pth/L2nrm_pth)**2)/nS)
    print(f'The maximum point error for {nS} path is {Linf:.6e}')
    print(f'The average L2norm error for {nS} path is {L2err_rel:.6e}')

    n = onet2.nVar
    if lcs is not None:
        nLC = lcs.shape[0]
        pLC = torch.zeros(nOut, nLC, n)
        pLC[0, :, 0:n] = torch.tensor(lcs[:, 0:n]).float()
        nt = 10
        with torch.no_grad():
            print('Calculating limit cycle data ...', end=' ')
            for ip in range(nLC):
                Tlc = lcs[ip, n]
                for i in range(nOut-1):
                    dt = Tlc/(nOut-1)/nt
                    pLC[i+1, ip, :] = onet2.ode_rk3(pLC[i, ip, :], dt, nt)
            print('done.')
    else:
        nLC = 0

    f = plt.figure(figsize=[12, 10], dpi=144)
    dt_out = T/nOut
    ax = f.add_subplot(311)
    nErrOut = nOut
    ii = np.arange(nErrOut)
    tt = ii*dt_out
    ipp = L2err_pth.argmax() 
    plt.plot(tt, p2[ii, ipp, 0], label='X learned ODE')
    plt.plot(tt, p1[ii, ipp, 0], '.', markersize=2, zorder=3,
             alpha=0.8, label='X original ODE')
    plt.plot(tt, p2[ii, ipp, 1], label='Y learned ODE')
    plt.plot(tt, p1[ii, ipp, 1], '.', markersize=2, zorder=3,
             alpha=0.8, label='Y original ODE')
    plt.plot(tt, p2[ii, ipp, 2], label='Z learned ODE')
    plt.plot(tt, p1[ii, ipp, 2], '.', markersize=2, zorder=3,
             alpha=0.8, label='Z original ODE')
    ax.set_title('Trajectory with max error')
    plt.tick_params(axis='y', which='both', labelleft='on', labelright='on')
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.xlabel('t')
    plt.legend(fontsize='small', ncol=3, loc="best")

    ax = f.add_subplot(312)
    plt.plot(tt, p1[ii, ipp, 0]-p2[ii, ipp, 0], label='X error')
    plt.plot(tt, p1[ii, ipp, 1]-p2[ii, ipp, 1], label='Y error')
    plt.plot(tt, p1[ii, ipp, 2]-p2[ii, ipp, 2], label='Z error')
    plt.tick_params(axis='y', which='both', labelleft='on', labelright='on')
    plt.xlabel('t')
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.legend(fontsize='small', loc=0, ncol=3)

    ax = f.add_subplot(337) 
    for ip in np.arange(nS):
        plt.plot(p1[:, ip, 0], p1[:, ip, 1], '.',
                 markersize=1, alpha=0.8, zorder=4)
        plt.plot(p1[:, ip, 0], p1[:, ip, 1], color='grey',
                 linewidth=0.5, alpha=0.2, zorder=1)
        if ip == ipp:
            plt.plot(p2[:, ip, 0], p2[:, ip, 1], color='C3',
                     linewidth=1, alpha=0.9, zorder=3)
        else:
            plt.plot(p2[:, ip, 0], p2[:, ip, 1], color='C0',
                     linewidth=0.75, alpha=0.7, zorder=2)
        for ip in np.arange(nLC):
            plt.plot(pLC[:, ip, 0], pLC[:, ip, 1], color='yellow',
                     linewidth=1, alpha=0.6, zorder=4)
        ax.scatter(fixed_pts[:, 0], fixed_pts[:, 1], color='red',
                   marker='+', alpha=0.9, edgecolors=None,
                   zorder=5)
    plt.xlabel('X')
    plt.ylabel('Y')

    ax = f.add_subplot(338)
    for ip in np.arange(nS):
        plt.plot(p1[:, ip, 0], p1[:, ip, 2], '.',
                 markersize=1, alpha=0.8, zorder=4)
        plt.plot(p1[:, ip, 0], p1[:, ip, 2], color='grey',
                 linewidth=0.5, alpha=0.2, zorder=1)
        if ip == ipp:
            plt.plot(p2[:, ip, 0], p2[:, ip, 2], color='C3',
                     linewidth=1, alpha=0.9, zorder=3)
        else:
            plt.plot(p2[:, ip, 0], p2[:, ip, 2], color='C0',
                     linewidth=0.75, alpha=0.7, zorder=2)
        for ip in np.arange(nLC):
            plt.plot(pLC[:, ip, 0], pLC[:, ip, 2], color='yellow',
                     linewidth=1, alpha=0.6, zorder=4)
        ax.scatter(fixed_pts[:, 0], fixed_pts[:, 2], color='red',
                   marker='+', alpha=0.9, edgecolors=None,
                   zorder=5)
    plt.xlabel('X')
    plt.ylabel('Z')

    ax = f.add_subplot(339)
    for ip in np.arange(nS):
        plt.plot(p1[:, ip, 1], p1[:, ip, 2], '.',
                 markersize=1, alpha=0.8, zorder=4)
        plt.plot(p1[:, ip, 1], p1[:, ip, 2], color='grey',
                 linewidth=0.5, alpha=0.2, zorder=1)
        if ip == ipp:
            plt.plot(p2[:, ip, 1], p2[:, ip, 2], color='C3',
                     linewidth=1, alpha=0.9, zorder=3)
        else:
            plt.plot(p2[:, ip, 1], p2[:, ip, 2], color='C0',
                     linewidth=0.75, alpha=0.7, zorder=2)
        for ip in np.arange(nLC):
            plt.plot(pLC[:, ip, 1], pLC[:, ip, 2], color='yellow',
                     linewidth=1, alpha=0.6, zorder=4)
        ax.scatter(fixed_pts[:, 1], fixed_pts[:, 2], color='red',
                   marker='+', alpha=0.9, edgecolors=None,
                   zorder=5)
    plt.xlabel('Y')
    plt.ylabel('Z')

    plt.savefig(savefile+'.pdf', bbox_inches='tight', dpi=288)


def plot_ode_structure(ode_net: onet.ODENet,
                       fixed_pts, lcs, nOut=100, savefile=None):
    ''' ode_net, the ode system
        @fixed_pts, an numpy arrange store fixed points
        @lcs, a numpy array store the start points and periods of
        limit cycles
    '''
    n = ode_net.nVar
    nLC = lcs.shape[0]
    pLC = torch.zeros(nOut, nLC, n)
    tt = torch.zeros(nOut, nLC)
    pLC[0, :, 0:n] = torch.tensor(lcs[:, 0:n]).float()
    tt[0, :] = 0.
    nt = 10
    with torch.no_grad():
        print('Calculating evaluation data ...', end=' ')
        for ip in range(nLC):
            T = lcs[ip, n]
            for i in range(nOut-1):
                dt = T/(nOut-1)/nt
                pLC[i+1, ip, :] = ode_net.ode_rk3(pLC[i, ip, :], dt, nt)
        print('done.')

    f = plt.figure(figsize=[12, 4], dpi=144)

    ax = f.add_subplot(131)
    for ip in np.arange(nLC):
        plt.plot(pLC[:, ip, 0], pLC[:, ip, 1],
                 linewidth=1, alpha=0.5, zorder=1)
    ax.scatter(fixed_pts[:, 0], fixed_pts[:, 1], color='red',
               marker='+', alpha=0.9, edgecolors=None)
    plt.xlabel('X')
    plt.ylabel('Y')

    ax = f.add_subplot(132)
    for ip in np.arange(nLC):
        plt.plot(pLC[:, ip, 0], pLC[:, ip, 2],
                 linewidth=1, alpha=0.5, zorder=1)
    ax.scatter(fixed_pts[:, 0], fixed_pts[:, 2], color='red',
               marker='.', alpha=0.8, edgecolors=None,
               zorder=3)
    plt.xlabel('X')
    plt.ylabel('Z')

    ax = f.add_subplot(133)
    for ip in np.arange(nLC):
        plt.plot(pLC[:, ip, 1], pLC[:, ip, 2],
                 linewidth=1, alpha=0.5, zorder=1)
    ax.scatter(fixed_pts[:, 1], fixed_pts[:, 2], color='red',
               marker='.', alpha=0.8, edgecolors=None,
               zorder=3)
    plt.xlabel('Y')
    plt.ylabel('Z')

    plt.tight_layout()
    plt.savefig(savefile+'_structure.pdf', bbox_inches='tight', dpi=144)


def find_fixed_pt(ode, x0):
    def ode_fun(x):
        ''' need convert x to tensor '''
        shape = x.shape
        x0 = torch.tensor(x).float().view(-1, ode.nVar)
        f = ode(x0)
        return f.detach().numpy().reshape(shape)

    xfix = pyopt.fsolve(ode_fun, x0, full_output=1)
    return xfix


def find_limit_cycle(ode_net, x0, T0, niter=100, lr=0.00128, d_fix=1):
    """ Find the limit cycle for given initial points and period
        by using least square method with Adam optimizer in pytorch
        d_fix: x0[d_fix] is fixed
     """
    x = torch.nn.Parameter(torch.tensor(x0, requires_grad=True).float())
    T = torch.nn.Parameter(torch.tensor(T0, requires_grad=True).float())
    optimizer = optim.Adam([{'params': [x, T]}], lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                     'min',
                                                     factor=0.5,
                                                     patience=8)
    nt = 80
    T_beta = 10
    Tth = torch.tensor(T0, requires_grad=False).float()
    for e in range(niter):
        xmiddle = ode_net.ode_rk3(x, T/2./nt, nt, test=False)
        xt = ode_net.ode_rk3(xmiddle, T/2./nt, nt, test=False)
        loss = (torch.sum((xt-x)**2) + T_beta * F.relu(Tth/3-T)
                + T_beta * F.relu(1.0-torch.sum(xmiddle-x)**2))
        optimizer.zero_grad()
        loss.backward()
        x.grad.data[d_fix] = 0
        nn.utils.clip_grad_norm_([x, T], 1.0)
        optimizer.step()

        scheduler.step(loss)
        last_lr = optimizer.param_groups[0]["lr"]
        if loss < 9e-5:
            break

        if e % 5 == 0 or e == niter-1:
            print(f'iter:{e+1:4d}/{niter}', end=' ')
            print(f'loss: {loss.item():.3e}', end=' ')
            print(f'x: {x.data}', end=' ')
            print(f'T: {T.data}', end=' ')
            print(f'lr: {last_lr}', flush=True)
    return x.detach().data, T.detach().data


def estimate_Lyapunov_exp1(data, dt, P, J, m, K):
    ''' a quick implementation
        dt: the time stepsize of the series
        P:  mean period
        J:  time lag
        m:  embedding dimension
        K:  number of distances used to fit the index
    '''
    # Step 0: prepare X
    N = len(data)
    M = N - (m-1)*J
    nbs = np.zeros((M-2*K, ), dtype=int)
    d = np.zeros((2*K, M-2*K), dtype=np.float64)
    Xt = np.zeros((m, M), dtype=np.float64)
    dmax = np.sqrt((np.max(data) - np.min(data))**2 * m) + 1.0
    for j in np.arange(m):
        Xt[j, :] = data[j*J:j*J+M]
    X = Xt.transpose()

    # Step 1: find neighbor index with minum distance to i
    #         but with index distance > P
    for j in np.arange(M-2*K):
        dist = np.linalg.norm(X[0:M-2*K, :] - X[j, :],  ord=2, axis=1)
        ii = np.arange(M-2*K)
        i_mask = np.logical_and(ii >= j-P, ii <= j+P)
        dist[i_mask] = dmax
        nbs[j] = np.argmin(dist)

    # Step 2: calculate d_j(i)
    for i in np.arange(2*K):
        j = np.arange(M-2*K)
        j1 = j + i
        j2 = nbs[j] + i
        d[i, j] = np.linalg.norm(X[j1, :]-X[j2, :], ord=2, axis=1)

    # Step 3: average over j
    y = np.mean(np.log(d+1e-20), axis=1) / dt
    ii = np.arange(int(0.2*K), 2*K) 
    poly = np.polyfit(ii, y[ii], deg=1)
    print('lsq coef =', poly)
    print('Lyapunov index ~=', poly[0])
    plt.subplot(224)
    plt.plot(y)
    plt.xlabel('k')
    plt.ylabel('<log(d(k))>')
    plt.title(f'Estimated Lyapunov index ~={poly[0]}')
    plt.draw()
    plt.pause(1)
    plt.close()
    return poly[0], y


def plot_fft(x, y, th=1e-4):
    """ Do FFT analysis on time series, find its mean period
        x: independ variable
        y: depend variable
        th: threshold below which the frequency will not be plotted
    """
    n = x.size
    Lx = x[-1]-x[0]
    yf = np.fft.rfft(y)
    xf = np.fft.rfftfreq(n, d=Lx/n)
    fig = plt.figure(figsize=[9, 9])
    ax = fig.add_subplot(211)
    ax.plot(x, y)
    plt.title('1) first component of ODE solution')

    ax = fig.add_subplot(223)
    yf = yf / (n/2)
    ii = (np.abs(yf) > th)
    ii[0] = False
    plt.plot(xf[ii], np.abs(yf[ii]))
    T0 = 1.0/np.mean(xf*np.abs(yf))
    plt.title('2) power spectrum')
    plt.draw()
    plt.pause(2)
    return T0


def calc_Lyapunov_exp1(lz_net, T0=5, nOut=5000, dt=0.01):
    print(f'T0={T0}, nOut={nOut}, dt={dt}')
    T = int(dt*nOut)
    nOut_tot = int((T+T0)/dt)
    Path = lz_net.gen_sample_paths(nS=1, T=T+T0, dt=0.001, nOut=nOut_tot)
    data = Path[::2, 0].numpy()
    data = data[nOut_tot-nOut:]
    print(f'dt={dt}, T={T},  len(data)=', data.shape)
    x = np.arange(nOut) * dt
    Tmean = plot_fft(x, data)
    K = int(2/dt)
    P = nOut//15
    print('Tmean=', Tmean)
    Lindex, yy = estimate_Lyapunov_exp1(data, dt, P=P, J=11, m=5, K=K)


def analyze_Lorenz_structure(ode_net, r, b, lr, niter):
    q = np.sqrt(b*(r-1.0))
    x10 = np.array([q, q, r-1.0])
    x20 = np.array([-q, -q, r-1.0])
    x10 = torch.tensor(x10).float()
    x20 = torch.tensor(x20).float()
    x1 = find_fixed_pt(ode_net, x10)
    x2 = find_fixed_pt(ode_net, x20)
    print('Initial guess of two fixed points are\n',
          f' x1={x10}\n',
          f' x2={x20}')
    print('The two fixed points found are\n',
          f' x1={x1}\n',
          f' x2={x2}')
    x1 = x1[0]
    x2 = x2[0]
    x1t = torch.tensor(x1).float()
    f1 = ode_net(x1t)
    x2t = torch.tensor(x2).float()
    f2 = ode_net(x2t)
    print('Check the two fixed points found\n',
          f' f(x1)={f1}\n',
          f' f(x2)={f2}')
    fixed_pts = np.vstack((x1, x2))

    if r == 16:
        nLC = 2
        x0 = np.zeros((nLC, ode_net.nVar))
        T0 = np.zeros((nLC, 1))
        x0[0] = np.array([1.1138, 1.8421, 3.1879], dtype=float)
        T0[0] = 1.3027
        x0[1] = np.array([-1.1055, -1.8277, 3.1954], dtype=float)
        T0[1] = 1.3027
    elif r == 22:
        nLC = 2
        x0 = np.zeros((nLC, ode_net.nVar))
        T0 = np.zeros((nLC, 1))
        x0[0] = np.array([10.3266, 13.3565, 20.1329], dtype=float)
        T0[0] = 0.7638
        x0[1] = np.array([-10.3266, -13.3565, 20.1329], dtype=float)
        T0[1] = 0.7638
    else:
        nLC = 0
    nPC = ode_net.nVar
    lcs = np.zeros((nLC, nPC+1), dtype=float)

    for i in np.arange(nLC):
        lcs[i, 0:nPC], lcs[i, nPC] = find_limit_cycle(ode_net, x0[i], T0[i],
                                                      niter=niter, lr=lr)
        print('The start point and poerid of limit cycle is\n',
              f' xlc={lcs[i,0:nPC]}\n',
              f' Tlc={lcs[i, nPC]}')
    return fixed_pts, lcs
