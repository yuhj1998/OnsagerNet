#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
@File 	 : test_ode_error.py
@Time 	 : 2021/08/1
@Author  : Haijn Yu <hyu@lsec.cc.ac.cn>
@Desc	 : test the accuracy of learned ODE system for long time predition
'''

# %% 1. import libs
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import torch
import torch.nn as nn
import torch.nn.functional as F

import ode_net as ode
from scipy.special import binom
import time
import argparse
import matplotlib as mpl

mpl.use('agg', force=True)
plt.switch_backend('agg')
mpl.rc('lines', linewidth=2, markersize=3)

float_formatter = "{:.6e}".format
np.set_printoptions(formatter={'float_kind': float_formatter})

class LangevinNet(ode.ODENet):
    """ A neural network to simulate Langevin dyanmics """

    def __init__(self, gamma, kappa, force=0,
                 gid=0,   # 0=constant, 1=nonlinear resistence
                 kid=0,   # 0=Hookean, 1=Pendulum, 2=double_well
                 ):
        super().__init__()
        self.nVar = 2
        self.gamma = gamma
        self.kappa = kappa
        self.force = force
        self.gid = gid
        self.kid = kid

    def calc_gamma(self, inputs):
        """ calculate the nonlinear resistance coefficients """
        if self.gid == 0:  # constant diffusion
            gamma = self.gamma + 0 * inputs
        if self.gid == 1:  # quadratic diffusion
            gamma = self.gamma * torch.sum(inputs**2, dim=1)
        return gamma

    def calc_Vx(self, inputs):
        """ calculate the force due to potential """
        if self.kid == 0:
            Vx = self.kappa * inputs[:, 0]
        if self.kid == 1:
            c = np.pi * 0.5
            Vx = self.kappa/(c) * torch.sin(c*inputs[:, 0])
        if self.kid == 2:
            Vx = self.kappa * (inputs[:, 0]**3 - 0.5*inputs[:, 0])
        return Vx

    def calc_U(self, inputs):
        """ calculate the physical potential """
        inputs = inputs.view(-1, 2)
        if self.kid == 0:
            U = self.kappa/2 * (inputs[:, 0]**2)
        if self.kid == 1:
            c = np.pi * 0.5
            U = self.kappa/(c**2) * (1 - torch.cos(c*inputs[:, 0]))
        if self.kid == 2:
            U = self.kappa * (inputs[:, 0]**2 - 0.5)**2/4.
        return U

    def calc_potential(self, inputs):
        """ calculate the total energy"""
        U = self.calc_U(inputs)
        V = U + 0.5 * inputs[:, 1]**2
        return V

    def forward(self, inputs):
        """ the inputs is a tensor of size=batch_size x 2 """
        inputs = inputs.view(-1, 2)
        if self.gid == 0:
            gamma = self.gamma
        else:
            gamma = self.calc_gamma(inputs)
        if self.kid == 0:
            Vx = self.kappa * inputs[:, 0]
        else:
            Vx = self.calc_Vx(inputs)
        force = self.force
        output = inputs[:, [1, 0]]
        output[:, 1] = force - gamma*inputs[:, 1] - Vx
        return output

    def gen_sample_trajs(self, nS=1, T=10, dt=0.001, nOut=100,
                         region=[-1.0, 1., -1., 1.],
                         fname='Langevin_samples.txt', noise=0.0):
        x = torch.Tensor(nS).uniform_(region[0], region[1])
        v = torch.Tensor(nS).uniform_(region[2], region[3])
        n = self.nVar
        paths = torch.zeros(nS, 2*n*nOut)
        paths[:, 0], paths[:, 1] = x, v
        with torch.no_grad():
            for i in range(nOut):
                step1 = self.ode_rk3(paths[:, 2*n*i:2*n*i+n], dt, 1)
                paths[:, 2*n*i+n:2*n*i+2*n] = step1
                nt = int(T/nOut/dt)
                step_nextOut = self.ode_rk3(step1, dt, nt)
                if i != nOut-1:
                    paths[:, 2*n*i+2*n:2*n*i+3*n] = step_nextOut
        paths = paths.reshape([-1, n])
        if noise > 1e-8:
            path_noise = paths.uniform_(-1, 1)*paths * noise
            paths += path_noise
        return paths


def get_test_initV(region):
    nS = 12  # number of samples
    n = 2    # dimension
    iv = torch.zeros(nS, n)
    x = torch.linspace(region[0], region[1], 5)
    y = torch.linspace(region[2], region[3], 5)
    iv[:, 0] = torch.cat((x[1:4], x[1:4],
                          torch.tensor([region[0]]*3+[region[1]]*3).float()
                          ), dim=0)
    iv[:, 1] = torch.cat((torch.tensor([region[2]]*3+[region[3]]*3).float(),
                          y[1:4], y[1:4]), dim=0)
    return iv


def long_run_cmp(onet1, onet2, T, dt=0.001, nOut=100,
                 region=[-1., 1., -1., 1.0], savefile='structure'):
    ''' @onet1 : underlying ODE,
        @onet2 : learned ODE
    '''
    # %% 1. setup the initial values to be tested
    iv = get_test_initV(region)
    nS = iv.shape[0]
    n = onet1.nVar
    p1 = torch.zeros(nOut, nS, n)
    p2 = torch.zeros(nOut, nS, n)
    p1[0, :, 0:n] = iv
    p2[0, :, 0:n] = iv

    # %% 2. marhing the initial values and calculate numerical errors
    with torch.no_grad():
        print('Calculating evaluation data ...', end=' ')
        for i in range(nOut-1):
            nt = int(T/nOut/dt)
            p1[i+1, :, :] = onet1.ode_rk3(p1[i, :, :], dt, nt)
            p2[i+1, :, :] = onet2.ode_rk3(p2[i, :, :], dt, nt)
        print('Done.')

    L2err_pts = torch.sum((p1-p2)**2, dim=2).sqrt()
    Linf = torch.max(L2err_pts)
    L2nrm_pth = torch.sqrt(torch.sum(p1**2, dim=[0, 2]) * T/nOut)
    L2err_pth = torch.sqrt(torch.sum((p1-p2)**2, dim=[0, 2])*T/nOut)
    L2err_rel = torch.sqrt(torch.sum((L2err_pth/L2nrm_pth)**2)/nS)
    print(f'The maximum point error for {nS} path is {Linf:.6e}')
    print(f'The average relative L2norm error for {nS} path is {L2err_rel:.6e}')

    # %% 3. Calculate the time volutional error
    time_errors = torch.zeros(nOut, 3)
    time_errors[:, 0] = torch.linspace(0, T, nOut)
    time_errors[:, 1] = torch.mean(L2err_pts, dim=1)
    time_errors[:, 2] = torch.std(L2err_pts, dim=1)
    errfile = savefile+'_err_meanstd.txt'
    np.savetxt(errfile, time_errors, delimiter=', ', fmt='%.3e')

    # %% 4. Plot the trajectory figures
    f = plt.figure(figsize=[10, 5], dpi=144)
    dt_out = T/nOut
    ax = f.add_subplot(221)
    tt = np.arange(nOut)*dt_out
    ip = (L2err_pth/L2nrm_pth).argmax()
    plt.plot(tt, p1[:, ip, 0], 'o', label=r'$x$ original ODE', zorder=0,
             markersize=2, alpha=0.8, color='C3')
    plt.plot(tt, p1[:, ip, 1], 'd', label=r'$v$ original ODE', zorder=0,
             markersize=2, alpha=0.8, color='C2')
    plt.plot(tt, p2[:, ip, 0], label=r'$x$ learned ODE', color='C0', zorder=2)
    plt.plot(tt, p2[:, ip, 1], '--',
             label=r'$v$ learned ODE', color='C1', zorder=2)
    ax.set_title(r'Trajectory with max error')
    # plt.tick_params(axis='y', which='both', labelleft='on', labelright='on')
    plt.grid(True, which='major')
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.xlabel(r'$t$')
    plt.legend(loc=0, ncol=2)

    ax = f.add_subplot(223)
    plt.plot(tt, p1[:, ip, 0]-p2[:, ip, 0], label=r'$x$ error')
    plt.plot(tt, p1[:, ip, 1]-p2[:, ip, 1], '--', label=r'$v$ error')
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.grid(True, which='major')
    plt.xlabel(r'$t$')
    plt.legend(loc=0, ncol=2)

    ax = f.add_subplot(122)  # plot multiple paths
    ipmax = ip
    for ip in np.arange(nS):
        if ip == ipmax:
            plt.plot(p1[:, ip, 0], p1[:, ip, 1], 'o', markersize=2, color='C3',
                     alpha=0.9, zorder=5)
            plt.plot(p2[:, ip, 0], p2[:, ip, 1],
                     linewidth=1.5, color='C0', alpha=0.8, zorder=4)
        else:
            plt.plot(p1[:, ip, 0], p1[:, ip, 1], '-o', color='grey', markersize=4,
                     linewidth=0.5, alpha=0.2, zorder=1)
            plt.plot(p2[:, ip, 0], p2[:, ip, 1],
                     linewidth=0.5, color='C0', alpha=0.9)
    ax.set_xlim(region[0:2])
    ax.set_ylim(region[2:4])
    plt.xlabel('x')
    plt.ylabel('v')

    plt.tight_layout()
    plt.savefig(savefile+'.pdf', bbox_inches='tight', dpi=288)


if __name__ == '__main__':
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
    parser.add_argument('-o', '--onet',  type=str,
                        choices=['ons', 'ode', 'sym'],
                        default='ons', metavar='onet',
                        help='type of the ODE net (default ons)')
    parser.add_argument('--nL',  type=int,
                        default=1, metavar='nHiddenLayers',
                        help='number of hidden layers')
    parser.add_argument('-f', '--fid', type=int, default=0,
                        metavar='FID',
                        help='the id of activation function')
    parser.add_argument('--nHnodes',  type=int,
                        default=12, metavar='nHnodes',
                        help='number of nodes in each hidden layers')
    parser.add_argument('--seed', type=int, default=0, metavar='SEED',
                        help='The first SEED to test the performance')
    parser.add_argument('-p', '--print_net', action='store_true',
                        help='flag to print the coefficients of ODE net')
    
    args = parser.parse_args()
    print(args)

gamma = args.gamma
kappa = args.kappa
gid = args.gid
kid = args.kid

ode.fid = args.fid
onet = args.onet
nHnodes = args.nHnodes
nL = args.nL
seed = args.seed

LgNet = LangevinNet(gamma, kappa, force=0, gid=gid, kid=kid)

nOut = 100
Tforcast = 25
dt = 0.001

# %% 2. Load hidden variables and the learned ODE model
if args.onet == 'ode':
    ode_nodes = np.array([2, ]+[nHnodes, ]*(nL+1), dtype=int)
    ONet = ode.ODENet(ode_nodes)
elif args.onet == 'sym':
    ode_nodes = np.array([2, ]+[nHnodes, ]*nL, dtype=int)
    ONet = ode.SymODEN(ode_nodes, forcing=False,
                       pot_beta=0.0, ons_min_d=0.0)
elif args.onet == 'ons':
    ode_nodes = np.array([2, ]+[nHnodes, ]*nL, dtype=int)
    ONet = ode.OnsagerNet(ode_nodes, forcing=False,
                          pot_beta=0.0, ons_min_d=0.0)
else:
    print(f'ERRROR! network {args.onet} is not implemented!')


basefile = (f'results/Langevin_k{kid}_{int(kappa)}_g{gid}_{int(gamma)}'
            + f'-{args.onet}_f{args.fid}_s{args.seed}')
ode_dict_file = basefile + '_model_dict.pth'
ONet.load_state_dict(torch.load(ode_dict_file))

print(f'Loading {ode_dict_file} ...,  Done.')

long_run_cmp(LgNet, ONet, T=Tforcast, dt=dt, nOut=nOut, savefile=basefile)
