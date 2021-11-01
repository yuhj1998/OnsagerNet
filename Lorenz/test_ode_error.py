#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
@File 	 : test_ode_error.py
@Time 	 : 2021/08/1
@Desc	 : test the accuracy of learned Lorenz63 system for long time predition
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

float_formatter = "{:.6e}".format
np.set_printoptions(formatter={'float_kind': float_formatter})


class LorenzNet(ode.ODENet):
    """ A neural network to simulate Lorenz dyanmics """

    def __init__(self, r, sigma=10, b=8.0/3):
        super().__init__()
        self.nVar = 3
        self.sigma = sigma
        self.b = b
        self.r = r

    def calc_potential(self, inputs):
        inputs = inputs.view(-1, self.nVar)
        pot = torch.sum(inputs**2, dim=1)/2.0
        return pot

    def forward(self, inputs):
        """ the inputs is a tensor of size=batch_size x 2 """
        inputs = inputs.view(-1, self.nVar)
        x, y, z = inputs[:, 0], inputs[:, 1], inputs[:, 2]
        output = torch.zeros_like(inputs)
        s = self.sigma
        r = self.r
        b = self.b
        output[:, 0] = s * (y-x)
        output[:, 1] = x * (r-z) - y
        output[:, 2] = x*y - b*z
        return output

    def gen_sample_paths(self, nS=1, T=10, dt=0.001, nOut=100,
                         region=[-25., 25, -25, 25, -5, 30],
                         fname='Lorenz_samples.txt'):
        n = self.nVar
        paths = torch.zeros(nS, 2*self.nVar*nOut)
        for i in np.arange(n):
            paths[:, i] = torch.Tensor(nS).uniform_(region[2*i], region[2*i+1])
        with torch.no_grad():
            for i in range(nOut):
                di = 2 * n * i
                step1 = self.ode_rk3(paths[:, di:di+n], dt, 1)
                paths[:, di+n:di+2*n] = step1
                nt = int(T/nOut/dt)-1
                step_nextOut = self.ode_rk3(step1, dt, nt)
                if i != nOut-1:
                    paths[:, di+2*n:di+3*n] = step_nextOut
        paths = paths.reshape([-1, n])
        return paths


def get_test_initV(region):
    nS = 12  # number of samples
    n = 3    # dimension
    iv = torch.zeros(nS, n)
    iv[:, 0] = torch.linspace(region[0], region[1], nS)
    iv[:, 1] = torch.linspace(region[2], region[3], nS)
    iv[:, 2] = torch.tensor(region[4]).float()
    return iv


def long_run_cmp(onet1, onet2, T, dt=0.001, nOut=100,
                 region=[-10., 10., -10., 10.0, 0, 1], savefile='tpred'):
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
    print(f'Maximum point error for {nS} path is {Linf:.6e}')
    print(f'Average L2rel error for {nS} path is {L2err_rel:.6e}')

    # %% 3. Calculate the time volutional error
    time_errors = torch.zeros(nOut, 3)
    time_errors[:, 0] = torch.linspace(0, T, nOut)
    time_errors[:, 1] = torch.mean(L2err_pts, dim=1)
    time_errors[:, 2] = torch.std(L2err_pts, dim=1)
    errfile = savefile+'_err_meanstd.txt'
    np.savetxt(errfile, time_errors, delimiter=', ', fmt='%.3e')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Test long time prediction for learned Lorenz ODE')
    parser.add_argument('-r', type=float, default=16,
                        help='scaled Rayleigh number')
    parser.add_argument('--sigma', type=float, default=10,
                        help='Prandtl number (default: 10)')
    parser.add_argument('-o', '--onet',  type=str,
                        choices=['ons', 'ode', 'sym'],
                        default='ons', metavar='onet',
                        help='type of the ODE net (default ons)')
    parser.add_argument('-L', '--nL',  type=int,
                        default=1, metavar='nHiddenLayers',
                        help='number of hidden layers')
    parser.add_argument('-f', '--fid', type=int, default=0,
                        metavar='FID',
                        help='the id of activation function')
    parser.add_argument('-n', '--nHnodes',  type=int,
                        default=12, metavar='nHnodes',
                        help='number of nodes in each hidden layers')
    parser.add_argument('-s', '--seed', type=int, default=0, metavar='SEED',
                        help='The first SEED to test the performance')
    parser.add_argument('-p', '--print_net', action='store_true',
                        help='flag to print the coefficients of ODE net')

    args = parser.parse_args()
    print(args)

sigma = args.sigma
b = 8.0/3
r = args.r

ode.fid = args.fid
onet = args.onet
nHnodes = args.nHnodes
nL = args.nL
seed = args.seed

nPC = 3
nOut = 50
Tforcast = 25
dt = 0.001

LzNet = LorenzNet(r=r, sigma=sigma, b=b)


# %% 2. Load hidden variables and the learned ODE model
if args.onet == 'ode':
    ode_nodes = np.array([nPC, ]+[nHnodes, ]*(nL+1), dtype=int)
    ONet = ode.ODENet(ode_nodes)
elif args.onet == 'ons':
    ode_nodes = np.array([nPC, ]+[nHnodes, ]*nL, dtype=int)
    ONet = ode.OnsagerNet(ode_nodes, forcing=True,
                          pot_beta=0.1, ons_min_d=0.1)
else:
    print(f'ERRROR! network {args.onet} is not implemented!')


basefile = f'results/Lorenz_r{int(r)}_{args.onet}_f{args.fid}_s{seed}'
ode_dict_file = basefile + '_model_dict.pth'
ONet.load_state_dict(torch.load(ode_dict_file))

print(f'Loading {ode_dict_file} ...,  Done.')

long_run_cmp(LzNet, ONet, T=Tforcast, dt=dt, nOut=nOut, savefile=basefile)
