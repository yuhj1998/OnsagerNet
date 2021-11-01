#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
@File 	 : test_ode_Langevin.py
@Time 	 : 2020/06/17
@Author  :  Haijn Yu <hyu@lsec.cc.ac.cn>
         : Xinyan Tian <txy@lsec.cc.ac.cn>
@Desc	 : Learn Langevin dynamics using OnsagerNet and other ODE Nets
'''
# %% 1. import library and set parameters
import numpy as np
import torch
import matplotlib.pyplot as plt
import argparse
import matplotlib as mpl
import ode_net as onet
import viztools as viz

mpl.use('agg', force=True)
plt.switch_backend('agg')
mpl.rc('lines', linewidth=2, markersize=3)


class LangevinNet(onet.ODENet):
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


def long_run_cmp(onet1, onet2, T=10, dt=0.001, nOut=100,
                 region=[-1., 1., -1., 1.0], savefile='run_cmp'):
    ''' @onet1 : underlying ODE,
        @onet2 : learned ODE
    '''
    nS = 12
    n = onet1.nVar
    p1 = torch.zeros(nOut, nS, n)
    p2 = torch.zeros(nOut, nS, n)
    x = torch.linspace(region[0], region[1], 5)
    y = torch.linspace(region[2], region[3], 5)
    p1[0, :, 0] = torch.cat((x[1:4], x[1:4],
                             torch.tensor([region[0]]*3+[region[1]]*3).float()
                             ), dim=0)
    p1[0, :, 1] = torch.cat((torch.tensor([region[2]]*3+[region[3]]*3).float(),
                             y[1:4], y[1:4]), dim=0)
    p2[0, :, 0:n] = p1[0, :, 0:n]
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
    print(f'The average L2norm error for {nS} path is {L2err_rel:.6e}')

    f = plt.figure(figsize=[12, 6], dpi=144)
    dt_out = T/nOut
    ax = f.add_subplot(221)
    tt = np.arange(nOut)*dt_out
    ip = (L2err_pth/L2nrm_pth).argmax()
    plt.plot(tt, p2[:, ip, 0], label='x learned ODE', color='C0', alpha=0.9)
    plt.plot(tt, p2[:, ip, 1], '--',
             label='v learned ODE', color='C1', alpha=0.9)
    plt.plot(tt, p1[:, ip, 0], 'o', label='x original ODE',
             color='C3', zorder=3, alpha=0.7)
    plt.plot(tt, p1[:, ip, 1], 'd', label='v original ODE',
             color='C2', zorder=3, alpha=0.7)
    ax.set_title(r'Trajectory with max error')
    # plt.tick_params(axis='y', which='both', labelleft='on', labelright='on')
    plt.grid(True, which='major')
    plt.xlabel('t')
    plt.legend(loc=0, ncol=2)

    ax = f.add_subplot(223)
    plt.plot(tt, p1[:, ip, 0]-p2[:, ip, 0], label='x error', alpha=0.9)
    plt.plot(tt, p1[:, ip, 1]-p2[:, ip, 1], '--', label='v error', alpha=0.9)
    # ax.yaxis.set_ticks_position('both')
    plt.xlabel('t')
    plt.legend(loc=0, ncol=2)

    ax = f.add_subplot(122)  # plot multiple paths
    for ip in np.arange(nS):
        plt.plot(p1[:, ip, 0], p1[:, ip, 1], 'o', color='C3',
                 alpha=0.6, zorder=3)
        plt.plot(p1[:, ip, 0], p1[:, ip, 1], color='grey',
                 linewidth=0.5, alpha=0.2, zorder=1)
        plt.plot(p2[:, ip, 0], p2[:, ip, 1],
                 linewidth=1.5, color='C0', alpha=0.9)
    ax.set_xlim(region[0:2])
    ax.set_ylim(region[2:4])
    plt.xlabel('x')
    plt.ylabel('v')

    plt.tight_layout()
    plt.savefig(savefile+'_eval.pdf', bbox_inches='tight', dpi=288)
    if mpl.get_backend() in mpl.rcsetup.interactive_bk:
        plt.draw()
        plt.pause(2)
    plt.close()


def show_potential_double_well(odelg, ons, savefile='results/Langevin'):
    nx = ny = 100
    x = np.linspace(-1, 1, nx)
    v = np.linspace(-1, 1, ny)
    xx, vv = np.meshgrid(x, v)
    inputs = np.stack([xx, vv], axis=-1).reshape([-1, 2])
    inputs = torch.FloatTensor(inputs)
    with torch.no_grad():
        Potex = odelg.calc_potential(inputs)
    Potex = Potex.detach().numpy()
    PotExMax = np.max(Potex)
    PotExMin = np.min(Potex)
    Potex = Potex.reshape([nx, ny])

    with torch.no_grad():
        Pot = ons.calc_potential(inputs)
    Pot = Pot.numpy()
    PotMax = np.max(Pot)
    PotMin = np.min(Pot)
    Pot = Pot.reshape([nx, ny])
    G0, H0, V0 = ons.calc_potHessian(torch.FloatTensor([0, 0]))
    print('The grident, Hessian, and potential at (0,0) are:')
    print(f'G0={G0}, H0={H0}, V0={V0}')

    V0 = V0.flatten().numpy()
    G0 = torch.squeeze(G0)
    H0 = torch.squeeze(H0)

    clev = [0.04, 0.16, 0.36, 0.64, 1, 1.44]
    PotMin = np.min(Pot)
    viz.plot_field2d(xx, vv, Pot-PotMin, Potex, clev,
                     savefile=savefile+'_potential')
    Hex = torch.FloatTensor([[odelg.kappa, 0], [0, 1]])
    with torch.no_grad():
        PotP = ons.calc_potential(inputs)
    PotP = PotP.numpy().reshape([nx, ny])
    PotP = PotExMin + (PotP-PotMin)/(PotMax-PotMin) * (PotExMax-PotExMin)
    pot_err = PotP-Potex
    pot_errMax = np.max(np.abs(pot_err))
    dx = dy = (np.max(x) - np.min(x)) / len(x)
    pot_errL2 = np.sqrt(np.sum(pot_err**2) * dx*dy)
    potL2 = np.sqrt(np.sum(Potex**2) * dx*dy)
    print(f'The maximum potential error after alignment is {pot_errMax}!')
    print(
        f'\t >> The relative L^2 potential error after alignment is {pot_errL2/potL2}!')

    viz.plot_3field2d(xx, vv, Pot, PotP, Potex, clev,
                      savefile=savefile+'_potentials')

    viz.plot_field2d(xx, vv, pot_err, savefile=savefile+'_pot_err',
                     stitle='Potential error after alignment',
                     xylabel=('x', 'v'))


def show_potential(odelg, ons, savefile='results/Langevin'):
    nx = ny = 100
    x = np.linspace(-1, 1, nx)
    v = np.linspace(-1, 1, ny)
    xx, vv = np.meshgrid(x, v)
    inputs = np.stack([xx, vv], axis=-1).reshape([-1, 2])
    inputs = torch.FloatTensor(inputs)
    with torch.no_grad():
        Potex = odelg.calc_potential(inputs)
    Potex = Potex.detach().numpy().reshape([nx, ny])

    with torch.no_grad():
        Pot = ons.calc_potential(inputs)
    Pot = Pot.numpy().reshape([nx, ny])
    G0, H0, V0 = ons.calc_potHessian(torch.FloatTensor([0, 0]))
    print('The gradient, Hessian, and potential at (0,0) are:')
    print(f'G0={G0}, H0={H0}, V0={V0}')

    V0 = V0.flatten().numpy()
    G0 = torch.squeeze(G0)
    H0 = torch.squeeze(H0)

    clev = [0.04, 0.16, 0.36, 0.64, 1, 1.44]
    Hex = torch.FloatTensor([[odelg.kappa, 0], [0, 1]])
    Lex = torch.cholesky(Hex)
    Linv = torch.inverse(torch.cholesky(torch.FloatTensor(H0)))
    xvPt = torch.mm(torch.mm(inputs, Lex), Linv)
    with torch.no_grad():
        PotP = ons.calc_potential(xvPt)
    PotP = PotP.numpy().reshape([nx, ny])
    pot_err = PotP-V0-Potex
    pot_errMax = np.max(np.abs(pot_err))
    dx = dy = (np.max(x) - np.min(x)) / len(x)
    pot_errL2 = np.sqrt(np.sum(pot_err**2) * dx*dy)
    potL2 = np.sqrt(np.sum(Potex**2) * dx*dy)
    print(f'The maximum potential error after alignment is {pot_errMax}!')
    print(
        f'\t >>The relative L^2 potential error after alignment is {pot_errL2/potL2}!')

    viz.plot_3field2d_new(xx, vv, Pot-V0, PotP-V0, Potex, clev,
                          savefile=savefile+'_potentials')

    viz.plot_field2d(xx, vv, pot_err, savefile=savefile+'_pot_err',
                     stitle='Potential error after alignment',
                     xylabel=('x', 'v'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Learning Langevin dynamics')
    parser.add_argument('-gamma', type=float, default=1, metavar='g',
                        help='input diffusion constant gamma (default: 1.0)')
    parser.add_argument('-kappa', type=float, default=4, metavar='k',
                        help='input elastic constant kappa (default: 4.0)')
    parser.add_argument('-gid', type=int, default=0, metavar='gid',
                        help='input diffusion type (0=const, 1=nonlinear)')
    parser.add_argument('-kid', type=int, default=0, metavar='kid',
                        help='input potential type (0=Hookean, 1=Pendulum, 2=DoubleWell)')
    parser.add_argument('-fid', type=int, default=2, metavar='FID',
                        help='the id of activation function (default 2=ReQU)')
    parser.add_argument('--nL',  type=int,
                        default=1, metavar='nHiddenLayers',
                        help='number of hidden layers')
    parser.add_argument('-n', '--nHnodes', type=int, default=12,
                        metavar='nHnodes',
                        help='number of nodes in hidden layer')
    parser.add_argument('--onet',  type=str,
                        choices=['ons', 'ode', 'sym'],
                        default='ons', metavar='onet',
                        help='input name of the ODE net (default ons)')
    parser.add_argument('-ig', type=float, default=0.1, metavar='IG',
                        help='gain used to initialize the network')
    parser.add_argument('-bs', type=int, default=200, metavar='BS',
                        help='batch size (default 200)')
    parser.add_argument('-lr', type=float, default=0.0128, metavar='LR',
                        help='learning rate (default: 0.0128)')
    parser.add_argument('--patience', type=int, default=25, metavar='PAT',
                        help='patience to reduce lr (default 25)')
    parser.add_argument('--epochs', type=int, default=100, metavar='epoch',
                        help='epochs (default:1000, <100 for testing)')
    parser.add_argument('--seed', type=int, default=0, metavar='SEED',
                        help='The first SEED to test the performance')
    parser.add_argument('--nseeds', type=int, default=1, metavar='NSEEDs',
                        help='number of seeds(runs) to test the performance')
    parser.add_argument('-p', '--print', default=False, action='store_true',
                        help='print detailed testing for path, potential etc.')
    parser.add_argument('-v', '--verbose', type=int, default=1,
                        metavar='Verbose',
                        help='Verbose level v=0: no data and figures to disk')

    args = parser.parse_args()
    print(args)

    gamma = args.gamma
    kappa = args.kappa
    gid = args.gid
    kid = args.kid

    onet.fid = args.fid
    onet.ode_sparsity_alp = 0e-7
    nHnodes = args.nHnodes
    nL = args.nL
    init_gain = args.ig

    epochs = args.epochs
    batch_size = args.bs
    tr_ratio = 8/10
    lr = args.lr
    patience = args.patience
    nseeds = args.nseeds
    st_seed = args.seed

    if kid == 2:
        nTraj = 1000
        nOut = 40
        T = 1
    else:
        nTraj = 100
        nOut = 100
        T = 5

    dt = 0.001
    for iseed in np.arange(nseeds):
        np.random.seed(st_seed+iseed)
        torch.manual_seed(st_seed+iseed)

        odelg = LangevinNet(gamma, kappa, gid=gid, kid=kid)
        trajs = odelg.gen_sample_trajs(nTraj, T, dt, nOut)
        if args.verbose > 0:
            datafile = f'results/Langevin_k{kid}_{kappa:1.0f}_g{gid}_{gamma:1.0f}'
            np.savetxt(datafile+'_samples.txt.gz', trajs,
                       delimiter=', ', fmt='%.9e')
            viz.plot_nTraj2d_quiver(trajs, datafile+'_samples',
                                    xlabel='x', ylabel='v')

        # %% 2. train OnsagerNet
        if args.onet == 'ode':
            ode_nodes = np.array([2, ]+[nHnodes, ]*(nL+1), dtype=int)
            ONet = onet.ODENet(ode_nodes, init_gain=init_gain)
        elif args.onet == 'sym':
            ode_nodes = np.array([2, ]+[nHnodes, ]*nL, dtype=int)
            ONet = onet.SymODEN(ode_nodes, forcing=False,
                                init_gain=init_gain,
                                pot_beta=0.0,
                                ons_min_d=0.0)
        elif args.onet == 'ons':
            ode_nodes = np.array([2, ]+[nHnodes, ]*nL, dtype=int)
            ONet = onet.OnsagerNet(ode_nodes, forcing=False,
                                   init_gain=init_gain,
                                   pot_beta=0.0,
                                   ons_min_d=0.0)
        else:
            print(f'ERRROR! network {args.onet} is not implemented!')

        optimizer, scheduler = onet.get_opt_sch(ONet,
                                                lr=lr, patience=patience)
        datal_train, data_test = viz.data_to_loaders(trajs,
                                                     tr_ratio=tr_ratio,
                                                     batch_size=batch_size)
        log = ONet.train_ode(optimizer, datal_train,
                             epochs, data_test, scheduler, dt=dt)
        print(f'>>Langevin: kid={kid} kappa={kappa} gid={gid} gamma={gamma} ',
              f'net={args.onet} ',
              f'fact={onet.fnames[onet.fid]} ',
              f'nnodes={nHnodes} ',
              f'nDoF={ONet.size()} ',
              f'ig={init_gain} seed={st_seed+iseed} ',
              f'train_loss={log[-1, 1]:.3e} ',
              f'test_loss={log[-1, 2]:.3e}')

        if args.verbose > 0:
            datafile = (f'results/Langevin_k{kid}_{kappa:1.0f}_g{gid}_{gamma:1.0f}' +
                        f'-{args.onet}_f{onet.fid}_s{st_seed+iseed}')
            torch.save(ONet.state_dict(), datafile + '_model_dict.pth')
            viz.plot_ode_train_log(log, datafile)
        if args.verbose > 1:
            Ttest = 2*T if gid == 0 else 4*T
            print('Testing the learned dynamics with exact one, be patient ...')
            long_run_cmp(odelg, ONet, Ttest, savefile=datafile)

        # %% check the learned ODE
        if args.print:
            ONet.print()
            if args.onet == 'ons':
                if kid == 2:
                    show_potential_double_well(odelg, ONet, savefile=datafile)
                else:
                    show_potential(odelg, ONet, savefile=datafile)
