#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
@File 	 : test_ode_Lorenz.py
@Time 	 : 2020/07/21
@Author  : Haijn Yu <hyu@lsec.cc.ac.cn>
         : Xinyan Tian <txy@lsec.cc.ac.cn>
@Desc	 : Learn Lorenz dynamics using OnsagerNet
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

class LorenzNet(onet.ODENet):
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


def long_run_cmp(onet1, onet2, T=5, dt=0.001, nOut=100,
                 region=[-10.0, 10, -10, 10, 0, 1], savefile='run_cmp'):
    ''' @onet1 : underlying ODE,
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

    f = plt.figure(figsize=[12, 10], dpi=144)
    dt_out = T/nOut
    ax = f.add_subplot(311)
    nErrOut = nOut
    ii = np.arange(nErrOut)
    tt = ii*dt_out
    ipp = L2err_pth.argmax()
    plt.plot(tt, p2[ii, ipp, 0], label='x learned ODE')
    plt.plot(tt, p1[ii, ipp, 0], '.', markersize=2, zorder=3,
             alpha=0.8, label='x original ODE')
    plt.plot(tt, p2[ii, ipp, 1], label='y learned ODE')
    plt.plot(tt, p1[ii, ipp, 1], '.', markersize=2, zorder=3,
             alpha=0.8, label='y original ODE')
    plt.plot(tt, p2[ii, ipp, 2], label='z learned ODE')
    plt.plot(tt, p1[ii, ipp, 2], '.', markersize=2, zorder=3,
             alpha=0.8, label='z original ODE')
    ax.set_title('Trajectory with max error')
    plt.tick_params(axis='y', which='both', labelleft='on', labelright='on')
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.xlabel('t')
    plt.legend(fontsize='small', ncol=3, loc="best")

    ax = f.add_subplot(312)
    plt.plot(tt, p1[ii, ipp, 0]-p2[ii, ipp, 0], label='x error')
    plt.plot(tt, p1[ii, ipp, 1]-p2[ii, ipp, 1], label='y error')
    plt.plot(tt, p1[ii, ipp, 2]-p2[ii, ipp, 2], label='z error')
    plt.tick_params(axis='y', which='both', labelleft='on', labelright='on')
    plt.xlabel('t')
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.legend(fontsize='small', loc=0, ncol=3)

    ax = f.add_subplot(337)  # plot multiple paths
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
    plt.xlabel('x')
    plt.ylabel('y')

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
    plt.xlabel('x')
    plt.ylabel('z')

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
    plt.xlabel('y')
    plt.ylabel('z')

    plt.savefig(savefile+'_eval.pdf', bbox_inches='tight', dpi=288)


# %% 3. run the main script
if __name__ == '__main__':
    # %% 1. Generate sample the plot the results
    parser = argparse.ArgumentParser(description='Lorentz example')
    parser.add_argument('-r', type=float, default=16,
                        help='scaled Rayleigh number')
    parser.add_argument('-s', '--sigma', type=float, default=10,
                        help='Prandtl number (default: 10)')
    parser.add_argument('-f', '--fid', type=int, default=0,
                        metavar='FID',
                        help='the id of activation function')
    parser.add_argument('--onet',  type=str,
                        choices=['ons', 'ode'],
                        default='ons', metavar='ONET',
                        help='input name of the ODE net (default ons)')
    parser.add_argument('-L', '--nL', type=int, default=1,
                        metavar='nL',
                        help='number of hidden layers')
    parser.add_argument('-n', '--nHnodes', type=int, default=20,
                        metavar='nHnodes',
                        help='number of nodes in hidden layer')
    parser.add_argument('-ig', type=float, default=0.1, metavar='IG',
                        help='gain used to initialize the network')
    parser.add_argument('-bs', type=int, default=200, metavar='BS',
                        help='batch size')
    parser.add_argument('-lr', type=float, default=0.0256, metavar='LR',
                        help='learning rate')
    parser.add_argument('--patience', type=int, default=25, metavar='PAT',
                        help='patience to reduce lr (default 25)')
    parser.add_argument('--epochs', type=int, default=200, metavar='epochs',
                        help='epochs (default:200 for quick testing)')
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

    sigma = args.sigma
    b = 8.0/3
    r = args.r

    onet.fid = args.fid
    nL = args.nL
    nHnodes = args.nHnodes
    init_gain = args.ig

    nPC = 3
    nTraj = 100
    nOut = 100
    T = 3
    dt = 0.001
    tr_ratio = 8/10

    batch_size = args.bs
    lr = args.lr
    epochs = args.epochs
    patience = args.patience
    st_seed = args.seed
    nseeds = args.nseeds

    for iseed in np.arange(nseeds):
        np.random.seed(st_seed+iseed)
        torch.manual_seed(st_seed+iseed)

        LzNet = LorenzNet(r=r, sigma=sigma, b=b)
        paths = LzNet.gen_sample_paths(nTraj, dt=dt, T=T, nOut=nOut)
        if args.verbose > 0:
            datafile = f'results/Lorenz_r{int(r)}_s{iseed+st_seed}'
            np.savetxt(datafile+'_samples.txt.gz', paths,
                       delimiter=', ', fmt='%.9e')
            viz.plot_nTraj3d_scatter(paths[::2, :].numpy(), datafile+'_samples',
                                     nPath=nTraj, azim=-60, elev=25)

        # %% 2. train OnsagerNet
        if args.onet == 'ode':
            ode_nodes = np.array([nPC,]+ [nHnodes]*(nL+1))
            ONet = onet.ODENet(ode_nodes, ResNet=True, init_gain=init_gain)
        elif args.onet == 'ons':
            ode_nodes = np.array([nPC,]+ [nHnodes]*nL)
            ONet = onet.OnsagerNet(ode_nodes, forcing=True,
                                   init_gain=init_gain,
                                   pot_beta=0.1,
                                   ons_min_d=0.1)
        else:
            print(f'ERRROR! network {args.onet} is not implemented!')

        optimizer, scheduler = onet.get_opt_sch(ONet, lr=lr, weight_decay=0,
                                                 patience=patience)
        datal_train, data_test = viz.data_to_loaders(paths,
                                                     tr_ratio=tr_ratio,
                                                     batch_size=batch_size, ns=0)
        log = ONet.train_ode(optimizer, datal_train,
                             epochs, data_test, scheduler, nt=1, dt=dt)
        print(f'>>Lorenz: r={r}',
              f'onet={args.onet} ',
              f'fname={onet.fnames[onet.fid]} ',
              f'fid={onet.fid} ',
              f'nnodes={nHnodes} ',
              f'nDoF={ONet.size()} ',
              f'gain={init_gain} seed={iseed+st_seed} ',
              f'train_loss={log[-1, 1]:.3e} ',
              f'test_loss={log[-1, 2]:.3e}')

        if args.verbose > 0:
            datafile = f'results/Lorenz_r{int(r)}_{args.onet}_f{args.fid}_s{iseed+st_seed}'
            torch.save(ONet.state_dict(), datafile + '_model_dict.pth')
            viz.plot_ode_train_log(log, datafile)
        if args.verbose > 1:
            Ttest = 3 * ( 3 + int(args.r/4) )
            long_run_cmp(LzNet, ONet, Ttest, savefile=datafile)

        if args.print:
            ONet.print()
