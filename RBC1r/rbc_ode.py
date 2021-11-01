#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""  rbc_ode.py
     Test ODENet (OnsagerNet or plain multi-layer perception net) on RBC PCA data.
     @author: Haijun Yu <hyu@lsec.cc.ac.cn>
"""
# %%
import config as cfgs
import ode_net as ode
import rbctools as rbc
import argparse
from scipy.special import binom
import torch.utils.data as data
import torch
import numpy as np
from torch import manual_seed
from numpy import random
random.seed(1)
manual_seed(2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Learn ODE for Rayleigh-Bernard convection encoded data')
    parser.add_argument('-tc', type=int, default=cfgs.DEFAULT_CASE_ID,
                        metavar='tc',
                        help='id of the test case')
    parser.add_argument('nPC', type=int, nargs='?', default=-1, metavar='nPC',
                        help='number of hidden variables')
    parser.add_argument('--method', type=str, choices=['pca', 'ae'],
                        default='pca', metavar='METHOD',
                        help='input model of dim reduction (default pca)')
    parser.add_argument('--onet',  type=str, choices=['ons', 'ode', 'res'],
                        default='ons', metavar='onet',
                        help='input name of the ODE net (default ons)')
    parser.add_argument('--nHnode',  type=int,
                        default=-1, metavar='nHnode',
                        help='number of nodes in each hidden layers')
    parser.add_argument('--nL',  type=int,
                        default=1, metavar='nHiddenLayers',
                        help='number of hidden layers')
    parser.add_argument('-f', '--fid', type=int, default=0,
                        metavar='FID',
                        help='the id of activation function')
    parser.add_argument('--ig', type=float, default=0.1, metavar='IG',
                        help='gain used to initialize the network')
    parser.add_argument('-e', '--epochs', type=int, default=-1,
                        metavar='epochs',
                        help='number of epochs')
    parser.add_argument('-lr', type=float, default=-0.1, metavar='LR',
                        help='learning rate')
    parser.add_argument('--patience', type=int, default=0, metavar='PAT',
                        help='patience to reduce lr (default 25)')
    parser.add_argument('--seed', type=int, default=0, metavar='SEED',
                        help='The first SEED to test the performance')
    parser.add_argument('--nseeds', type=int, default=1, metavar='NSEEDs',
                        help='number of seeds(runs) to test the performance')
    parser.add_argument('--no_amsgrad', default=False, action='store_true',
                        help='Set Adam parameter amsgrad')
    args = parser.parse_args()
    print(args)

    test_id = args.tc
    cfg = cfgs.get_test_case(test_id)
    outloc = cfg.outloc
    lr_min = 5e-6

    epochs = args.epochs if args.epochs > 0 else cfg.epochs
    nPC = args.nPC if args.nPC > 0 else cfg.nPC
    st_seed = args.seed if args.seed > 0 else cfg.iseed
    nHnode = args.nHnode if args.nHnode > 0 else int(
        cfg.iNodeC * binom(nPC+2, 2))

    amsgrad = not args.no_amsgrad
    ode.fid = args.fid
    init_gain = args.ig
    method = args.method
    onet = args.onet
    nL = args.nL
    nseeds = args.nseeds

    lr = args.lr if args.lr >0 else cfg.lr_ode
    patience = args.patience if args.patience > 0 else cfg.patience

    batch_size = cfg.batch_size
    wt_decay = cfg.wt_decay

    if method == 'pca':
        datfile = cfg.h5fname+f'_{method}{nPC}_enc_data.txt.gz'
        hvar = np.loadtxt(datfile, delimiter=',')
    else:
        datfile = outloc+f'_{method}{nPC}_{onet}_enc_data.txt.gz'
        hvar = np.loadtxt(datfile, delimiter=',')
    print('Data loaded from ', datfile)

    nf = hvar.shape[0]
    nS_train = int(nf//2 * cfg.tr_ratio)
    nS_test = nf//2 - nS_train
    ds1 = torch.FloatTensor(hvar[0:2*nS_train:2, :])
    ds2 = torch.FloatTensor(hvar[1:2*nS_train:2, :])
    dataset_train = data.TensorDataset(ds1, ds2)
    dt1 = torch.FloatTensor(hvar[2*nS_train::2, :])
    dt2 = torch.FloatTensor(hvar[2*nS_train+1::2, :])
    dataset_test = (dt1, dt2)

    # %% Train model
    for iseed in range(nseeds):
        seed = st_seed + iseed
        np.random.seed(seed=seed)
        torch.manual_seed(seed)
        dataloader_train = data.DataLoader(dataset_train,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           num_workers=1)
        ode_nodes = cfg.get_ode_nodes(nPC, nHnode, nL, onet)
        if onet == 'ode':
            ONet = ode.ODENet(ode_nodes, init_gain=init_gain)
        elif onet == 'res':
            ONet = ode.ResODENet(ode_nodes, init_gain=init_gain)
        else:
            ONet = ode.OnsagerNet(ode_nodes, init_gain=init_gain,
                                  pot_beta=cfg.pot_beta,
                                  ons_min_d=cfg.ons_min_d
                                  )
        nHnode = ode_nodes[1]
        print(f'\t Trainable paramters: {ONet.size()}')
        optimizer, scheduler = ode.get_opt_sch(ONet, lr=lr,
                                               weight_decay=wt_decay,
                                               patience=patience,
                                               amsgrad=amsgrad,
                                               lr_min=lr_min, method='Adam',
                                               epoch=epochs)
        log = ONet.train_ode(optimizer,
                             dataloader_train,
                             epochs,
                             dataset_test, scheduler, nt=1, dt=cfg.dt)
        if np.isnan(log).any():
            continue

        train_loss, test_loss = log[-1, 1], log[-1, 2]
        print(f'>>Results: r={cfg.rRa}',
              f'n={nPC:2d}',
              f'net={onet}',
              f'L={nL}',
              f'nH={nHnode}',
              f'nDoF={ONet.size()}',
              f'fid={ode.fid}',
              f'gain={init_gain} seed={seed}',
              f'ode_train={train_loss:.3e}',
              f'ode_test={test_loss:.3e}')

        if np.isnan(log).any():
            print('The fitting for last parameter set failed due to NAN!')
        else:
            savefile = outloc + f'_{method}{nPC}_{onet}_f{ode.fid}_L{nL}_s{seed}'
            torch.save(ONet.state_dict(), savefile+'_model_dict.pth')
            np.savetxt(savefile+'.txt', log, delimiter=', ', fmt='%.3e')
            rbc.plot_ode_train_log(log, savefile)
