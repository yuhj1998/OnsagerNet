#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" rbc_ae_ode.py
    Train the Auto-encoder and ODENet together for
        the Rayleigh-Bernard-Convection problem

    @author: Haijun Yu <hyu@lsec.cc.ac.cn>
"""
# %%
import config as cfgs
import ode_net as ode
import autoencoders as ae
import rbctools as rbc
import matplotlib as mpl
import torch.nn.functional as F
from itertools import chain
from scipy.special import binom
import torch.optim as optim
import torch
import torch.utils.data as data
import matplotlib.pyplot as plt
import argparse
import time
import numpy as np
from torch import manual_seed
from numpy import random
random.seed(1)
manual_seed(2)

a_ae = 0.0
orth_con = 0
amsgrad = True
ode_sparsity_alp = 0
EPS = 1e-7

# %% Define parameters
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Learn ODE for Rayleigh-Bernard convection PDE data')
    parser.add_argument('-tc', type=int, default=cfgs.DEFAULT_CASE_ID,
                        metavar='tc',
                        help='id of the test case')
    parser.add_argument('nPC', type=int, nargs='?', default=-1, metavar='nPC',
                        help='number of hidden variables')
    parser.add_argument('--onet',  type=str, choices=['ons', 'ode', 'res'],
                        default='ons', metavar='onet',
                        help='type of the ODE net (default ons)')
    parser.add_argument('--nL',  type=int,
                        default=1, metavar='nHiddenLayers',
                        help='number of hidden layers')
    parser.add_argument('--nHnode', type=int, default=-1,
                        metavar='nHnode',
                        help='number of nodes in each hidden layers')
    parser.add_argument('--enNet',  type=str, choices=['PCAResNet', 'SAE'],
                        default='PCAResNet', metavar='enNet',
                        help='type of the auto-encoder net (default PCAResNet)')

    parser.add_argument('-e', '--epochs', type=int, default=-1,
                        metavar='epochs',
                        help='number of epochs')
    parser.add_argument('--seed', type=int, default=0, metavar='SEED',
                        help='The first SEED to test the performance')
    parser.add_argument('--nseeds', type=int, default=1, metavar='NSEEDs',
                        help='number of seeds(runs) to test the performance')

    parser.add_argument('--b_ae', type=float, default=0.5, metavar='b_ae')
    parser.add_argument('--a_isom', type=float, default=0.8, metavar='a_isom')
    parser.add_argument('--b_isom', type=float, default=2.0, metavar='b_isom')
    parser.add_argument('--b_cae', type=float, default=0.0, metavar='b_cae')
    parser.add_argument('--e2e', type=float, default=0., metavar='e2e')
    args = parser.parse_args()
    print(args)

# %% 0. Setup parameters
a_isom = args.a_isom
b_isom = args.b_isom
b_ae = args.b_ae
b_cae = args.b_cae
beta_e2e = args.e2e
enNet = args.enNet

test_id = args.tc
cfg = cfgs.get_test_case(test_id)

nPC = args.nPC if args.nPC > 0 else cfg.nPC

nTraj = cfg.nTraj
h5fname = cfg.h5fname
outloc = cfg.outloc

epochs = args.epochs if args.epochs > 0 else cfg.epochs
seed = args.seed if args.seed > 0 else cfg.iseed
nHnode = args.nHnode if args.nHnode > 0 else int(cfg.iNodeC * binom(nPC+2, 2))

dt = cfg.dt
nt = cfg.nt
e2e_epoch = epochs*2//2
onet = args.onet
nL = args.nL
iNodeC = cfg.iNodeC
ode.fid = cfg.ode_fid
nseeds = args.nseeds
wt_decay = cfg.wt_decay

lr_ode = cfg.lr_ode
lr_e2e = cfg.lr_e2e
lr_ode_min = 5e-5
lr_e2e_min = 1e-5
batch_size_MIN = cfg.batch_size
patience_ode = cfg.patience
ode.ode_sparsity_alp = ode_sparsity_alp
patience_e2e = cfg.e2e_patience

torch.manual_seed(seed)
np.random.seed(seed)

# %% 1. Load training data and do a basic PCA
ts_loaddata = time.time()
uvh, coordx, coordy, nx, ny, nf = rbc.load_uvh_data(h5fname+'.h5')
nf = uvh.shape[0]
nOut = nf//nTraj//2
te = (nOut - cfg.te)*2
uvh, coordx, coordy, nx, ny = rbc.downsampling(uvh, coordx, coordy,
                                               ratio=2, ts=2*cfg.ts, te=te,
                                               nRun=nTraj)
te_loaddata = time.time()

nf = uvh.shape[0]
nVar = uvh.shape[1]
wt = 1/np.sqrt(nVar/3.0) * cfg.wt_factor
dataset = uvh[::, :] * wt

PCA = ae.PCA_Encoder(nPC, dataset)

nS_train = int(nf//2 * cfg.tr_ratio)
nS_test = nf//2 - nS_train

ds1 = torch.FloatTensor(dataset[0:2*nS_train:2, :])
ds2 = torch.FloatTensor(dataset[1:2*nS_train:2, :])
dataset_train = data.TensorDataset(ds1, ds2)
with torch.no_grad():
    ds_h1, img1 = PCA(ds1)
    ds_h2, img2 = PCA(ds2)
pca_loss_train = ae.mse_loss(img1, ds1)
pca_loss_train += ae.mse_loss(img2, ds2)
pca_loss_train = pca_loss_train.item()
pca_isom_loss = 1.0/(dt**2) * torch.mean(
    torch.abs(torch.sum((ds_h1-ds_h2)**2, dim=1)
              - torch.sum((ds1-ds2)**2, dim=1)))
pca_isom_loss = pca_isom_loss.item()

dt1 = torch.FloatTensor(dataset[2*nS_train::2, :])
dt2 = torch.FloatTensor(dataset[2*nS_train+1::2, :])
dataset_test = (dt1, dt2)
with torch.no_grad():
    dt_h1, img1 = PCA(dt1)
    dt_h2, img2 = PCA(dt2)
pca_loss_test = ae.mse_loss(img1, dt1)
pca_loss_test += ae.mse_loss(img2, dt2)
pca_loss_test = pca_loss_test.item()

dataset_h_train = data.TensorDataset(ds_h1, ds_h2)
dataset_h_test = (dt_h1, dt_h2)
batch_size = min(nS_train, batch_size_MIN)

print(
    f'Finish loading data {h5fname} in {te_loaddata-ts_loaddata:.3e} seconds.')
print(f'Sample Size = {nVar}')
print(f'Total number of samples = {nf//2}')
print('Number of trainning sample =', nS_train)
print('Number of testing samples =', nS_test)
print(f'pca_loss_train = {pca_loss_train:.3e}')
print(f'pca_loss_test = {pca_loss_test:.3e}')
print(f'pca_isometric_loss = {pca_isom_loss:.3e}')

ae_loss_init = pca_loss_train
ode_loss_init = pca_loss_train * 10

# with torch.no_grad():
#     output_e = PCA.encode(dataset)
#     encode = output_e.reshape([nf, nPC])

ode_nodes = cfg.get_ode_nodes(nPC, nHnode, nL, onet)
if onet == 'ode':
    ONet = ode.ODENet(ode_nodes)
else:
    ONet = ode.OnsagerNet(ode_nodes, pot_beta=cfg.pot_beta,
                          ons_min_d=cfg.ons_min_d)

ae_nodes = cfg.get_ae_nodes(nVar, nPC)
if enNet == 'PCAResNet':
    PCANet = ae.PCA_ResNet(ae_nodes, dataset)
else:
    ae.fid = 7  # 0 1 6 7 8 9
    PCANet = ae.SimpleAE(ae_nodes)

# print summary info
print('Number of nodes in AE layers:', ae_nodes)
print(f'Number of nodes in ODENet({onet}) layers:', ode_nodes)
print(f'Number of trainable parameters: {PCANet.size()}+{ONet.size()}')
print(f'nPC={nPC}, iNodeC={iNodeC}, Epochs={epochs}, batch_size={batch_size}')
print(f'learning rate ode= {lr_ode}')
print(f'learning rate e2e= {lr_e2e}')
print(f'patience_ode={patience_ode}, patience_e2e={patience_e2e}')
print('-'*80)


def train_ae(enNet, optimizer, data_train, epochs=1,
             data_test=None, scheduler=None,
             c_isom=0., c_cae=0.,
             pre_train=0):
    ''' pretrain auto-encoder 
        '''
    nS_train = len(data_train.dataset)
    batch_size = data_train.batch_size

    log = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lr_init = optimizer.param_groups[0]["lr"]
    if pre_train > 0:
        optimizer.param_groups[0]["lr"] = 1e-5

    enNet.to(device)
    if data_test is not None:
        dt1, dt2 = data_test
        dt1 = dt1.to(device)
        dt2 = dt2.to(device)

    print(f'Training AE on {device} with c_isom={c_isom}, c_cae={c_cae}:...')
    for e in range(epochs):
        loss_acc = 0
        isom_loss_acc = 0
        ae_loss_acc = 0
        cae_loss_acc = 0
        if e == pre_train:
            optimizer.param_groups[0]["lr"] = lr_init
        for i, (img1, img2) in enumerate(data_train):
            img1 = img1.to(device)
            img2 = img2.to(device)

            img1.requires_grad_(True)
            # img1.retain_grad()
            h1, output1 = enNet(img1)
            h2, output2 = enNet(img2)
            ae_loss_train = ae.mse_loss(output1, img1)
            ae_loss_train += ae.mse_loss(output2, img2)

            isom_loss = 1.0/(dt**2) * torch.mean(
                torch.abs(torch.sum((h1-h2)**2, dim=1)
                          - torch.sum((img1-img2)**2, dim=1)))
            isom_obj = F.relu(isom_loss - a_isom * pca_isom_loss)

            loss = ae_loss_train + c_isom * isom_obj

            for i in np.arange(nPC):
                gv = torch.zeros_like(h1)
                gv[:, i] = 1.0
                g, = torch.autograd.grad(
                    h1, img1, grad_outputs=gv, create_graph=True)
                g = torch.reshape(g, (-1,))
                if i == 0:
                    cae_loss = torch.sum(g**2)/batch_size
                else:
                    cae_loss += torch.sum(g**2)/batch_size
            if b_cae > EPS:  # contractive regularization
                loss += cae_loss * c_cae

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(enNet.parameters(), 1.0)
            optimizer.step()
            loss_acc += loss.item()
            isom_loss_acc += isom_loss.item()
            ae_loss_acc += ae_loss_train.item()
            cae_loss_acc += cae_loss.item()

        loss_acc = (loss_acc*batch_size) / nS_train
        isom_loss_acc = (isom_loss_acc*batch_size) / nS_train
        ae_loss_acc = (ae_loss_acc*batch_size) / nS_train
        cae_loss_acc = (cae_loss_acc*batch_size) / nS_train

        if (scheduler is not None) and e > pre_train:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(loss_acc)
            else:
                scheduler.step()

        if data_test is not None:
            with torch.no_grad():
                h1t, tr1 = enNet(dt1)
                h2t, tr2 = enNet(dt2)
                ae_loss_test = ae.mse_loss(tr1, dt1)
                ae_loss_test += ae.mse_loss(tr2, dt2)

        last_lr = optimizer.param_groups[0]["lr"]
        errs = [e, loss_acc, ae_loss_acc, isom_loss_acc, cae_loss_acc]
        if data_test is not None:
            errs.append(ae_loss_test.item())
        errs.append(last_lr)
        log.append(errs)

        if e % 10 == 0 or e == epochs-1:  # print results and adjust coefficients
            print(f'e:{e:4d}/{epochs}', end=' ')
            print(f'mse: {loss_acc:.3e}', end=' ')
            print(f'ae: {ae_loss_acc:.3e}', end=' ')
            print(f'isom: {isom_loss_acc:.3e}', end=' ')
            print(f'cae: {cae_loss_acc:.2f}', end=' ')
            if data_test is not None:
                print(f'aet: {ae_loss_test.item():.3e}', end=' ')
            print(f'lr:{last_lr:.3e}', flush=True)
            # c_isom = b_isom * ae_loss_acc / isom_loss_acc
            # c_cae = b_cae * ae_loss_acc / cae_loss_acc
    log = np.squeeze(np.array(log))
    print(f'isom_loss={isom_loss_acc:.3e} ',
          f'pca_isom_loss={pca_isom_loss:.3e}',
          f'c_isom={c_isom:.3e}',
          f'c_cae={c_cae:.3e}',
          f'cae_loss={cae_loss_acc:.3e}')
    return log


def train_all(enNet, odeNet, optimizer, data_train, epochs=1,
              data_test=None, scheduler=None,
              ode_only=False, pre_train=0):
    ''' train auto-encoder and ODENet together
        '''
    nS_train = len(data_train.dataset)
    batch_size = data_train.batch_size

    log = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Training AutoEncoder and NerualODE together on {device}: ...')
    lr_init = optimizer.param_groups[0]["lr"]
    if pre_train > 0:
        optimizer.param_groups[0]["lr"] = 1e-5

    enNet.to(device)
    odeNet.to(device)

    c_ae = b_ae
    c_isom = b_isom * ode_loss_init / pca_isom_loss
    c_cae = b_cae
    print(
        f'Train AE+OnsNet with c_ae={c_ae}, c_isom={c_isom}, c_cae={c_cae}, e2e={beta_e2e}:...')
    if data_test is not None:
        dt1, dt2 = data_test
        dt1 = dt1.to(device)
        dt2 = dt2.to(device)
    logheader = ('  epoch   tot_loss  ae_loss   ode_loss  e2e_loss  ' +
                 'cae_loss   ae_test  ode_test     lr')
    for e in range(epochs):
        loss_acc = 0
        isom_loss_acc = 0
        ode_loss_acc = 0
        e2e_loss_acc = 0
        ae_loss_acc = 0
        cae_loss_acc = 0
        if e == pre_train:
            optimizer.param_groups[0]["lr"] = lr_init
        for i, (img1, img2) in enumerate(data_train):
            img1 = img1.to(device)
            img2 = img2.to(device)

            img1.requires_grad_(True)
            img1.retain_grad()
            h1, output1 = enNet(img1)
            h2, output2 = enNet(img2)
            ae_loss_train = ae.mse_loss(output1, img1)
            ae_loss_train += ae.mse_loss(output2, img2)
            ae_obj = F.relu(ae_loss_train - a_ae * pca_loss_train)

            h1.requires_grad_(True)
            h2_ode = odeNet.ode_rk2(h1, dt/nt, nt)
            h2_img = enNet.decode(h2_ode)
            e2e_loss = ae.mse_loss(h2_img, img2)
            ode_loss_train = ode.mse_p_loss(h2_ode, h2, p=2, dt=dt)
            if ode_sparsity_alp > 0:
                ode_loss_train += ode_sparsity_alp * odeNet.sparsity_loss()

            isom_loss = 1.0/(dt**2) * torch.mean(
                torch.abs(torch.sum((h1-h2)**2, dim=1)
                          - torch.sum((img1-img2)**2, dim=1)))
            isom_obj = F.relu(isom_loss - a_isom * pca_isom_loss)
            loss = ode_loss_train + (c_ae * ae_obj + c_isom * isom_obj)
            if beta_e2e > EPS:
                loss += beta_e2e * e2e_loss

            for i in np.arange(nPC):
                gv = torch.zeros_like(h1)
                gv[:, i] = 1.0
                g, = torch.autograd.grad(
                    h1, img1, grad_outputs=gv, create_graph=True)
                g = torch.reshape(g, (-1,))
                if i == 0:
                    cae_loss = torch.sum(g**2)/batch_size
                else:
                    cae_loss += torch.sum(g**2)/batch_size
            if b_cae > 1e-10:  # contractive regularization
                loss += c_cae * cae_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(enNet.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(odeNet.parameters(), 1.0)
            optimizer.step()
            loss_acc += loss.item()
            isom_loss_acc += isom_loss.item()
            ode_loss_acc += ode_loss_train.item()
            e2e_loss_acc += e2e_loss.item()
            ae_loss_acc += ae_loss_train.item()
            cae_loss_acc += cae_loss.item()

        loss_acc = (loss_acc*batch_size) / nS_train
        isom_loss_acc = (isom_loss_acc*batch_size) / nS_train
        ode_loss_acc = (ode_loss_acc*batch_size) / nS_train
        e2e_loss_acc = (e2e_loss_acc*batch_size) / nS_train
        ae_loss_acc = (ae_loss_acc*batch_size) / nS_train
        cae_loss_acc = (cae_loss_acc*batch_size) / nS_train

        if (scheduler is not None) and e > pre_train:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(loss_acc)
            else:
                scheduler.step()

        if data_test is not None:
            with torch.no_grad():
                h1t, tr1 = enNet(dt1)
                h2t, tr2 = enNet(dt2)
                ae_loss_test = ae.mse_loss(tr1, dt1)
                ae_loss_test += ae.mse_loss(tr2, dt2)

                h1t.requires_grad_(True)
                h2t_ode = odeNet.ode_rk2(h1t, dt/nt, nt)
                ode_loss_test = ode.mse_p_loss(h2t_ode, h2t, p=2, dt=dt)

        last_lr = optimizer.param_groups[0]["lr"]
        errs = [e, loss_acc, ae_loss_acc, ode_loss_acc]
        if data_test is not None:
            errs.append(ae_loss_test.item())
            errs.append(ode_loss_test.item())
        errs.append(last_lr)
        log.append(errs)

        if e % 10 == 0 or e == epochs-1:
            if e % 100 == 0:
                print(logheader)
            print(f'{e:4d}/{epochs}', end=' ')
            print(f'{loss_acc:.3e}', end=' ')
            print(f'{ae_loss_acc:.3e}', end=' ')
            print(f'{ode_loss_acc:.3e}', end=' ')
            print(f'{e2e_loss_acc:.3e}', end=' ')
            print(f'{cae_loss_acc:.3e}', end=' ')
            if data_test is not None:
                print(f'{ae_loss_test.item():.3e}', end=' ')
                print(f'{ode_loss_test.item():.3e}', end=' ')
            print(f'{last_lr:.2e}', flush=True)
    log = np.squeeze(np.array(log))
    print(f'isom_loss={isom_loss_acc:.3e} ',
          f'pca_isom_loss={pca_isom_loss:.3e}',
          f'cae_loss={cae_loss_acc:.3e}')
    return log


# %% 3. Train AE model first if encode Net is not PCAResNet
dataloader_train = torch.utils.data.DataLoader(dataset_train,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=0)
ts_train1 = time.time()
if enNet != 'PCAResNet':
    print('Train the AE net first ...')
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad,
                                   PCANet.parameters()),
                            lr=lr_ode,
                            amsgrad=amsgrad,
                            weight_decay=wt_decay)
    e = e2e_epoch
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           factor=0.25,
                                                           patience=patience_e2e,
                                                           min_lr=lr_e2e_min)
    log_ae = train_ae(PCANet, optimizer,
                      dataloader_train,
                      e2e_epoch,
                      dataset_test, scheduler,
                      c_isom=b_isom * ae_loss_init / pca_isom_loss,
                      c_cae=b_cae)
    ae_loss_init = log_ae[-1, 2]

    with torch.no_grad():
        ds_h1, img1 = PCANet(ds1)
        ds_h2, img2 = PCANet(ds2)
    dataset_h_train = data.TensorDataset(ds_h1, ds_h2)

    with torch.no_grad():
        dt_h1, img1 = PCANet(dt1)
        dt_h2, img2 = PCANet(dt2)
    dataset_h_test = (dt_h1, dt_h2)

ts_train2 = time.time()
print(f'Finish training of AENet in {ts_train2-ts_train1:.1e} seconds.')
dataloader_h_train = torch.utils.data.DataLoader(dataset_h_train,
                                                 batch_size=batch_size,
                                                 shuffle=True,
                                                 num_workers=0)
#%% 4. Train ODE based on trained autoencoder nets
print('Train the ODE model first with PCA or trained SAE ...')
optimizer, scheduler = ode.get_opt_sch(ONet, lr=lr_ode,
                                       weight_decay=wt_decay,
                                       patience=patience_ode,
                                       amsgrad=amsgrad,
                                       lr_min=lr_ode_min, epoch=epochs)
log = ONet.train_ode(optimizer,
                     dataloader_h_train,
                     epochs,
                     dataset_h_test, scheduler, dt, p=2)
te_train1 = time.time()
print(f'Finish pre-train of ODENet in {te_train1-ts_train2:.1e} seconds.')
ode_loss_init = log[-1, -2]

print(f'pca_isom_loss={pca_isom_loss:.3e}',
      f'pca_ae_loss={pca_loss_train:.3e}',
      f'ode_loss={ode_loss_init:.3e}')

#%% 5. Train AE and ODENet together
print('Train the Auto-Encoder Net and ODENet together ...')
optimizer = optim.AdamW(chain(filter(lambda p: p.requires_grad,
                                     PCANet.parameters()),
                              ONet.parameters()),
                        lr=lr_e2e,
                        amsgrad=amsgrad,
                        weight_decay=wt_decay)
e = e2e_epoch
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=[3*e//10, 7*e//10, 9*e//10], gamma=0.25)

log2 = train_all(PCANet, ONet, optimizer,
                 dataloader_train,
                 e2e_epoch,
                 dataset_test, scheduler)
te_train2 = time.time()
print(f'Finish main-train of AE-ODENet in {te_train2-te_train1:.1e} seconds.')

train_loss, test_loss = log2[-1, 3], log2[-1, 5]
ae_loss_train, ae_loss_test = log2[-1, 2], log2[-1, 4]
print(f'>>Results: r={cfg.rRa}',
      f'nPC={nPC}',
      f'nH={nHnode}',
      f'L={nL}',
      f'iseed={seed}',
      f'ae_train={ae_loss_train:.2e}',
      f'ae_test={ae_loss_test:.2e}',
      f'ode_train={train_loss:.2e}',
      f'ode_test={test_loss:.2e}')

logt = np.zeros([log.shape[0], log2.shape[1]])
logt[:, [0, 3, 5, 6]] = log  # epoch, loss_ode, test_ode, lr
logt[:, 2] = ae_loss_init
logt[:, 4] = ae_loss_init
logt[:, 1] = logt[:, 2] + logt[:, 3]
log = np.vstack([logt, log2])
np.savetxt(outloc+f'_ae{nPC}_{onet}_train_s{seed}.txt',
           log, delimiter=', ', fmt='%.3e')
torch.save(PCANet.state_dict(), outloc +
           f'_{onet}_ae{nPC}_model_dict.pth')
torch.save(ONet.state_dict(), outloc +
           f'_ae{nPC}_{onet}_model_dict_L{nL}_s{seed}.pth')

# %% plot the result
f = plt.figure()
ax = f.add_subplot(111)
nlog = log.shape[0]
plt.semilogy(np.arange(nlog)+1, log[:, 2], label='ae_loss_train')
plt.semilogy(np.arange(nlog)+1, log[:, 3], label='ode_loss_train')
plt.semilogy(np.arange(nlog)+1, log[:, 4], label='ae_loss_test')
plt.semilogy(np.arange(nlog)+1, log[:, 5], label='ode_loss_test')
ax.yaxis.set_ticks_position('both')
plt.legend()
plt.tick_params(axis='y', which='both', labelleft='on', labelright='on')
plt.grid(True, which='major')
plt.xlabel('epoch')
plt.ylabel('relative MSE')
plt.tight_layout()
plt.savefig(outloc+f'_ae{nPC}_train_s{seed}.pdf', bbox_inches='tight', dpi=200)
if mpl.get_backend() in mpl.rcsetup.interactive_bk:
    plt.draw()
    plt.pause(5)

# %% save encoded samples
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = torch.tensor(dataset).float().to(device)
with torch.no_grad():
    output_e = PCANet.encode(dataset)
    encode = output_e.reshape([nf, nPC])
np.savetxt(outloc+f'_ae{nPC}_{onet}_enc_data.txt.gz',
           encode, delimiter=', ', fmt='%.9e')
