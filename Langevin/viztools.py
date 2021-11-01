#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" some utility functions to make visualizations
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import torch.utils.data as data
import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl
from mpl_toolkits.axes_grid1.axes_grid import Grid
mpl.rcParams['legend.fontsize'] = 12
mpl.rcParams['figure.subplot.left'] = 0.
mpl.rcParams['figure.subplot.bottom'] = 0.
mpl.rcParams['figure.subplot.right'] = 1.


def data_to_loaders(hvar, tr_ratio, batch_size, ns=0):
    nf = hvar.shape[0]
    nS_train = int(nf//2 * tr_ratio)
    ds1 = torch.FloatTensor(hvar[0:2*nS_train-ns*2:2, :])
    ds2 = torch.FloatTensor(hvar[ns*2+1:2*nS_train:2, :])
    dataset_train = data.TensorDataset(ds1, ds2)
    dataloader_train = data.DataLoader(dataset_train,
                                       batch_size=batch_size,
                                       shuffle=True,
                                       num_workers=0)
    dt1 = torch.FloatTensor(hvar[2*nS_train-ns*2:nf-ns*2:2, :])
    dt2 = torch.FloatTensor(hvar[2*nS_train+1::2, :])
    dataset_test = (dt1, dt2)
    return dataloader_train, dataset_test

def plot_ode_train_log(log, savefile):
    f = plt.figure()
    ax = f.add_subplot(111)
    nlog = log.shape[0]
    ee = np.arange(nlog)+1
    plt.semilogy(ee, log[:, 1], label='train error')
    plt.semilogy(ee, log[:, 2], label='test error')
    ax.yaxis.set_ticks_position('both')
    plt.tick_params(axis='y', which='both', labelleft='on', labelright='on')
    plt.grid(True, which='major')
    plt.xlabel('epoch')
    plt.ylabel('MSE')
    plt.legend()
    plt.tight_layout()
    plt.savefig(savefile+'_train.pdf', bbox_inches='tight', dpi=144)
    if mpl.get_backend() in mpl.rcsetup.interactive_bk:
        plt.draw()
        plt.pause(5)
        plt.close()


def plot_nTraj3d_scatter(PCs, stitle, nPath=1, ts=1, azim=-60, elev=25):
    nS = PCs.shape[0]
    PathLen = nS//nPath
    print(f'Scatter3d data: nS={nS}, nPath={nPath}, PathLen={PathLen}')

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(azim=azim, elev=elev)
    Paths = PCs.reshape([nPath, PathLen, -1])
    Paths = Paths[:, ts:, 0:3]
    Vel = np.sqrt(np.sum((Paths[:, 1:, :] - Paths[:, 0:-1, :])**2, axis=-1))
    Vel = Vel.flatten()/np.max(Vel) * 255
    Vel = Vel.astype(int)
    PC = np.reshape(Paths[:, :, 0:3], [-1, 3])
    for i in np.arange(nPath):
        ax.plot(Paths[i, :, 0], Paths[i, :, 1], Paths[i, :, 2],
                c='gray', linewidth=0.5, alpha=0.3)
    ii = np.arange(PC.shape[0])
    ii = ii % (PathLen-ts)  # make color according to time index
    size = 0.5 if PC.shape[0] > 1000 else 1.5
    ax.scatter(PC[:, 0], PC[:, 1], PC[:, 2], c=ii, s=size,
               marker='o', alpha=0.7, edgecolors=None, cmap='hot')
    ax.grid(False)
    ax.set_xlabel('P1')
    ax.set_ylabel('P2')
    ax.set_zlabel('P3')
    plt.tight_layout()
    plt.savefig(stitle+'.pdf', bbox_inches='tight', dpi=400)
    if mpl.get_backend() in mpl.rcsetup.interactive_bk:
        plt.draw()
        plt.pause(3)
    plt.close()


def plot_nTraj2d_quiver(dataset, savefile, C=None, xlabel='P1', ylabel='P2'):
    X = dataset[::2, 0]
    Y = dataset[::2, 1]
    dX = dataset[1::2, 0] - X
    dY = dataset[1::2, 1] - Y
    fig = plt.figure(figsize=(8, 8), dpi=144)
    ax1 = fig.add_subplot(111)
    if C is None:
        C = np.sqrt(X**2+Y**2)
    plt.quiver(X, Y, dX, dY, C, cmap='RdBu', headwidth=2, headlength=2)
    # brg, hsv, CMRmap, gnuplot
    ax1.set_aspect('equal')
    ax1.set(xlim=(-1, 1), ylim=(-1, 1))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(savefile+'.pdf', bbox_inches='tight', dpi=288)
    plt.close()


def plot_field2d(xx, vv, field, method='imshow', stitle=' ',
                 savefile=None, xylabel=('x1', 'x2')):
    """ Plot 2d field by imshow or contour """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    if method == 'contour':
        vmin = min(field)
        vmax = max(field)
        clevels = np.linspace(vmin, vmax, 6)
        cp = ax.contour(xx, vv, field, clevels)
        plt.clabel(cp, fontsize=12)
    else:
        xmin = np.min(xx)
        xmax = np.max(xx)
        ymin = np.min(vv)
        ymax = np.max(vv)
        im = ax.imshow(field, origin='lower',
                       extent=[xmin, xmax, ymin, ymax],
                       cmap='jet')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.08)
        plt.colorbar(im, cax=cax)
    ax.set_aspect('equal')
    ax.set_xlabel(xylabel[0])
    ax.set_ylabel(xylabel[1])
    ax.set_title(stitle)
    plt.tight_layout()
    plt.draw()
    if savefile is not None:
        plt.savefig(savefile+'.pdf', bbox_inches='tight', dpi=200)
        plt.close()
    else:
        plt.show()


def plot_3field2d(xx, vv, Pot, PotP, Potex,
                  clevels, savefile,
                  tit1='Learned energy',
                  tit2='Learned energy after alignment',
                  tit3='Exact energy'):
    """ Plot three potential contour side by side in one figure """
    f = plt.figure(figsize=[12, 4])
    ax = f.add_subplot(131)
    cp = ax.contour(xx, vv, Pot, clevels)
    plt.clabel(cp, fontsize=10)
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('v')
    ax.set_title(tit1)

    ax = f.add_subplot(132)
    cp = ax.contour(xx, vv, PotP, clevels)
    plt.clabel(cp, fontsize=10)
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('v')
    ax.set_title(tit2)

    ax = f.add_subplot(133)
    cp = ax.contour(xx, vv, Potex, clevels)
    plt.clabel(cp, fontsize=10)
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('v')
    ax.set_title(tit3)

    plt.tight_layout()
    plt.savefig(savefile+'.pdf', bbox_inches='tight', dpi=200)
    if mpl.get_backend() in mpl.rcsetup.interactive_bk:
        plt.draw()
        plt.pause(3)
        plt.close()


def plot_3field2d_new(xx, vv, Pot, PotP, Potex,
                      clevels, savefile,
                      tit1='Learned energy',
                      tit2='Learned energy (aligned)',
                      tit3='Exact energy'):
    """ Plot three potential contour side by side in one figure """
    f = plt.figure(figsize=[9, 3.5])
    grid = Grid(f, 111,
                nrows_ncols=(1, 3),
                axes_pad=0.15,
                label_mode="L",
                share_all=True,
                )
    ax = grid[0]
    cp=ax.contour(xx, vv, Pot, clevels)
    plt.clabel(cp, fontsize=9, fmt='%.2f')
    ax.set_title(tit1, fontsize=14)
    ax.set_xlabel('$x$', fontsize=12)
    # ax.set_aspect('equal')
    ax.set_ylabel('$v$', fontsize=12)
        
    ax = grid[1]
    cp=ax.contour(xx, vv, PotP, clevels)
    plt.clabel(cp, fontsize=9, fmt='%.2f')
    ax.set_title(tit2, fontsize=14)
    ax.set_xlabel('$x$', fontsize=12)
    
    
    ax = grid[2]
    cp=ax.contour(xx, vv, Potex, clevels)
    plt.clabel(cp, fontsize=9, fmt='%.2f')
    ax.set_title(tit3, fontsize=14)
    ax.set_xlabel('$x$', fontsize=12)
    
    
    # This only affects axes in first column and second row as share_all=False.
    grid.axes_llc.set_xticks([-0.8, 0, 0.8])
    grid.axes_llc.set_yticks([-1, 0, 1])

    plt.tight_layout()
    plt.savefig(savefile+'.pdf', bbox_inches='tight', dpi=200)
