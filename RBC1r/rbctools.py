#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Utility functions to Read, preprocess Rayleigh-Bernard Convection data
    and make visualizations
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

mpl.use('agg')
mpl.rcParams['legend.fontsize'] = 12


def data_to_loaders(hvar, tr_ratio, batch_size):
    nf = hvar.shape[0]
    nS_train = int(nf//2 * tr_ratio)
    ds1 = torch.FloatTensor(hvar[0:2*nS_train:2, :])
    ds2 = torch.FloatTensor(hvar[1:2*nS_train:2, :])
    dataset_train = data.TensorDataset(ds1, ds2)
    dataloader_train = data.DataLoader(dataset_train,
                                       batch_size=batch_size,
                                       shuffle=True,
                                       num_workers=0)
    dt1 = torch.FloatTensor(hvar[2*nS_train::2, :])
    dt2 = torch.FloatTensor(hvar[2*nS_train+1::2, :])
    dataset_test = (dt1, dt2)
    return dataloader_train, dataset_test


def load_uvh_data(h5file):
    """ Load u,v, h (temperature) from hdf5 file `h5file`
        return uvh, coordx, coordy, nx, ny, nf
        with: nx = coordx.size, ny=coordy.size, uvh.shape == (3*nf, nx*ny)
         """
    with h5py.File(h5file, 'r') as F:
        coordx = F['/coordx/value'][()]
        coordy = F['/coordy/value'][()]
        u = F['/ua/value'][()]
        v = F['/va/value'][()]
        h = F['/ha/value'][()]
    [nf, nx, ny] = u.shape
    u = np.reshape(u, (nf, nx*ny))
    v = np.reshape(v, (nf, nx*ny))
    h = np.reshape(h, (nf, nx*ny))
    uvh = np.concatenate((u, v, h), axis=1)
    coordx = coordx.flatten()
    coordy = coordy.flatten()
    print(f' Loading {h5file} done')
    print(f' nS={nf}, nx={nx}, ny={ny}', flush=True)
    return uvh, coordx, coordy, nx, ny, nf


def downsampling(uvh, coordx, coordy, ratio, ts=0, nRun=1, te=None):
    nS = uvh.shape[0]
    nx = coordx.size
    ny = coordy.size
    nOut = nS//nRun
    uvh = uvh.reshape([nRun, nOut, 3, nx, ny])
    if te is None:
        te = nOut
    uvh_ds = uvh[:, ts:te, :, ::ratio, ::ratio]
    nOut1 = uvh_ds.shape[1]
    nS1 = nRun * nOut1
    uvh_ds = uvh_ds.reshape([nRun*nOut1, -1])
    coordx_ds = coordx[::ratio]
    coordy_ds = coordy[::ratio]
    nx = coordx_ds.size
    ny = coordy_ds.size
    print('Downsample to')
    print(f'nS={nS1}, nx={nx}, ny={ny}, nRun={nRun}, nOut={nOut1}', flush=True)
    return uvh_ds, coordx_ds, coordy_ds, nx, ny


def plot_pca_var(var_ratio, var_cumsum, stitle='pca_var'):
    nPC = len(var_ratio)
    f = plt.figure()
    ax = f.add_subplot(111)
    plt.semilogy(np.arange(nPC)+1, var_ratio, 'o-', markersize=4)
    plt.semilogy(np.arange(nPC)+1, 1-var_cumsum, '+-', markersize=8)
    ax.yaxis.set_ticks_position('both')
    plt.tick_params(axis='y', which='both', labelleft='on', labelright='on')
    plt.grid(True, which='major')
    plt.xlabel('Number of components')
    plt.ylabel('Variance')
    plt.legend(['variance ratio', 'variance residual'])
    plt.tight_layout()
    figName = stitle+'.pdf'
    plt.savefig(figName, bbox_inches='tight', dpi=200)
    if mpl.get_backend() in mpl.rcsetup.interactive_bk:
        plt.draw()
        plt.pause(5)


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


def plot_2field2d(xx, vv, Pot, Potex,
                  clevels, savefile,
                  tit1='Learned Potential', tit2='Exact Potential'):
    """ Plot two potential contour side by side in one figure """
    f = plt.figure()
    ax = f.add_subplot(121)
    cp = ax.contour(xx, vv, Pot, clevels)
    plt.clabel(cp, fontsize=12)
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('v')
    ax.set_title(tit1)

    ax = f.add_subplot(122)
    cp = ax.contour(xx, vv, Potex, clevels)
    plt.clabel(cp, fontsize=12)
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('v')
    ax.set_title(tit2)

    plt.tight_layout()
    plt.savefig(savefile+'.pdf', bbox_inches='tight', dpi=200)
    if mpl.get_backend() in mpl.rcsetup.interactive_bk:
        plt.draw()
        plt.pause(5)
        plt.close()


def plot_3field2d(xx, vv, Pot, PotP, Potex,
                  clevels, savefile,
                  tit1='Learned Potential',
                  tit2='Learned Potential after alignment',
                  tit3='Exact Potential'):
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
        plt.pause(5)
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
        plt.pause(2)
        plt.close()
    else:
        plt.show()


def plot_nTraj2d_quiver(dataset, savefile, C=None, xlabel='$h_1$', ylabel='$h_2$'):
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
    # plt.draw()
    plt.close()


def plot_nTran3d_quiver2d(dataset, stitle):
    X = dataset[::2, 0]
    Y = dataset[::2, 1]
    Z = dataset[::2, 2]
    dX = dataset[1::2, 0] - X
    dY = dataset[1::2, 1] - Y
    dZ = dataset[1::2, 2] - Z
    fig = plt.figure(figsize=(12, 12))

    ax1 = fig.add_subplot(221)
    plt.title(stitle)
    plt.quiver(X, Y, dX, dY, Z, cmap='hot')
    ax1.set_aspect('equal')
    ax1.set(xlim=(-1, 1), ylim=(-1, 1))
    plt.xlabel('$h_1$')
    plt.ylabel('$h_2$')
    plt.colorbar()

    ax2 = fig.add_subplot(222)
    plt.quiver(X, Z, dX, dZ, Y, cmap='hot')
    ax2.set_aspect('equal')
    ax2.set(xlim=(-1, 1), ylim=(-1, 1))
    plt.xlabel('$h_1$')
    plt.ylabel('$h_3$')
    plt.colorbar()

    ax3 = fig.add_subplot(223)
    plt.quiver(Y, Z, dY, dZ, X, cmap='hot')
    ax3.set_aspect('equal')
    ax3.set(xlim=(-1, 1), ylim=(-1, 1))
    plt.xlabel('$h_2$')
    plt.ylabel('$h_3$')
    plt.colorbar()

    plt.savefig(stitle+'.pdf', bbox_inches='tight')
    plt.draw()
    plt.pause(5)


def make_quiver3d_movie(dataset, sfile):
    X = dataset[::2, 0]
    Y = dataset[::2, 1]
    Z = dataset[::2, 2]
    dX = dataset[1::2, 0] - X
    dY = dataset[1::2, 1] - Y
    dZ = dataset[1::2, 2] - Z

    fig = plt.figure()
    ax = Axes3D(fig, elev=30, azim=60)
    V = np.sqrt(dX**2 + dY**2 + dZ**2)
    Vm = np.max(V)
    V = np.sqrt(V/Vm)    # too much V close to 0.
    dX = dX/Vm
    dY = dY/Vm
    dZ = dZ/Vm
    c = plt.cm.plasma(np.sqrt(V))  # viridis, plasma, hot
    ax.quiver3D(X, Y, Z, dX, dY, dZ, colors=c, length=0.1, linewidth=1,
                alpha=1, normalize=False, arrow_length_ratio=0)

    ax.set_title('Trajectories of the first 3 components')
    ax.set_axis_on()
    ax.set_frame_on(True)
    ax.set_autoscale_on(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    def update(i):
        ax.view_init(elev=30, azim=i)
        plt.draw()
        return ax

    anim = FuncAnimation(fig, update,
                         frames=np.arange(0, 360, 15), interval=200)
    anim.save(sfile+'.gif', dpi=150, writer='imagemagick')
    plt.pause(30)


def plot_2Traj3d_tphase(h_ode, htest, stitle, pp=False):
    ''' draw one trajectory in different views '''
    nx = htest.shape[0]
    nh = h_ode.shape[0]
    xt = np.arange(nx)
    xh = np.arange(nh)/nh*nx

    if nx > 100:
        fig = plt.figure(figsize=[8, 2.5])
    else:
        fig = plt.figure(figsize=[8, 4])
    plt.subplot(111)
    plt.plot(xh, h_ode[:, 0], color='C0', alpha=0.7,
             linewidth=2, label='$h_1$ ODE', zorder=3)
    plt.plot(xh, h_ode[:, 1], '--', color='C1', alpha=0.7,
             linewidth=2, label='$h_2$ ODE', zorder=3)
    plt.plot(xh, h_ode[:, 2],  color='C3', alpha=0.7,
             linewidth=1, label='$h_3$ ODE', zorder=3)
    plt.plot(xt, htest[:, 0], 'ko', alpha=0.7,
             markersize=2, label='$h_1$ sample')
    plt.plot(xt, htest[:, 1], 'ks', alpha=0.7,
             markersize=2, label='$h_2$ sample')
    plt.plot(xt, htest[:, 2], 'kd', alpha=0.7,
             markersize=2, label='$h_3$ sample')
    plt.legend(loc=0, ncol=3)
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.xlabel('Time')
    plt.savefig(stitle+'.pdf', bbox_inches='tight', dpi=200)
    plt.close()

    if pp:
        PCs = h_ode
        PCref = htest
        nVar = PCs.shape[-1]
        PCs = PCs.reshape([-1, nVar])
        ii = np.arange(PCref.shape[0])

        fig = plt.figure(figsize=[12, 4])
        xa = [0, 0, 1]
        ya = [1, 2, 2]
        for i in range(3):
            ax = fig.add_subplot(1, 3, i+1)
            ax.scatter(PCref[:, xa[i]], PCref[:, ya[i]],
                       c=ii, s=2, marker='.', alpha=0.8, cmap='hot',
                       edgecolors=None, zorder=2)
            ax.plot(PCref[:, xa[i]], PCref[:, ya[i]], '-', color='grey',
                    linewidth=0.5, alpha=0.3, zorder=1)
            ax.plot(PCs[:, xa[i]], PCs[:, ya[i]], 'r-',
                    linewidth=1.5, alpha=0.4, zorder=3)
            ax.set_xlabel(f'P{xa[i]+1}')
            ax.set_ylabel(f'P{ya[i]+1}')
            ax.grid(True)
            plt.tight_layout()

        plt.savefig(stitle+'_pp.pdf', bbox_inches='tight', dpi=200)
        if mpl.get_backend() in mpl.rcsetup.interactive_bk:
            plt.draw()
            plt.pause(1)
        plt.close()


def plot_2Traj3d_3view(PCs, PCref, stitle, nPath=1,
                       xyzlim=[-40, 45, -18, 20, -26, 38]):
    nVar = PCs.shape[-1]
    PCs = PCs.reshape([nPath, -1, nVar])
    ii = np.arange(PCref.shape[1])

    xyzlim = np.array(xyzlim) * 0.9
    fig = plt.figure(figsize=[12, 4])
    fig.subplots_adjust(top=0.3, wspace=0.2, hspace=0.2)

    def plot_subfig():
        for ip in np.arange(nPath-1):
            ax.scatter(PCref[ip, :, 0], PCref[ip, :, 1], PCref[ip, :, 2],
                       c=ii, s=1, marker='.', alpha=0.2,
                       edgecolors=None, zorder=1)
            ax.plot(PCs[ip, :, 0], PCs[ip, :, 1], PCs[ip, :, 2], color='grey',
                    linewidth=0.5, alpha=0.2, zorder=0)
        ip = nPath-1
        ax.plot(PCs[ip, :, 0], PCs[ip, :, 1], PCs[ip, :, 2], 'r-',
                linewidth=1, alpha=0.5, zorder=2)
        ax.scatter(PCref[ip, :, 0], PCref[ip, :, 1], PCref[ip, :, 2],
                   c=ii, s=1, marker='o', alpha=0.5, zorder=3,
                   edgecolors=None, cmap='hot')
        ax.grid(False)
        ax.set_xlabel('$h_1$')
        ax.set_ylabel('$h_2$')
        ax.set_zlabel('$h_3$')
        ax.set_xticks([-30, 30])
        ax.set_yticks([-10, 10])
        ax.set_zticks([-20, 20])
        ax.set_xlim(xyzlim[0], xyzlim[1])
        ax.set_ylim(xyzlim[2], xyzlim[3])
        ax.set_zlim(xyzlim[4], xyzlim[5])
        plt.tight_layout()

    ax = fig.add_subplot(131, projection='3d')
    plot_subfig()
    ax.view_init(azim=-80, elev=50)
    plt.tight_layout()

    ax = fig.add_subplot(132, projection='3d')
    plot_subfig()
    ax.view_init(azim=-145, elev=65)
    plt.tight_layout()

    plt.savefig(stitle+'.pdf', bbox_inches='tight', dpi=200)
    if mpl.get_backend() in mpl.rcsetup.interactive_bk:
        plt.draw()
        plt.pause(2)
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
    ax.set_xlabel('$h_1$')
    ax.set_ylabel('$h_2$')
    ax.set_zlabel('$h_3$')
    plt.tight_layout()
    plt.savefig(stitle+'.pdf', bbox_inches='tight', dpi=400)
    if mpl.get_backend() in mpl.rcsetup.interactive_bk:
        plt.draw()
        plt.pause(3)
    plt.close()


def plot_nTraj_t2d(PCs, stitle, nPath=1):
    ''' Plot multiple trajectories in time space, put the trajectories
        next to each other along t-axis, so it can be used to find trajectories
        with particular properties
    '''
    nS = PCs.shape[0]
    nPC = PCs.shape[1]
    PathLen = nS//nPath
    fig = plt.figure(figsize=(12, 4))
    ax2 = fig.add_subplot(111)
    PCs = PCs.reshape([nPath, PathLen, -1])
    ymin = PCs.min()
    ymax = PCs.max()
    for i in np.arange(nPath):
        ii = np.arange(PCs.shape[1]) + i*PathLen
        ax2.plot(ii, PCs[i, :, 0], color='C0', alpha=0.7)
        ax2.plot(ii, PCs[i, :, 1], color='C1', alpha=0.7)
        if nPC > 2:
            ax2.plot(ii, PCs[i, :, 2], color='C2', alpha=0.7)
        ypos = np.array([1.1*ymin, 1.1*ymax])
        ax2.plot(ypos*0+i*PathLen, ypos, linewidth=0.5, color='grey')
        plt.text((i+0.1)*PathLen, 1.06*ymin, f'{i}', fontsize=10)
    ax2.legend(['PC1', 'PC2', 'PC3'], ncol=3, loc='upper center',
               framealpha=1., fontsize='medium')
    ax2.set_xticks([0, PathLen])
    plt.tight_layout()
    plt.xlim(0, nS)
    plt.ylim(1.1*ymin, 1.1*ymax)
    plt.grid(False)
    plt.savefig(stitle+'.pdf', bbox_inches='tight', dpi=150)
    if mpl.get_backend() in mpl.rcsetup.interactive_bk:
        plt.draw()
        plt.pause(5)
    plt.close()


def plot_nTraj2d_phase(PCs, stitle, nPath=1, xlabel='$h_1$', ylabel='$h_2$'):
    ''' Plot multiple 2d trajectories in phase space, i.e phase portraits
        plot the trajectory curves using plt.plot, scatter the snapshots using
        different colors.
    '''
    nS = PCs.shape[0]
    PathLen = nS//nPath
    fig = plt.figure(figsize=(10, 10))
    ax2 = fig.add_subplot(111)
    PCs = PCs.reshape([nPath, PathLen, -1])
    ii = np.arange(PathLen)
    z0 = np.mean(PCs[:, -1, 2])
    zabs = np.max(np.abs(PCs[:, -1, 2].flatten() - z0))
    for i in np.arange(nPath):
        ax2.plot(PCs[i, :, 0], PCs[i, :, 1], linewidth=2,
                 alpha=0.4, label=f'Path {i}')
        ii = (PCs[i, :, 2] - z0)/(zabs*2) + 0.5
        ax2.scatter(PCs[i, :, 0], PCs[i, :, 1], c=ii, s=6,
                    marker='.', alpha=0.9, edgecolors=None, zorder=3,
                    cmap='gray')
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.grid('major')
    plt.savefig(stitle+'_phased.pdf', bbox_inches='tight', dpi=288)
    if mpl.get_backend() in mpl.rcsetup.interactive_bk:
        plt.draw()
        plt.pause(5)
    plt.close()


def plotflow(coordx, coordy, uvh, stitle):
    coordx = coordx.flatten()
    coordy = coordy.flatten()
    nx = coordx.size
    ny = coordy.size
    uvh = uvh.flatten()
    u = uvh[0:nx*ny].reshape([nx, ny]).T
    v = uvh[nx*ny:2*nx*ny].reshape([nx, ny]).T
    h = uvh[2*nx*ny:3*nx*ny].reshape([nx, ny]).T

    xmin = np.min(coordx)
    xmax = np.max(coordx)
    ymin = np.min(coordy)
    ymax = np.max(coordy)
    lx = xmax - xmin
    ly = ymax - ymin

    width = 14
    height = width/lx*ly * 1
    fig = plt.figure(figsize=(width, height))

    if nx > 80:
        di = 2
        dj = 2
    else:
        di = 1
        dj = 1
    ii = np.arange(0, ny, di).reshape(-1, 1)
    jj = np.arange(0, nx, dj).reshape(1, -1)

    ax1 = fig.add_subplot(1, 1, 1)
    im = ax1.imshow(h[ii, jj], origin='lower',
                    extent=[xmin, xmax, ymin, ymax],
                    cmap='jet', norm=mpl.colors.Normalize(-0.8, 0.8))

    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = plt.colorbar(im, cax=cax)
    cbar_ticks = np.linspace(-0.8, 0.8, num=5, endpoint=True)
    cbar.ax.set_autoscale_on(True)
    cbar.set_ticks(cbar_ticks)
    ax1.quiver(coordx[jj], coordy[ii], u[ii, jj], v[ii, jj],
               units='xy', pivot='tail', headwidth=2, scale=8, color='black')
    max_vel = np.max(np.sqrt(u**2 + v**2))
    print(stitle+f' max velocity={max_vel}')

    plt.tight_layout()
    plt.savefig(stitle+'.pdf', bbox_inches='tight', dpi=288)
    if mpl.get_backend() in mpl.rcsetup.interactive_bk:
        plt.draw()
        plt.pause(5)
        plt.close()
