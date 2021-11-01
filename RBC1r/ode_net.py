#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
@File 	 : ode_net.py
@Time 	 : 2020/05/1
@Author  : Haijn Yu <hyu@lsec.cc.ac.cn>
         : Xinyan Tian <txy@lsec.cc.ac.cn>
@Desc	 : Define OnsagerNet and several other nerual network models for
           representing and learning ODEs.
           The right hand functions (vector field) of ODE systems are
           represented by nerual networks.
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
import torch.optim.lr_scheduler as lr_scheduler
import copy

fid = 0			 # ID of actiation function, see F_act()

fnames = ('ReQUr', 'ReLU', 'ReQU', 'ReCU', 'x^2', 'x^3',
          'softplus', 'ELU', 'sigmoid', 'tanh')

def F_act(x):
    if fid == 0:
        return F.relu(x)**2 - F.relu(x-0.5)**2
    if fid == 1:
        return F.relu(x)
    if fid == 2:
        return F.relu(x)**2
    if fid == 3:
        return F.relu(x)**3
    if fid == 4:
        return x**2
    if fid == 5:
        return x**3
    if fid == 6:
        return F.softplus(x)
    if fid == 7:
        return F.elu(x)
    if fid == 8:
        return torch.sigmoid(x)
    if fid == 9:
        return torch.tanh(x)
    return F.relu(x)


def mse_p_loss(h2_ode, h2, p=2.0, dt=0.001):
    """ define the MSE loss function with different p-norm
        eth: error threshold
    """
    diff = (h2_ode - h2)/dt
    if np.abs(p-2.0) < 1e-7:
        mse = torch.mean(F.relu(torch.sum(diff**2, dim=1)))
    else:
        mse = torch.pow(torch.mean(torch.pow(
            F.relu(torch.sum(diff**2, dim=1)),
            p/2)),
            2.0/p)
    return mse


class ODENet(nn.Module):
    """ A neural network to for the rhs function of an ODE,
    used to fitting data """

    def __init__(self, n_nodes=None, ResNet=True,
                 init_gain=0.01,
                 f_act=F_act):
        super().__init__()
        if n_nodes is None:
            return
        self.nVar = n_nodes[0]        # number of variables
        self.nL = n_nodes.size        # number of layers
        self.nNodes = np.zeros(self.nL+1, dtype=int)
        self.nNodes[:self.nL] = n_nodes
        self.nNodes[self.nL] = self.nVar
        self.F_act = f_act
        self.layers = nn.ModuleList([nn.Linear(self.nNodes[i],
                                               self.nNodes[i+1])
                                     for i in range(self.nL)])
        if ResNet:
            self.ResNet = 1.0
            assert np.sum(n_nodes[1:]-n_nodes[1]) == 0, \
                f'ResNet structure is not implemented for {n_nodes}'
        else:
            self.ResNet = 0.0

        for i in range(self.nL):
            init.xavier_uniform_(self.layers[i].weight, gain=init_gain)
            init.uniform_(self.layers[i].bias, 0, init_gain)

    def print(self):
        for name, p in self.named_parameters():
            print(name, ' Size=', p.size(), ' DoF=', p.numel())
        total_num = sum(p.numel() for p in self.parameters())
        trainable_num = sum(p.numel()
                            for p in self.parameters() if p.requires_grad)
        print('\t >> Total parameters:', total_num, ' Trainable:', trainable_num)
        for name, p in self.named_parameters():
            print(name, ' ', p)

    def size(self):
        trainable_num = sum(p.numel()
                            for p in self.parameters() if p.requires_grad)
        return trainable_num

    def forward(self, inputs, test=False):
        shape = inputs.shape
        output = inputs.view(-1, self.nVar)
        output = self.F_act(self.layers[0](output))
        for i in range(1, self.nL-1):
            output = self.F_act(self.layers[i](output)) + self.ResNet * output
        output = self.layers[self.nL-1](output)
        output = output.view(*shape)
        return output

    forward_test = forward

    def calc_Jacobi(self, inputs):
        """ Calculate the Jacobi of the rhs """
        inputs = torch.tensor(inputs).float().view(-1, self.nVar)
        bs = inputs.shape[0]
        nVar = self.nVar
        with torch.enable_grad():
            inputs.requires_grad_(True)
            inputs.retain_grad()
            R = self(inputs)
            J = torch.zeros(bs, nVar, nVar, requires_grad=False)
            for i in range(nVar):
                Gy = torch.sum(R[:, i])
                Gy.backward(retain_graph=True)
                J[:, i, :] = inputs.grad
                inputs.grad.data.zero_()
        J = J.detach().view(-1, self.nVar, self.nVar).squeeze()
        return J

    def ode_rk1(self, h_in, dt, nt=1):
        """ Solve the ODE system with Euler's method """
        hn = h_in
        for i in np.arange(nt):
            h0 = hn
            rhs1 = self(h0)
            hn = h0 + dt * rhs1
        return hn

    def ode_rk2(self, h_in, dt, nt=1):
        """ March the ODE system with RK2 (Heun's method) """
        hn = h_in
        for i in np.arange(nt):
            h0 = hn
            rhs1 = self(h0)
            h1 = h0 + dt * rhs1
            rhs2 = self(h1)
            hn = h0 + dt/2 * (rhs1 + rhs2)
        return hn

    def ode_rk3(self, h_in, dt, nt=1):
        """ Solve the ODE system with RK3 """
        hn = h_in
        for i in np.arange(nt):
            h0 = hn
            rhs1 = self(h0)
            h1 = h0 + dt * rhs1
            rhs2 = self(h1)
            h2 = 3/4*h0 + 1/4*h1 + 1/4*dt*rhs2
            rhs3 = self(h2)
            hn = 1/3*h0 + 2/3*h2 + 2/3*dt*rhs3
        return hn

    def ode_rk4(self, h_in, dt, nt=1):
        """ Solve the ODE system with RK4 """
        hn = h_in
        for i in np.arange(nt):
            h0 = hn
            rhs1 = self(h0)
            h1 = h0 + dt/2.0 * rhs1
            rhs2 = self(h1)
            h2 = h0 + dt/2.0 * rhs2
            rhs3 = self(h2)
            h3 = h0 + dt * rhs3
            rhs4 = self(h3)
            hn = h0 + (rhs1 + 2*rhs2 + 2*rhs3 + rhs4)*dt/6.
        return hn

    def ode_run(self, hinit, dt, T=1, Tout=1):
        Terr = abs(T - int(T/Tout)*Tout)
        assert Terr <= 1e-6, f'T={T} should be multiple of Tout={Tout}'
        Toerr = abs(Tout - int(Tout/dt)*dt)
        assert Toerr < 1e-6,  f'Time step error: Tout={Tout}, dt={dt}'
        nPath = hinit.shape[0]
        nPC = hinit.shape[1]
        nOut = int(T/Tout) + 1
        with torch.no_grad():
            h_ode = np.zeros([nPath, nOut, nPC])
            h_ode[:, 0, :] = hinit[:, :]
            for it in np.arange(nOut-1):
                h0 = torch.FloatTensor(h_ode[:, it, :])
                hf = self.ode_rk3(h0, dt, int(Tout/dt))
                h_ode[:, it+1, :] = hf.data
        return h_ode

    def sparsity_loss(self):
        loss = 0.0
        for name, p in self.named_parameters():
            if 'weight' in name:
                loss += torch.sum(torch.abs(p))
        return loss

    def train_ode(self, optimizer, data_train, epochs,
                  data_test, scheduler, dt, nt=1, p=2):
        ''' train ODENet '''
        nS_train = len(data_train.dataset)
        nS_test = len(data_test[0])
        batch_size = data_train.batch_size
        nPC = self.nVar
        lr = optimizer.param_groups[0]["lr"]

        print('Number of trainning samples =', nS_train)
        print('Number of test samples =', nS_test)
        print(f'Epochs={epochs}, nPC={nPC}, batch_size={batch_size}')
        print(f'initial learning rate = {lr}')

        log = []
        device = torch.device('cpu')
        self.to(device)
        if data_test is not None:
            dt1, dt2 = data_test
            dt1 = dt1.to(device)
            dt2 = dt2.to(device)

        for e in range(epochs):
            loss_acc = 0
            for i, (h1, h2) in enumerate(data_train):
                h1 = h1.to(device)
                h2 = h2.to(device)

                h2_ode = self.ode_rk2(h1, dt/nt, nt)
                loss = mse_p_loss(h2_ode, h2, p=p, dt=dt)
                if loss.item() > 1e6:
                    print(f'e={e}, i={i}, loss=', loss.item())

                loss_acc += loss.item()
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), 1.)
                optimizer.step()
                optimizer.zero_grad()
            loss_acc = (loss_acc * batch_size) / nS_train
            last_lr = optimizer.param_groups[0]["lr"]
            if scheduler is not None:
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(loss_acc)
                else:
                    scheduler.step()
            lr = optimizer.param_groups[0]["lr"]
            errs = [e, loss_acc]

            if data_test is not None:
                with torch.no_grad():
                    dt2_ode = self.ode_rk2(dt1, dt/nt, nt)
                    ode_loss_test = mse_p_loss(dt2_ode, dt2, p=p, dt=dt)
                errs.append(ode_loss_test.item())
            errs.append(last_lr)
            log.append(errs)

            if e % 10 == 0 or e == epochs-1 or lr != last_lr:
                print(f'epoch:{e:4d}/{epochs}', end=' ')
                print(f'loss: {loss_acc:.3e}', end=' ')
                if data_test is not None:
                    print(f'loss_test: {ode_loss_test.item():.3e}', end=' ')
                print(f'lr: {last_lr:.4e}', flush=True)

        log = np.squeeze(np.array(log))
        self.to(torch.device('cpu'))
        return log

def makePDM(matA):
    """ Make Positive Definite Matrix from a given matrix
    matA has a size (batch_size x N x N) """
    AL = torch.tril(matA, 0)
    AU = torch.triu(matA, 1)
    Aant = AU - torch.transpose(AU, 1, 2)
    Asym = torch.bmm(AL, torch.transpose(AL, 1, 2))
    return Asym,  Aant


def makeSPD(A, n):
    """ Make Symmetric Positive Definite matrix from a given matrix
    A has a size (batch_size x N), where N = n*(n+1)/2 """
    A = A.view(-1, (n*(n+1))//2)
    bs = A.shape[0]
    matA = torch.zeros(bs, n*n)
    tril_ind = torch.tril_indices(n, n)
    matA[:, tril_ind[0, :]*n+tril_ind[1, :]] = A
    matA = matA.view(-1, n, n)
    AL = torch.tril(matA, 0)
    Asym = torch.bmm(AL, torch.transpose(AL, 1, 2))
    return Asym


class OnsagerNet(ODENet):
    """ A neural network to for the rhs function of an ODE,
    used to fitting data """

    def __init__(self, n_nodes=None, forcing=True, ResNet=True,
                 pot_beta=0.0,
                 ons_min_d=0.0,
                 init_gain=0.1,
                 f_act=F_act,
                 f_linear=True,
                 ):
        super().__init__()
        if n_nodes is None:   # used for subclasses
            return
        self.nL = n_nodes.size
        self.nVar = n_nodes[0]
        self.nNodes = np.zeros(self.nL+1, dtype=np.int32)
        self.nNodes[:self.nL] = n_nodes
        self.nNodes[self.nL] = self.nVar**2
        self.nPot = self.nVar
        self.forcing = forcing
        self.pot_beta = pot_beta
        self.ons_min_d = ons_min_d
        self.F_act = f_act
        self.f_linear = f_linear
        if ResNet:
            self.ResNet = 1.0
            assert np.sum(n_nodes[1:]-n_nodes[1]) == 0, \
                f'ResNet structure is not implemented for {n_nodes}'
        else:
            self.ResNet = 0.0
        self.baselayer = nn.ModuleList([nn.Linear(self.nNodes[i],
                                                  self.nNodes[i+1])
                                        for i in range(self.nL-1)])
        self.MatLayer = nn.Linear(self.nNodes[self.nL-1], self.nVar**2)
        self.PotLayer = nn.Linear(self.nNodes[self.nL-1], self.nPot)
        self.PotLinear = nn.Linear(self.nVar, self.nPot)

        bias_eps = 0.5
        for i in range(self.nL-1):
            init.xavier_uniform_(self.baselayer[i].weight, gain=init_gain)
            init.uniform_(self.baselayer[i].bias, 0, bias_eps*init_gain)

        init.xavier_uniform_(self.MatLayer.weight, gain=init_gain)
        w = torch.empty(self.nVar, self.nVar, requires_grad=True)
        nn.init.orthogonal_(w, gain=1.0)
        self.MatLayer.bias.data = w.view(-1, self.nVar**2)

        init.orthogonal_(self.PotLayer.weight, gain=init_gain)
        init.uniform_(self.PotLayer.bias, 0, init_gain)
        init.orthogonal_(self.PotLinear.weight, gain=init_gain)
        init.uniform_(self.PotLinear.bias, 0, init_gain)

        if self.forcing:
            if self.f_linear:
                self.lforce = nn.Linear(self.nVar, self.nVar)
            else:
                self.lforce = nn.Linear(self.nNodes[self.nL-1], self.nVar)
            init.orthogonal_(self.lforce.weight, init_gain)
            init.uniform_(self.lforce.bias, 0.0, bias_eps*init_gain)

    def forward(self, inputs, test=False):
        shape = inputs.shape
        inputs = inputs.view(-1, self.nVar)
        with torch.enable_grad():
            inputs.requires_grad_(True)
            if not test:
                inputs.retain_grad()
            output = self.F_act(self.baselayer[0](inputs))
            for i in range(1, self.nL-1):
                output = (self.F_act(self.baselayer[i](output))
                          + self.ResNet*output)
            PotLinear = self.PotLinear(inputs)
            Pot = self.PotLayer(output) + PotLinear
            V = torch.sum(Pot**2) + self.pot_beta * torch.sum(inputs**2)
            if test:
                g, = torch.autograd.grad(V, inputs)
            else:
                g, = torch.autograd.grad(V, inputs, create_graph=True)
            g = - g.view(-1, self.nVar, 1)

        matA = self.MatLayer(output)
        matA = matA.view(-1, self.nVar, self.nVar)
        AM, AW = makePDM(matA)
        MW = AW+AM

        if self.forcing:
            if self.f_linear:
                lforce = self.lforce(inputs)
            else:
                lforce = self.lforce(output)

        output = torch.matmul(MW, g) + self.ons_min_d * g
        if self.forcing:
            output = output + lforce.view(-1, self.nVar, 1)

        output = output.view(*shape)
        return output

    def calc_potential(self, inputs):
        ''' Calculate the potential for post-analysis '''
        output = inputs.view(-1, self.nVar)
        PotLinear = self.PotLinear(output)
        output = self.F_act(self.baselayer[0](output))
        for i in range(1, self.nL-1):
            output = (self.F_act(self.baselayer[i](output))
                      + self.ResNet*output)
        Pot = self.PotLayer(output) + PotLinear
        V = torch.sum(Pot**2, dim=1)
        V += self.pot_beta*torch.sum(inputs**2, dim=1)
        return V

    def save_potential(self, hrange, n=30,
                       savefile='results/OnsagerNet_test_pot'):
        d = min(len(hrange[0]), self.nVar)
        np.savetxt(savefile+'_meta.txt', hrange,
                   delimiter=', ', fmt='%.3e')
        nx = ny = nz = 100
        x = np.linspace(hrange[0][0], hrange[1][0], nx)
        y = np.linspace(hrange[0][1], hrange[1][1], ny)
        device = torch.device('cpu')
        if d == 2:
            xx, yy = np.meshgrid(x, y)
            inputs = np.stack([xx, yy], axis=-1).reshape([-1, 2])
            inputs = torch.FloatTensor(inputs)
        else:
            z = np.linspace(hrange[0][2], hrange[1][2], nz)
            xx, yy, zz = np.meshgrid(x, y, z)
            input3 = np.stack([xx, yy, zz], axis=-1).reshape([-1, 3])
            inputs = np.zeros([input3.shape[0], self.nVar])
            inputs[:, :3] = input3
            inputs = torch.FloatTensor(inputs)
        inputs.to(device)
        self.to(device)
        with torch.no_grad():
            Pot = self.calc_potential(inputs)
        Pot = Pot.detach().numpy().reshape([nx, ny, nz])
        np.savetxt(savefile+'.txt.gz', Pot.reshape([-1, nz]),
                   delimiter=', ', fmt='%.3e')

    def calc_potHessian(self, inputs):
        inputs = inputs.view(-1, self.nVar)
        bs = inputs.shape[0]
        with torch.enable_grad():
            inputs.requires_grad_(True)
            inputs.retain_grad()
            output = inputs.view(-1, self.nVar)
            PotLinear = self.PotLinear(output)
            output = self.F_act(self.baselayer[0](output))
            for i in range(1, self.nL-1):
                output = (self.F_act(self.baselayer[i](output))
                          + self.ResNet*output)
            Pot = self.PotLayer(output) + PotLinear
            V0 = torch.sum(Pot**2, dim=1)
            V0 += self.pot_beta * torch.sum(inputs**2, dim=1)
            V = torch.sum(V0)
            G, = torch.autograd.grad(V, inputs, create_graph=True)

            nVar = self.nVar
            H = torch.zeros(bs, nVar, nVar, requires_grad=False)
            for i in range(nVar):
                y = torch.zeros(nVar, requires_grad=False)
                y[i] = 1
                Gy = torch.sum(G @ y)
                Gy.backward(retain_graph=True)
                H[:, i, :] = inputs.grad
                inputs.grad.data.zero_()
        G = G.detach().view(-1, self.nVar)
        H = H.detach().view(-1, self.nVar, self.nVar)
        V0 = V0.detach()
        return G, H, V0  # Gradient, Hessian, and Potential

    def calc_matA(self, inputs):
        inputs = torch.tensor(inputs).float().view(-1, self.nVar)
        output = self.F_act(self.baselayer[0](inputs))
        for i in range(1, self.nL-1):
            output = (self.F_act(self.baselayer[i](output))
                      + self.ResNet*output)
        matA = self.MatLayer(output)
        matA = matA.view(-1, self.nVar, self.nVar)
        AM, AW = makePDM(matA)
        return AM

    def ode_rk3(self, h_in, dt, nt=1, test=False):
        """ Solve the ODE system with RK3 """
        hn = h_in
        for i in np.arange(nt):
            h0 = hn
            rhs1 = self(h0, test)
            h1 = h0 + dt * rhs1
            rhs2 = self(h1, test)
            h2 = 3/4*h0 + 1/4*h1 + 1/4*dt*rhs2
            rhs3 = self(h2, test)
            hn = 1/3*h0 + 2/3*h2 + 2/3*dt*rhs3
        return hn


class SymODEN(OnsagerNet):
    """ A diffusive Symplectic Net """

    def __init__(self, n_nodes, forcing=True, ResNet=True,
                 pot_beta=0.01,
                 ons_min_d=0.0,
                 init_gain=0.1,
                 f_act=F_act
                 ):
        super().__init__()
        self.nL = n_nodes.size
        self.nVar = n_nodes[0]
        assert self.nVar % 2 == 0, 'nVar in SymODEN must be even!'
        self.n = self.nVar//2
        self.nNodes = np.zeros(self.nL+1, dtype=np.int32)
        self.nNodes[:self.nL] = n_nodes
        self.nNodes[self.nL] = self.nVar**2
        self.nNodes[0] = self.n  # only q in first layer
        self.nPot = self.n
        self.nD = (self.nVar * (self.nVar + 1))//2
        self.nM = (self.n * (self.n+1))//2
        self.forcing = forcing
        self.pot_beta = pot_beta
        self.ons_min_d = ons_min_d
        self.F_act = f_act
        if ResNet:
            self.ResNet = 1.0
            assert np.sum(n_nodes[1:]-n_nodes[1]) == 0, \
                f'ResNet structure is not implemented for {n_nodes}'
        else:
            self.ResNet = 0.0
        self.baselayer = nn.ModuleList([nn.Linear(self.nNodes[i],
                                                  self.nNodes[i+1])
                                        for i in range(self.nL-1)])
        self.DLayer = nn.Linear(self.nNodes[self.nL-1], self.nD)
        self.MLayer = nn.Linear(self.nNodes[self.nL-1], self.nM)
        self.PotLayer = nn.Linear(self.nNodes[self.nL-1], self.nPot)

        for i in range(self.nL-1):
            init.xavier_uniform_(self.baselayer[i].weight, gain=init_gain)
            init.uniform_(self.baselayer[i].bias, 0, init_gain)

        init.xavier_uniform_(self.DLayer.weight, gain=init_gain)
        init.uniform_(self.DLayer.bias, 0, init_gain)

        init.xavier_uniform_(self.MLayer.weight, gain=init_gain)
        init.uniform_(self.MLayer.bias, 0, init_gain)

        init.orthogonal_(self.PotLayer.weight, gain=init_gain)
        init.uniform_(self.PotLayer.bias, 0, init_gain)

        if self.forcing:
            self.lforce = nn.Linear(self.n, self.nVar)
            init.uniform_(self.lforce.weight, 0.0, init_gain)
            init.uniform_(self.lforce.bias, 0.0, init_gain)

    def forward(self, inputs, test=False):
        shape = inputs.shape
        inputs = inputs.view(-1, self.nVar)
        with torch.enable_grad():
            inputs.requires_grad_(True)
            if not test:
                inputs.retain_grad()
            q = inputs[:, 0:self.n]
            output = self.F_act(self.baselayer[0](q))
            for i in range(1, self.nL-1):
                output = (self.F_act(self.baselayer[i](output))
                          + self.ResNet*output)
            Pot = self.PotLayer(output)
            V = torch.sum(Pot**2)
            M = self.MLayer(output)
            MS = makeSPD(M, self.n)
            p = inputs[:, self.n:].view(-1, self.n, 1)
            Mp = torch.matmul(MS, p)
            pMp = torch.matmul(p, Mp)
            H = V + torch.sum(pMp)/2 + self.pot_beta * torch.sum(p**2)
            if test:
                g, = torch.autograd.grad(H, inputs)
            else:
                g, = torch.autograd.grad(H, inputs, create_graph=True)
            g = g.view(-1, self.nVar, 1)
            Hq = g[:, 0:self.n, 0]
            Hp = g[:, self.n:, 0]

        matD = self.DLayer(output)
        DM = makeSPD(matD, self.nVar)
        output = -(torch.matmul(DM, g) + self.ons_min_d * g)
        output[:, 0:self.n, 0] += Hp
        output[:, self.n:, 0] -= Hq

        if self.forcing:
            lforce = self.lforce(q)
            output = output + lforce.view(-1, self.nVar, 1)

        output = output.view(*shape)
        return output


def get_opt_sch(ONet, lr, weight_decay=0,
                lr_min=5e-6, patience=20, cooldown=0, amsgrad=True,
                method='Adam', epoch=600):
    if method == 'Adam':
        optimizer = optim.AdamW(ONet.parameters(),
                                lr=lr,
                                amsgrad=amsgrad,
                                weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                              'min',
                                              factor=0.5,
                                              patience=patience,
                                              cooldown=cooldown,
                                              min_lr=lr_min)

    else:
        optimizer = optim.SGD(ONet.parameters(),
                              lr=lr,
                              momentum=0.9,
                              nesterov=True,
                              weight_decay=weight_decay)
        e = epoch
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[5*e//10, ], gamma=0.2)
    return optimizer, scheduler


# %% 3. run the main script
if __name__ == '__main__':
    print('This is a library that defines several ODE Nets.')
