#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" A module contains several auto-encoders
Last Modified on Wed Apr 14 2020
Last Modified on Wed May 6, 2020
@author: Haijun Yu <hyu@lsec.cc.ac.cn>
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
import torch.nn.functional as F

fid = 0

float_formatter = "{:.6e}".format
np.set_printoptions(formatter={'float_kind': float_formatter})

if __name__ == '__main__':
    print('This is library defines several autoencoders.')

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


def mse_loss_rel(outputs, imgs):
    """ define the relative MSE loss function """
    criterion = nn.MSELoss()
    mse = criterion(outputs, imgs)
    mse_tot = torch.mul(imgs, imgs).mean()
    mse = mse/mse_tot
    return mse


def eval_model_mse_rel(model, dataset):
    """ Evaluate the MSE of the NN constructed from PCA for dataset """
    dataset = torch.tensor(dataset)
    dataset = dataset.float()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    imgs = dataset.to(device)
    outputs = model(imgs)
    mse_err = mse_loss_rel(outputs, imgs)
    return mse_err


class PCA_Encoder(nn.Module):
    """ Create a simple one-layer encoder mimic PCA.
        """

    def __init__(self, nC, dataset=[],  nVar=0,  trainable=False):
        'dataset and nVar must not both be default'
        super().__init__()
        self.nL = 1
        self.nC = nC
        self.pca = PCA(n_components=nC, random_state=0)
        if np.size(dataset) > 0:
            self.nVar = dataset.shape[1]
            self.pca.fit(dataset)
            print('Initialize PCA_Encoder using dataset.')
            print('Variance Percentages for each PC =')
            print(self.pca.explained_variance_ratio_[0:nC])
            self.init_encode_mat(self.nC, trainable=trainable)
        else:
            self.nVar = nVar
            self.encoder = nn.Linear(nVar, nC)
            self.decoder = nn.Linear(nC, nVar)
        for para in self.parameters():
            para.requires_grad = trainable

    def init_encode_mat(self, nPC, PCmatrix=None, trainable=False):
        if PCmatrix is None:
            ematrix = self.pca.components_[0:nPC, :]
        else:
            ematrix = PCmatrix[0:nPC, :]
        nVar = self.nVar
        wt = 1
        self.encoder = nn.Linear(nVar, nPC)
        self.decoder = nn.Linear(nPC, nVar)
        self.encoder.weight.data = torch.tensor(ematrix*wt,
                                                dtype=torch.float)
        enc_bias = -ematrix.dot(wt*self.pca.mean_)
        self.encoder.bias.data = torch.tensor(enc_bias).float()
        self.decoder.weight.data = torch.tensor(ematrix.T/wt,
                                                dtype=torch.float)
        self.decoder.bias.data = torch.tensor(self.pca.mean_).float()
        for para in self.parameters():
            para.requires_grad = trainable

    def encode(self, inputs):
        if type(inputs) is not torch.FloatTensor:
            inputs = torch.FloatTensor(inputs)
        inputs = inputs.view(-1, self.nVar)
        output_e = self.encoder(inputs)
        return output_e

    def decode(self, output_e):
        if type(output_e) is not torch.FloatTensor:
            output_e = torch.FloatTensor(output_e)
        output_e = output_e.view(-1, self.nC)
        output = self.decoder(output_e)
        return output

    def forward(self, inputs):
        shape = inputs.shape
        inputs = inputs.view(-1, self.nVar)
        output_e = self.encoder(inputs)
        output = self.decoder(output_e)
        output = output.view(*shape)
        return output_e, output

    def size(self):
        trainable_num = sum(p.numel()
                            for p in self.parameters() if p.requires_grad)
        return trainable_num


class AutoEncoder(nn.Module):
    """ A multilayer autoencoder with tied weights
    using recursive PCA initializer """

    def __init__(self, n_nodes, dataset_train=None, f_act=F_act):
        super().__init__()
        self.nL = n_nodes.size - 1
        self.nVar = n_nodes[0]
        self.nC = n_nodes[self.nL]
        self.F_act = f_act
        self.encoders = nn.ModuleList([nn.Linear(n_nodes[i], n_nodes[i+1])
                                       for i in range(n_nodes.size-1)])
        self.decoders = nn.ModuleList([nn.Linear(n_nodes[i+1], n_nodes[i])
                                       for i in range(n_nodes.size-1)])
        for i in range(self.nL):
            wt= torch.nn.Parameter(self.encoders[i].weight.data)
            torch.nn.init.xavier_uniform_(wt, gain=1.0)
            self.encoders[i].weight = wt
            self.decoders[i].weight = wt.transpose(0,1)
            self.encoders[i].bias.data = torch.zeros(1, n_nodes[i+1])
            self.decoders[i].bias.data = torch.zeros(1, n_nodes[i])

        if dataset_train is not None:
            pMats = []
            for i in range(n_nodes.size-1):
                pca = PCA(n_components=n_nodes[i+1])
                if i == 0:
                    pca.fit(dataset_train)
                    pca_data = dataset_train.dot(pca.components_.T)
                else:
                    pca.fit(pca_data)
                    pca_data = pca_data.dot(pca.components_.T)
                param = pca.components_
                pMats.append(param)

            for i in range(self.nL):
                self.encoders[i].weight.data = torch.tensor(pMats[i],
                                                            dtype=torch.float)
                self.encoders[i].bias.data = torch.zeros(1, n_nodes[i+1],
                                                         dtype=torch.float)
                self.decoders[i].weight.data = self.encoders[i].weight.data.T
                self.decoders[i].bias.data = torch.zeros(1, n_nodes[i],
                                                         dtype=torch.float)

    def encode(self, inputs):
        """ Used for encode a small amount of data, need not on GPU """
        if type(inputs) is not torch.Tensor:
            inputs = torch.tensor(inputs).float()
        inputs = inputs.view(-1, self.nVar)
        output_e = self.F_act(self.encoders[0](inputs))
        for i in range(1, self.nL-1):
            output_e = self.F_act(self.encoders[i](output_e))
        output_e = self.encoders[self.nL-1](output_e)
        output_e = output_e.clone().detach().cpu().numpy()
        return output_e

    def decode(self, output_e):
        if type(output_e) is np.ndarray:
            output_e = torch.tensor(output_e).float()
        output = output_e.view(-1, self.nC)
        for i in range(self.nL-1, 0, -1):
            output = self.F_act(self.decoders[i](output))
        output = self.decoders[0](output)
        return output

    def forward(self, inputs):
        shape = inputs.shape
        inputs = inputs.view(-1, self.nVar)
        output_e = self.F_act(self.encoders[0](inputs))
        for i in range(1, self.nL-1):
            output_e = self.F_act(self.encoders[i](output_e))
        output_e = self.encoders[self.nL-1](output_e)

        output = output_e
        for i in range(self.nL-1, 0, -1):
            output = self.F_act(self.decoders[i](output))
        output = self.decoders[0](output)
        output = output.view(*shape)
        return output_e, output

    def size(self):
        trainable_num = sum(p.numel()
                            for p in self.parameters() if p.requires_grad)
        return trainable_num


class SimpleAE(nn.Module):
    """ A multilayer autoencoder with tied weights
    using recursive PCA initializer """

    def __init__(self, n_nodes, f_act=F_act):
        super().__init__()
        self.nL = n_nodes.size - 1
        self.nVar = n_nodes[0]
        self.nC = n_nodes[self.nL]
        self.F_act = f_act
        self.encwt = nn.ParameterList([nn.Parameter(torch.zeros(n_nodes[i], n_nodes[i+1]))
                                       for i in range(n_nodes.size-1)])
        self.encb = nn.ParameterList([nn.Parameter(torch.zeros(1,n_nodes[i+1]))
                                       for i in range(n_nodes.size-1)])
        self.decb = nn.ParameterList([nn.Parameter(torch.zeros(1,n_nodes[i]))
                                       for i in range(n_nodes.size-1)])
        for i in range(self.nL):
            torch.nn.init.xavier_uniform_(self.encwt[i], gain=1.0)
            self.encb[i].data = torch.zeros(1, n_nodes[i+1])
            self.decb[i].data = torch.zeros(1, n_nodes[i])

    def encode(self, inputs):
        """ Used for encode a small amount of data, need not on GPU """
        if type(inputs) is not torch.Tensor:
            inputs = torch.tensor(inputs).float()
        output_e = inputs.view(-1, self.nVar)
        for i in range(0, self.nL-1):
            olin = torch.mm(output_e, self.encwt[i]) + self.encb[i]
            output_e = self.F_act(olin)
        i = self.nL - 1
        output_e = torch.mm(output_e, self.encwt[i]) + self.encb[i]
        output_e = output_e.clone().detach().cpu().numpy()
        return output_e

    def decode(self, output_e):
        if type(output_e) is np.ndarray:
            output_e = torch.tensor(output_e).float()
        output = output_e.view(-1, self.nC)
        for i in range(self.nL-1, 0, -1):
            olin = torch.mm(output, self.encwt[i].T) + self.decb[i]
            output = self.F_act(olin)
        i = 0
        output = torch.mm(output, self.encwt[i].T) + self.decb[i]
        return output

    def forward(self, inputs):
        shape = inputs.shape
        output_e = inputs.view(-1, self.nVar)
        for i in range(0, self.nL-1):
            olin = torch.mm(output_e, self.encwt[i]) + self.encb[i]
            output_e = self.F_act(olin)
        i = self.nL - 1
        output_e = torch.mm(output_e, self.encwt[i]) + self.encb[i]

        output = output_e
        for i in range(self.nL-1, 0, -1):
            olin = torch.mm(output, self.encwt[i].T) + self.decb[i]
            output = self.F_act(olin)
        i = 0
        output = torch.mm(output, self.encwt[i].T) + self.decb[i]
        output = output.view(*shape)
        return output_e, output

    def size(self):
        trainable_num = sum(p.numel()
                            for p in self.parameters() if p.requires_grad)
        return trainable_num


class PCA_ResNet(nn.Module):
    """ Deep AutoEncoder as nonlinar correction of
    stacked PCAs """

    def __init__(self, n_nodes, dataset_train=None, pca_trainable=True,
                 tied_weight=False,
                 gain1=0.01,
                 gain2=0.01,
                 f_act=F_act):
        super().__init__()
        self.nL = n_nodes.size - 1    # number of layers in encoder
        self.nVar = n_nodes[0]        # dimension of input variables
        self.nC = n_nodes[self.nL]    # dimension of hidden varaibles
        self.F_act = f_act
        self.pca_encs = nn.ModuleList([nn.Linear(n_nodes[i], n_nodes[i+1])
                                       for i in range(self.nL)])
        self.pca_decs = nn.ModuleList([nn.Linear(n_nodes[i+1], n_nodes[i])
                                       for i in range(self.nL)])
        pMats = []
        pbias = []
        wt = 1
        for i in range(self.nL):
            pca = PCA(n_components=n_nodes[i+1])
            if i == 0:
                pca.fit(dataset_train)
                pca_data = pca.transform(dataset_train)
            else:
                pca.fit(pca_data)
                pca_data = pca.transform(pca_data)
            param = pca.components_
            pMats.append(param)
            pbias.append(pca.mean_)

        self.pca_encs[0].weight.data = torch.tensor(
            pMats[0]*wt, dtype=torch.float)
        self.pca_encs[0].bias.data = torch.tensor(
            -pMats[0].dot(wt*pbias[0]), dtype=torch.float)
        self.pca_decs[0].weight.data = torch.tensor(
            pMats[0].transpose()/wt, dtype=torch.float)
        self.pca_decs[0].bias.data = torch.tensor(
            pbias[0], dtype=torch.float)
        for i in range(1, self.nL):
            self.pca_encs[i].weight.data = torch.tensor(
                pMats[i], dtype=torch.float)
            self.pca_encs[i].bias.data = torch.zeros(
                1, n_nodes[i+1], dtype=torch.float)
            self.pca_decs[i].weight.data = torch.tensor(
                pMats[i].transpose(), dtype=torch.float)
            self.pca_decs[i].bias.data = torch.zeros(
                1, n_nodes[i], dtype=torch.float)
        for para in self.parameters():
            para.requires_grad = pca_trainable

        self.pca_encs[0].weight.requires_grad = False
        self.pca_encs[0].bias.requires_grad = False
        self.pca_decs[0].weight.requires_grad = False
        self.pca_decs[0].bias.requires_grad = False

        self.nl_encs1 = nn.ModuleList([nn.Linear(n_nodes[i], n_nodes[i+1])
                                       for i in range(self.nL)])
        self.nl_encs2 = nn.ModuleList([nn.Linear(n_nodes[i+1], n_nodes[i+1])
                                       for i in range(self.nL)])
        self.nl_decs1 = nn.ModuleList([nn.Linear(n_nodes[i+1], n_nodes[i])
                                       for i in range(self.nL)])
        self.nl_decs2 = nn.ModuleList([nn.Linear(n_nodes[i+1], n_nodes[i+1])
                                       for i in range(self.nL)])
        for i in range(self.nL):
            if i==0:
                nn.init.xavier_uniform_(self.nl_encs1[i].weight, gain=0)
                nn.init.xavier_uniform_(self.nl_encs2[i].weight, gain=0)
            else:
                nn.init.xavier_uniform_(self.nl_encs1[i].weight, gain=gain1)
                nn.init.xavier_uniform_(self.nl_encs2[i].weight, gain=gain2)
            if tied_weight:
                self.nl_decs1[i].weight.data = self.nl_encs1[i].weight.T
                self.nl_decs2[i].weight.data = self.nl_encs2[i].weight.T
            else:
                if i==0:
                    nn.init.xavier_uniform_(self.nl_decs1[i].weight, gain=0)
                    nn.init.xavier_uniform_(self.nl_decs2[i].weight, gain=0)
                else:
                    nn.init.xavier_uniform_(self.nl_decs1[i].weight, gain=gain2)
                    nn.init.xavier_uniform_(self.nl_decs2[i].weight, gain=gain1)
            self.nl_encs1[i].bias.data = torch.zeros(1, n_nodes[i+1])
            self.nl_encs2[i].bias.data = torch.zeros(1, n_nodes[i+1])
            self.nl_decs1[i].bias.data = torch.zeros(1, n_nodes[i])
            self.nl_decs2[i].bias.data = torch.zeros(1, n_nodes[i+1])
        self.nl_encs1[0].weight.requires_grad = False
        self.nl_encs1[0].bias.requires_grad = False
        self.nl_decs1[0].weight.requires_grad = False
        self.nl_decs1[0].bias.requires_grad = False
        self.nl_encs2[0].weight.requires_grad = False
        self.nl_encs2[0].bias.requires_grad = False
        self.nl_decs2[0].weight.requires_grad = False
        self.nl_decs2[0].bias.requires_grad = False

    def encode(self, inputs):
        """ Used for encode a small amount of data, need not on GPU """
        if type(inputs) is not torch.Tensor:
            inputs = torch.tensor(inputs).float()
        inputs = inputs.view(-1, self.nVar)
        output_e = inputs
        for i in range(0, self.nL):
            pca_e = self.pca_encs[i](output_e)
            nl_e = self.F_act(self.nl_encs1[i](output_e))
            nl_e = self.nl_encs2[i](nl_e)
            output_e = pca_e + nl_e
        output_e = output_e.clone().detach().cpu().numpy()
        return output_e

    def decode(self, output_e):
        if type(output_e) is np.ndarray:
            output_e = torch.tensor(output_e).float()
        output = output_e.view(-1, self.nC)
        for i in reversed(range(self.nL)):
            pca_o = self.pca_decs[i](output)
            nl_o = self.F_act(self.nl_decs2[i](output))
            nl_o = self.nl_decs1[i](nl_o)
            output = pca_o + nl_o
        return output

    def forward(self, inputs):
        shape = inputs.shape
        inputs = inputs.view(-1, self.nVar)
        output_e = inputs
        for i in range(0, self.nL):
            pca_e = self.pca_encs[i](output_e)
            nl_e = self.F_act(self.nl_encs1[i](output_e))
            nl_e = self.nl_encs2[i](nl_e)
            output_e = pca_e + nl_e
        output = output_e
        for i in reversed(range(self.nL)):
            pca_o = self.pca_decs[i](output)
            nl_o = self.F_act(self.nl_decs2[i](output))
            nl_o = self.nl_decs1[i](nl_o)
            output = pca_o + nl_o
        output = output.view(*shape)
        return output_e, output

    def size(self):
        trainable_num = sum(p.numel()
                            for p in self.parameters() if p.requires_grad)
        return trainable_num


def mse_loss(outputs, imgs):
    """ define the MSE loss function """
    criterion = nn.MSELoss()
    mse = criterion(outputs, imgs)
    return mse


def train_mse(model, optimizer, data_train, epochs=1, data_test=None):
    ''' train an auto-encoder with MSE loss,
        but show relative MSE loss to compare with PCA
        '''
    log = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    if data_test is not None:
        data_test = data_test.to(device)

    for e in range(epochs):
        if data_test is not None:    # test first
            with torch.no_grad():
                _, testres = model(data_test)
                test_err = mse_loss_rel(testres, data_test)

        for i, imgs in enumerate(data_train):
            imgs = imgs.to(device)
            with torch.no_grad():
                _, outputs = model(imgs)
                loss_rel = mse_loss_rel(outputs, imgs)

            optimizer.zero_grad()
            _, outputs = model(imgs)
            loss = mse_loss(outputs, imgs)
            loss.backward()
            optimizer.step()

        if data_test is not None:    # test first
            log.append([e, loss_rel.detach().float(), test_err.float()])
        else:
            log.append([e, loss_rel.detach().float()])

        if e % 10 == 0 or e == epochs-1:
            if data_test is not None:
                print(f'epoch/epochs: {e}/{epochs} ',
                      f'mse: {loss_rel.item():.3e} ',
                      f'test: {test_err:.3e} ')
            else:
                print(f'epoch/epochs: {e}/{epochs} ',
                      f'mse: {loss_rel.item():.3e}')
    log = np.squeeze(np.array(log))
    return log
