#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@File 	 : config.py
@Time 	 : 2020/07/1
@Author  : Haijn Yu <hyu@lsec.cc.ac.cn>
@Desc	 : Define global parameters used by different data files.
              We define a case_id for each datafile, the other
              parameters all depends on this case_id.
              Some parameters are obtained by fine tunning or data analysis
'''

import numpy as np

DEFAULT_CASE_ID = 1
DATA_DIR = '../dataRBC/'
OUTPUT_DIR = './results/'

float_formatter = "{:.6e}".format
np.set_printoptions(formatter={'float_kind': float_formatter})


class ONetConfig:
    def __init__(self,
                 case_id,
                 rcase,
                 rRa,               # scaled Rayleigh number
                 nTraj=100,         # number of Trajectories in data
                 dt=0.001,          # time stepsize of PDE sovler
                 nPC=7,             # number of variables for hidden ODEs
                 wt_factor=40,      # a factor to scale primanry components
                 tr_ratio=8/10,     # ratio of trainning data to all data
                 ts=0,              # number of snapshot to trim from start
                 te=0,
                 iNodeC=2,          # determin number of hidden nodes
                 ode_fid=0,         # activation function 0=ReQUr
                 pot_beta=0.1,      # coefficient of quadratic in potential
                 ons_min_d=0.1,     # minimum dissipation in OnsagerNet
                 batch_size=200,
                 lr_ode=0.005,
                 lr_e2e=0.0005,
                 epochs=300,
                 patience=20,
                 e2e_patience=20,
                 wt_decay=0e-5,
                 nt=1,
                 Paths=[0, ],
                 azim=60,
                 elev=30,
                 iseed=0
                 ):
        # data related parameters
        self.case_id = case_id
        self.rcase = rcase
        self.rRa = rRa
        self.nTraj = nTraj
        self.dt = dt
        self.wt_factor = wt_factor
        self.tr_ratio = tr_ratio
        self.ts = ts
        self.te = te

        # net structure related
        self.nPC = nPC
        self.iNodeC = iNodeC
        self.ode_fid = ode_fid
        self.pot_beta = pot_beta
        self.ons_min_d = ons_min_d

        # trainning parameters
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr_ode = lr_ode
        self.lr_e2e = lr_e2e
        self.patience = patience
        self.e2e_patience = e2e_patience
        self.wt_decay = wt_decay
        self.nt = nt

        # Others
        self.iseed = iseed
        self.Paths = Paths
        self.azim = azim
        self.elev = elev

        # other dependent variables
        self.h5fname = DATA_DIR + rcase
        self.outloc = OUTPUT_DIR + rcase

    def calc_patience(self, lr, lr_min, epoch):
        return int(epoch/4/(1+np.log2(lr/lr_min)))

    def get_ode_nodes(self, nPC, nHnode, nL, onet):
        if onet == 'ons':
            ode_nodes = np.array([nPC, ]+[nHnode, ]*nL, dtype=int)
        else:
            nOnsDoF = (nHnode+1)*(nPC+1)**2 + nHnode*(nHnode+1)*(nL-1)
            b = 2*nPC + nL + 1
            a = nL
            n = ( np.sqrt(b**2+4*a* nOnsDoF) - b ) / float(2*a)
            ode_nodes = np.array([nPC, ]+[int(n), ]*(nL+1), dtype=int)
        return ode_nodes

    def get_ae_nodes(self, nVar, nPC):
        ae_nodes = np.array([nVar, 128, 32, nPC], dtype=int)
        return ae_nodes

    def print(self):
        ''' Print the global configure variables
            @todo print all variables
         '''
        print(f'h5fname = {self.h5fname}')
        print(f'outloc = {self.outloc}', flush=True)
# end of ONetConfig class


configs = {
    0: ONetConfig(case_id=0,
                  rcase='RBC_LorenzI2_T100R20',
                  rRa=28,
                  nTraj=20,
                  dt=0.001,
                  nPC=5,
                  Paths=[1, 18, 5, 12, 2, 6],
                  azim=-98,
                  elev=22
                  ),
    1: ONetConfig(case_id=1,
                  rcase='RBC_r28L_T100R100',
                  rRa=28,
                  nTraj=100,
                  dt=0.001,
                  ts=0,
                  te=0,
                  ode_fid=0,
                  pot_beta=0.1,      # coefficient of quadratic in potential
                  ons_min_d=0.1,     # minimum dissipation in OnsagerNet
                  batch_size=200,
                  lr_ode=0.0064,
                  lr_e2e=0.0016,
                  epochs=600,
                  wt_decay=0e-5,
                  patience=25,
                  e2e_patience=25,
                  azim=-75,
                  elev=32,
                  Paths=[91, 85, 2, 86, 5, 82]
                  ),
    2: ONetConfig(case_id=2,
                  rcase='RBC_r84L_T400R100',
                  rRa=84,
                  nTraj=100,
                  dt=0.001,
                  ts=25,
                  te=175,
                  ode_fid=0,
                  pot_beta=0.1,      # coefficient of quadratic in potential
                  ons_min_d=0.1,     # minimum dissipation in OnsagerNet
                  batch_size=200,
                  lr_ode=0.008,
                  lr_e2e=0.0004,
                  epochs=600,
                  wt_decay=1e-5,
                  patience=25,
                  e2e_patience=25,
                  Paths=[85, 87, 88, 12],
                  azim=100,
                  elev=10,
                  ),
    3: ONetConfig(case_id=3,
                  rcase='RBC_test',
                  rRa=84,
                  nTraj=100,
                  dt=0.001,
                  ts=0,
                  te=0,
                  ),
}


def get_test_case(case_id=DEFAULT_CASE_ID):
    return configs[case_id]
