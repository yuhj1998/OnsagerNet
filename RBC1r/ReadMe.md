# ReadMe

This is an implementation of using OnsagerNet to learn Lorenz-like ODEs from Rayleigh-Bernard Convection(RBC)

## Source code files

1. `ode_net.py`: defines *OnsagerNet* and other baseline ODE nets and Runge-Kutta integrators.
2. `autoencoders.py`: defines several `autoencoder` including one proposed (PCA_ResNet) in the reference paper.
3. `ode_analyzer.py`: code to find fixed points and limit cycles of learned dynamics. See [GitHub/ode-analyzer](https://github.com/yuhj1998/ode-analyzer) for more information.
4. `config.py`: defines defaults parameters
5. `rbctools.py`: contains utility functions for RBC data loading and visualizations
6. `rbc_pca.py` : do PCA on given RBC dataset
7. `rbc_ode.py` : train the neural ODEs based on PCA data
7. `rbc_ae_ode.py` : train the neural ODEs and autoencoder simultaneously based on original data
8. `test_ode_RBC.py`: test the RBC dyanmics learned by OnsagerNet and other Neural ODE Nets
9. `test_ode_nn.py` and `paper_tabII.sh`: benchmark with neareast neighbor (NN) methods
10. `lstm.py`: used to generate data used for Table III in the reference paper. Note that, different to other files, this file uses `tensorflow` platform

Run files `paper_fig*.sh`, `paper_tab*.sh` to do similar tests as in the reference paper.

*Note that the results are only qualititive consistent to the results in the paper, due to the differences in hardware structure, software environment and randomness in SGD.*

## Data

The data file for relative Rayleigh number r=28 is available from:
[data@LSEC](http://lsec.cc.ac.cn/~hyu/research/share/RBC_r28L_T100R100.h5)

(Username: public,  Password:AcademicOnly)

Download the `RBC_r28L_T100R100.h5` file and put it in ./dataRBC/ subfolder 
before running any code for RBC problem.

More data (and the source code to generate data) will be available later.

## Reference

1. [PhyRevF] H. Yu, X. Tian, W. E and Q. Li, OnsagerNet: Learning Stable and Interpretable Dynamics using a Generalized Onsager Principle, [arxiv:2009.02327](https://arxiv.org/abs/2009.02327), to appear on *Physical Review Fluids*, 2021.
