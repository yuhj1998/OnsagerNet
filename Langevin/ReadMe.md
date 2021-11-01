# ReadMe

This is an implementation of using OnsagerNet to learn Langevin dynamics.

## Source code files

1. `ode_net.py`: defines *OnsagerNet* and other baseline ODE nets and Runge-Kutta integrators
2. `test_ode_Langevin.py`: to generate Langevin dynamical data and learn the dyanmics using Neural ODE Nets
3. `test_ode_error.py`: similar to `test_ode_Langevin.py` by without the training process, load trained model from disk.
4. `viztools.py`: contains utility functions for visualization
5. `plot_*.py` : used to plot figures in the reference paper.

Run files `paper_fig?.sh` **one by one** to do similar tests as in the reference paper.

*Note that the results are only qualititive consistent to the results in the paper, due to the differences in hardware structure, software environment and randomness in SGD.*

## Reference

1. [PhyRevF] H. Yu, X. Tian, W. E and Q. Li, OnsagerNet: Learning Stable and Interpretable Dynamics using a Generalized Onsager Principle, [arxiv:2009.02327](https://arxiv.org/abs/2009.02327), to appear on *Physical Review Fluids*, 2021.
