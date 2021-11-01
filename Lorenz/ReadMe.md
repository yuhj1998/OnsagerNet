# ReadMe

This is an implementation of using OnsagerNet to learn Lorenz 63 systems.

## Source code files

1. `ode_net.py`: defines *OnsagerNet* and other baseline ODE nets and Runge-Kutta integrators
2. `test_ode_Lorenz.py`: to generate Langevin dynamical data and learn the dyanmics using Neural ODE Nets
3. `test_ode_error.py`: similar to `test_ode_Lorenz.py` by without the training process, load trained model from disk.
4. `ode_analyzer.py`: code to find fixed points and limit cycles of learned dynamics. See [GitHub/ode-analyzer](https://github.com/yuhj1998/ode-analyzer) for more information.
5. `viztools.py`: contains utility functions for visualization
6. `plot_*.py` : used to plot figures in the reference paper.

Run files `paper_fig?.sh` **one by one** to do similar tests as in the reference paper.

*Note that the results are only qualititive consistent to the results in the paper, due to the differences in hardware structure, software environment and randomness in SGD.*

## Reference

1. [PhyRevF] H. Yu, X. Tian, W. E and Q. Li, OnsagerNet: Learning Stable and Interpretable Dynamics using a Generalized Onsager Principle, [arxiv:2009.02327](https://arxiv.org/abs/2009.02327), to appear on *Physical Review Fluids*, 2021.
