# Anomalous Platelet Transport & Fat-Tailed Distributions

## Project

This project implements the statistical analysis presented in: Kotsalos et al.: Anomalous Platelet Transport & Fat-Tailed Distributions (arXiv: ). The python script (meta_process.py) is tailored on the tools developed by Kotsalos et al. (see: https://arxiv.org/abs/1911.03062, https://arxiv.org/abs/1903.06479). These tools for the direct numerical simulations of cellular blood flow will be available in future release of Palabos (https://palabos.unige.ch/).

## Execute the code

```python
python3 meta_process.py PLTs 1.0 distros 1 49 10 10 300 600 1 tBB 50 0.125 2
```

1. PLTs: refers to the corpuscle to analyse (RBCs or PLTs).
2. 1.0: is the sampling window in ms (see PLTs_tau_1.0).
3. distros: refers to the type of analysis to do (MFP,distros,MSD,distrFromWalls).
4. 1 49: wall-bounded direction, positions of walls in um.
5. 10 10: exclude this amount of um from the walls (avoid wall effects in the analysis).
6. 300 600: The DNS (Direct Numerical Simulations) was executed for 1000 ms physical time. The analysis is done between 300 and 600 ms.
7. 1: refers to the numbers of zones to split the domain.
8. tBB: top wall boundary condition.
9. 50: channel lateral dimensions in um
10. 0.125: DNS fluid time step in us according to the Lattice Boltzmann Method.
11. 2: number of cores to use in finding the tail lower bound (x_min)

## Few explanations

The python script reads the positions of PLTs (center of mass) through time (stored in PLTs_... directory). Every file in the directory stores the individual trajectory of a PLT in the form: (t,x,y,z). This is how our simulations are built:

1. t: refers to fluid time steps. It should be multiplied by the fluid time-step to return physical time
2. x,z: vorticity, flow directions respectively
3. y: wall-bounded direction
The above order can change with the right modifications.

The given PLTs directory inlcudes the positions of 95 platelets for a simulation executed for 1s physical time, in a box of 50^3 um^3, and constant shear rate 100 s^-1. The DNS are sampled at 1ms window and the numerical time step is 0.125 us according to the Lattice Boltzmann Method.

## License

Shield: [![CC BY 4.0][cc-by-shield]][cc-by]

This work is licensed under a [Creative Commons Attribution 4.0 International
License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg
