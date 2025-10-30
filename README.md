# One-Dimensional Anyon Hubbard Model



## System

We consider the one-dimensional anyon-Hubbard in one-dimension. This system can be simulated considering bosons in a deep optical lattice with density-dependent hopping amplitudes.

The corresponding Hamiltonian for open boundary condition reads,
$$
\hat{H} = -J\sum_{j=1}^{L-1}\left(\hat{b}_j^\dagger e^{i\theta \hat{n}_j} \hat{b}_{j+1} +  \mathrm{H.c.} \right) + \frac{U}{2} \sum_{j=1}^{L} \hat{n}_j\left(\hat{n}_j- 1\right).
$$
The system parameters are:

- $L$: number of lattice sites
- $N$: number of atoms
- $J$: hopping amplitude
- $U$: on-site interaction
- $\theta$: statistical parameter
- $\hat{b}_i^{(\dagger)}$: bosonic annihilation (creation) operator at site $i$
- $\hat{n}_i= \hat{b}_i^\dagger \hat{b}_i$: number operator at site $i$

For $\theta=0$ the classical Bose-Hubbard Hamiltonian is restored.

Periodic and twisted boundary conditions have been also implemented. 



See also:

- J. Kwan *et al.*, "Realization of one-dimensional anyons with arbitrary statistical phase", Science **386** 1055 (2024), DOI:  [https://doi.org/10.1126/science.adi3252](https://doi.org/10.1126/science.adi3252)
- F. Theel, M. Bonkhoff, P. Schmelcher, T. Posske and N. L. Harshman, "Chirally Protected State Manipulation by Tuning One-Dimensional Statistics", Phys. Rev. Lett. **135** 063401(2025),  DOI: https://doi.org/10.1103/kzf6-yx24 



## Module and Usage

The program solves the one-dimensional anyon-Hubbard model via exact diagonalization and, therefore, gives full access to all eigenstates of the Hamiltonian. 

The limit of the system size is set by the computational resources. The Hamiltonian is expressed in a number state basis and corresponds to a matrix which scales, like
$$
(\mathcal{N}\times\mathcal{N}), \hspace{0.5cm} \rm{where} \hspace{0.2cm} \mathcal{N} = {N+L-1 \choose L-1}.
$$



For an example how to setup the system and visualize the ground state, see **ground_state_example_notebook.ipynb**.






## Install

Download and copy the program into a desired directory.

Change in the file /helper/path_dirs.py the parameters

- path_data = Path('../data_ahm')
- path_fig = Path(f'../figures_ahm')

to change the destination of data and figures that are stored while running the program.



For required modules see the file *requirements.txt*, which has been created using the module **pipreqs **(can be installed with pip).





## Test

Execute the following line in the main directory to test the code:

```bash
python -m unittest discover
```

During the test, for specific system settings the Hamiltonian is numerically created from scratch and compared to analytic derived solutions.



Setup hooks for checking code before commit using **pre-commit**, hooks are written in the file  **.pre-commit-conig.yaml**

hooks

- **black** auto-formatting according to PEP8
- **flake8** linting for style violations and errors, report issues
- **MyPy** checks types, reports mismatches
