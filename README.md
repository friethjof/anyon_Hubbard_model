# One-Dimensional Anyon Hubbard Model



## System

We consider the one-dimensional anyon-Hubbard in one-dimension. This system can be simulated considering two bosons in a deep optical lattice with density-dependent hopping amplitudes.

See also:

- J. Kwan *et al.*, "Realization of one-dimensional anyons with arbitrary statistical phase", Science **386** 1055 (2024), DOI:  [https://doi.org/10.1126/science.adi3252](https://doi.org/10.1126/science.adi3252)
- F. Theel, M. Bonkhoff, P. Schmelcher, T. Posske and N. L. Harshman, "Chirally Protected State Manipulation by Tuning One-Dimensional Statistics", Phys. Rev. Lett. **135** 063401(2025),  DOI: https://doi.org/10.1103/kzf6-yx24 





## Module

The programm solves the one-dimensional anyon-Hubbard model via exact diagonalization and, therefore, gives full access to the complete many-body wave-function.





## Install

Download and copy the program into a desired directory.

Change in the file /helper/path_dirs.py the parameters

- path_data = Path('../data_ahm')
- path_fig = Path(f'../figures_ahm')

to change the destination of data and figures that are stored and created while running the program.



Required modules:

​	



## Test

Execute the following line to test the code:

```bash
python -m unittest discover
```

During the test, for specific system settings the Hamiltonian is numerically created from scratch and compared to analytic derived solutions.
