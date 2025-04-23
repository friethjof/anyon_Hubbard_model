import os
import subprocess
from pathlib import Path

import numpy as np
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable

from helper import path_dirs
from hamiltonian.solve_hamilt import AnyonHubbardHamiltonian
from propagation.propagate_static import PropagationStatic



def make_test():
    """test 'randomness':

    so far: for the correlation hole, we create random states from random
    superpositions of eigenstates.
    since the spectrum is symmetric around E=0, we the contributions cancel and
    the random state has an energy close to 0.
    
    However, for different theta there seems to be a shift:
    theta = 0    : energy distr centered around 0
    theta = 0.2pi: energy distr centered around a value slighty below 0
    theta = 0.6pi: energy distr centered around a value slighty above 0

    Conjecture: Is this due the fact that we use complex coeff and this 
                will resolve once we use real coeffs?

    """


    bc = 'open'
    L = 40
    N = 2
    U = 0.0


    hamil_class_1 = AnyonHubbardHamiltonian(
        bc = bc,
        J = 1,
        L = L,
        N = N,
        U = U,
        theta = 0,
        bool_save=True,
    )
    ini_size = hamil_class_1.basis.length

    hamil_class_2 = AnyonHubbardHamiltonian(
        bc = bc,
        J = 1,
        L = L,
        N = N,
        U = U,
        theta = 0.2*np.pi,
        bool_save=True,
    )

    hamil_class_3 = AnyonHubbardHamiltonian(
        bc = bc,
        J = 1,
        L = L,
        N = N,
        U = U,
        theta = 1*np.pi,
        bool_save=True,
    )

    e_list_1, e_list_2, e_list_3 = [], [], []
    for i in range(1000):

        rand_populations = (
            np.random.uniform(0, 1, ini_size)
            + 1.j * np.random.uniform(0, 1, ini_size)
        )


        # plt.scatter(range(ini_size), np.abs(rand_populations), s=1)
        # ind_0 = np.where(np.abs(hamil_class_1.evals()) < 1e-10)
        # rand_populations[ind_0] = 0


        #-----------------------------------------------------------------------
        # energy 1
        rand_vec_1 = hamil_class_1.evecs().T.dot(rand_populations)
        rand_vec_1 /= np.sqrt(np.vdot(rand_vec_1, rand_vec_1))

        energy_1 = np.vdot(rand_vec_1, hamil_class_1.hamilt().dot(rand_vec_1)).real

        #-----------------------------------------------------------------------
        # energy 2
        rand_vec_2 = hamil_class_2.evecs().T.dot(rand_populations)
        rand_vec_2 /= np.sqrt(np.vdot(rand_vec_2, rand_vec_2))

        energy_2 = np.vdot(rand_vec_2, hamil_class_2.hamilt().dot(rand_vec_2)).real

        #-----------------------------------------------------------------------
        # energy 3
        rand_vec_3 = hamil_class_3.evecs().T.dot(rand_populations)
        rand_vec_3 /= np.sqrt(np.vdot(rand_vec_3, rand_vec_3))

        energy_3 = np.vdot(rand_vec_3, hamil_class_3.hamilt().dot(rand_vec_3)).real


        e_list_1.append(energy_1)
        e_list_2.append(energy_2)
        e_list_3.append(energy_3)


    # plt.show()
    # plt.close()

    plt.hist(e_list_1, bins=50, color='tomato')
    plt.hist(e_list_2, bins=50, color='cornflowerblue')
    plt.hist(e_list_3, bins=50, color='forestgreen')

    plt.xlabel('energy')
    plt.ylabel('bin')


    plt.show()

    # print(e_list)
    # fidelity_av, rand_vec = get_fidelity(U=0, theta=0)
    # plt.plot(fidelity_av)
    # plt.show()
    # exit()