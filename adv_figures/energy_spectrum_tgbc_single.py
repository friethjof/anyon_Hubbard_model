import os
import subprocess
from pathlib import Path
from fractions import Fraction

import numpy as np
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable

from helper import other_tools
from helper import path_dirs
from hamiltonian.solve_hamilt import AnyonHubbardHamiltonian


def make_plot():


    path_fig = path_dirs.adv_fig


    L = 6
    N = 2
    U = 1

    fig_name = f'energy_spectrum_tgbc_L{L}_N{N}_U{U}.pdf'


    theta_list = np.linspace(-np.pi, np.pi, 201)


    #---------------------------------------------------------------------------
    energy_lol, U_lol = [], []
    for theta in theta_list:
        hamil_class = AnyonHubbardHamiltonian(
            bc='twisted_gauge',
            L=L,
            N=N,
            J=1,
            U=U,
            theta=theta
        )
        diag_H, diag_U = hamil_class.diag_H_and_transl_op()



        energy_lol.append(diag_H)
        U_lol.append(diag_U)
    energy_lol = np.array(energy_lol)
    U_lol = np.array(U_lol)

    U_flat_degen = other_tools.find_degeneracies(U_lol.flatten())


    # retrieve all angles which occur in eigenvalues of u
    Ueval_angle = []
    for el in U_flat_degen.keys():
        el_angle = np.round(np.angle(eval(el))/np.pi, 8)
        if el_angle == -1.0:
            Ueval_angle.append(1.0)
        else:
            Ueval_angle.append(el_angle)
    Ueval_angle = np.array(list(sorted(Ueval_angle)))


    # each angle corresponds to a color code
    col_list = ['blue', 'red', 'orange', 'green', 'purple', 'brown']

    col_ind_lol = []
    for u_i in U_lol:
        col_ind_list = []
        for u_ii in u_i:
            u_ii_angle = np.round(np.angle(u_ii)/np.pi, 8)

            if u_ii_angle == -1.0:
                u_ii_angle = 1.0
            ind = np.argmin(np.abs(Ueval_angle - u_ii_angle))
            col_ind_list.append(ind)
        col_ind_lol.append(col_ind_list)
    col_ind_lol = np.array(col_ind_lol)


    N_Uvals = len(Ueval_angle)

    for i, el in enumerate(Ueval_angle):
        print(col_list[i], Fraction(el).limit_denominator(100))

    label_list = [
        r'$e^{-i2\phi \pi/3}$',
        r'$e^{-i\phi \pi/3}$',
        r'$1$',
        r'$e^{i\phi /3}$',
        r'$e^{i2\phi /3}$',
    ]

    print('# colors:', N_Uvals)



    #===========================================================================
    """Define grid structure"""
    plt.rc('text', usetex=True)
    fig = plt.figure(figsize=(8.3, 5), dpi=300)
    # fig.subplots_adjust(left=0.2, right=0.8, top=0.9, bottom=0.17)

    #split vertical
    canv = gridspec.GridSpec(1, 1)

    ax1 = plt.subplot(canv[0, 0])




    #===========================================================================
    for e_i, c_i in zip(energy_lol.T, col_ind_lol.T):
        for i, (e_ii, c_ii) in enumerate(zip(e_i, c_i)):

            ax1.scatter(theta_list[i]/np.pi, e_ii.real, c=col_list[c_ii], marker='o', s=5, linewidths=0)




    ax1.set_xlim(-1, 1)
    ax1.set_title(r'periodic boundary $U=0$')
    ax1.set_ylabel(r'$E$', fontsize=12, labelpad=3)
    ax1.set_xlabel(r'$\theta/\pi$', fontsize=12, labelpad=0)


    #===========================================================================
    path_fig.mkdir(parents=True, exist_ok=True)
    path_cwd = os.getcwd()
    os.chdir(path_fig)
    plt.savefig(fig_name)
    plt.close()

    # if fig_name[-3:] == 'png':
    #     subprocess.check_output(["convert", fig_name, "-trim", fig_name])
    # elif fig_name[-3:] == 'pdf':
    #     subprocess.check_output(["pdfcrop", fig_name, fig_name])

    os.chdir(path_cwd)
