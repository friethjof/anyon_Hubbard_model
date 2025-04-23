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

from helper import other_tools
from helper import path_dirs
from hamiltonian.solve_hamilt import AnyonHubbardHamiltonian


def make_plot():


    path_fig = path_dirs.adv_fig


    L = 6
    N = 2

    fig_name = f'energy_spectrum_L{L}_N{N}.pdf'


    theta_list = np.linspace(-np.pi, np.pi, 200)


    #---------------------------------------------------------------------------
    def get_energy_spectrum(bc, U):

        energy_lol = []
        for theta in theta_list:
            hamil_class = AnyonHubbardHamiltonian(
                bc=bc,
                L=L,
                N=N,
                J=1,
                U=U,
                theta=theta
            )
            H_eval = hamil_class.evals()
            dict_degen = other_tools.find_degeneracies(H_eval)
            for k, v in dict_degen.items():
                if len(v) > 1:
                    print(k, v)

            energy_lol.append(H_eval)
        energy_lol = np.array(energy_lol)
        return energy_lol


    E_open_U0 = get_energy_spectrum(bc='open', U=0)
    E_open_U1 = get_energy_spectrum(bc='open', U=1)
    E_twisted_U0 = get_energy_spectrum(bc='twisted', U=0)
    E_twisted_U1 = get_energy_spectrum(bc='twisted', U=1)



    #---------------------------------------------------------------------------
    # get data
    #---------------------------------------------------------------------------

    #===========================================================================
    """Define grid structure"""
    plt.rc('text', usetex=True)
    fig = plt.figure(figsize=(8.3, 5), dpi=300)
    # fig.subplots_adjust(left=0.2, right=0.8, top=0.9, bottom=0.17)

    #split vertical
    canv = gridspec.GridSpec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1],
                             wspace=0.2, hspace=0.2)

    ax1 = plt.subplot(canv[0, 0])
    ax2 = plt.subplot(canv[0, 1])
    ax3 = plt.subplot(canv[1, 0])
    ax4 = plt.subplot(canv[1, 1])




    #===========================================================================
    for ax in [ax1, ax2, ax3, ax4]:
        # ax.set_ylim(-0.5, 0.5)
        # ax.axvline(0.2, color='red', ls=':')
        ax.set_xlim(-1, 1)

    for e_i in E_open_U0.T:
        ax1.scatter(theta_list/np.pi, e_i, c='black', marker='o', s=0.5, linewidths=0)

    for e_i in E_twisted_U0.T:
        ax2.scatter(theta_list/np.pi, e_i, c='black', marker='o', s=0.5, linewidths=0)

    for e_i in E_open_U1.T:
        ax3.scatter(theta_list/np.pi, e_i, c='black', marker='o', s=0.5, linewidths=0)

    for e_i in E_twisted_U1.T:
        ax4.scatter(theta_list/np.pi, e_i, c='black', marker='o', s=0.5, linewidths=0)


    ax1.set_title(r'open boundary $U=0$')
    ax2.set_title(r'twisted boundary $U=0$')
    ax3.set_title(r'open boundary $U=1$')
    ax4.set_title(r'twisted boundary $U=1$')


    ax1.set_ylabel(r'$E$', fontsize=12, labelpad=3)
    ax3.set_ylabel(r'$E$', fontsize=12, labelpad=3)

    ax3.set_xlabel(r'$\theta/\pi$', fontsize=12, labelpad=0)
    ax4.set_xlabel(r'$\theta/\pi$', fontsize=12, labelpad=0)


    ax1.set_xticklabels([])
    ax2.set_xticklabels([])



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
