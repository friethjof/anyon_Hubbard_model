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

    fig_name = f'energy_spectrum_obc_thpi_L{L}_N{N}.pdf'


    U_list = np.linspace(0, 6, 401)


    #---------------------------------------------------------------------------
    # get energy list and eigenvalues of U
    energy_lol, U_lol = [], []
    for U in U_list:
        hamil_class = AnyonHubbardHamiltonian(
            bc='open',
            L=L,
            N=N,
            J=1,
            U=U,
            theta=np.pi,
        )
        diag_H, diag_U = hamil_class.diag_H_and_U_I_op()
        assert np.max(np.abs(diag_H.imag)) < 1e-10
        energy_lol.append(diag_H.real)
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
        elif np.abs(el_angle) < 1e-10:
            Ueval_angle.append(0.0)
        else:
            Ueval_angle.append(el_angle)
    Ueval_angle = np.array(list(sorted(Ueval_angle)))


    # group energies and eigenvalues of U
    U_dict_lol = {str(el):[] for el in Ueval_angle}
    for i, (e_list, u_list) in enumerate(zip(energy_lol, U_lol)):
        u_dict = other_tools.find_degeneracies(u_list)

        # check if in dict
        for u_key, u_ind_list in u_dict.items():
            u_angle = np.round(np.angle(eval(u_key))/np.pi, 8)
            if u_angle == -1.0:
                u_angle = 1.0
            elif np.abs(u_angle) < 1e-10:
                u_angle = 0.0

            u_angle_lol = U_dict_lol[str(u_angle)]
            u_angle_lol.append(sorted([e_list[i] for i in u_ind_list]))


    U_dict_mat = {}
    for u_key, u_lol in U_dict_lol.items():
        U_dict_mat[u_key] = np.array(u_lol)




    #===========================================================================
    """Define grid structure"""
    plt.rc('text', usetex=True)
    fig = plt.figure(figsize=(8.3/2, 2.2), dpi=300)
    fig.subplots_adjust(left=0.15, right=0.87, top=0.85, bottom=0.18)

    #split vertical
    canv = gridspec.GridSpec(1, 2, width_ratios=[1, 0.7], wspace=0.3)

    canv_r = gridspec.GridSpecFromSubplotSpec(2, 1, canv[0, 1],
                                                  height_ratios=[1, 1], hspace=0.4)


    ax1 = plt.subplot(canv[0, 0])
    ax2 = plt.subplot(canv_r[0, 0])
    ax3 = plt.subplot(canv_r[1, 0])





    #===========================================================================
    # top
    #===========================================================================
    styles = [
        {'marker':'o', 'linewidths':0, 'color':'cornflowerblue', 'zorder':3},
        {'marker':'o', 'linewidths':0, 'color':'tomato', 'zorder':2},
        {'marker':'o', 'linewidths':0, 'color':'gold'},
        {'marker':'o', 'linewidths':0, 'color':'forestgreen'},
        {'marker':'o', 'linewidths':0, 'color':'orange'},
        {'marker':'o', 'linewidths':0, 'color':'dimgray'},
    ]

    l_list = []
    for i, (u_key, u_mat) in enumerate(U_dict_mat.items()):
        for u_arr in u_mat.T:
            ax1.scatter(U_list, u_arr, **styles[i], s=1)
            ax2.scatter(U_list, u_arr, **styles[i], s=2)
            ax3.scatter(U_list, u_arr, **styles[i], s=2)

            l1 = plt.scatter([], [], **styles[i], s=5)

        l_list.append(l1)
        # ax4.scatter(theta_list/np.pi, e_i, c='black', marker='o', s=0.5, linewidths=0)
        # ax5.scatter(theta_list/np.pi, e_i, c='black', marker='o', s=0.5, linewidths=0)
        # ax6.scatter(theta_list/np.pi, e_i, c='black', marker='o', s=0.5, linewidths=0)


        print(Fraction(eval(u_key)).limit_denominator(100), np.exp(1j*eval(u_key)*np.pi))



    ax1.set_xlim(-0.1, 6.1)

    ax2.set_xlim(1.2, 2.2)
    ax2.set_ylim(1, 2.5)


    ax3.set_xlim(1.7, 2.7)
    ax3.set_ylim(-1.8, -1.5)

    inset_indicator = ax1.indicate_inset_zoom(ax2, edgecolor='black', lw=0.5, alpha=1)
    inset_indicator[1][0].set(lw=0.5)
    inset_indicator[1][1].set(lw=0.5)
    inset_indicator[1][2].set(lw=0.5)
    inset_indicator[1][3].set(lw=0.5)
    inset_indicator = ax1.indicate_inset_zoom(ax3, edgecolor='black', lw=0.5, alpha=1)
    inset_indicator[1][0].set(lw=0.5)
    inset_indicator[1][1].set(lw=0.5)
    inset_indicator[1][2].set(lw=0.5)
    inset_indicator[1][3].set(lw=0.5)
    # -2/3
    # -1/3
    # 0
    # 1/3
    # 2/3
    # 1

    ax1.legend(
        l_list,
        (r'$1$',
         r'$-1$',),
        bbox_to_anchor=(1, 1.1), ncol=7, loc='center', fontsize=8,
        framealpha=0.9, borderpad=0.2, handlelength=1.5, handletextpad=0.3,
        labelspacing=0.2, columnspacing=0.5, frameon=True
    )

    #===========================================================================
    # for ax in [ax1]:
        # ax.set_xlim(-1, 1)
        # ax.set_ylim(-4.5, 4.5)


    for ax in [ax2, ax3]:
        ax.yaxis.tick_right()
        ax.tick_params(labelsize=10)



    ax1.set_ylabel(r'$E_i$', fontsize=12, labelpad=3)
    ax1.set_xlabel(r'$U/J$', fontsize=12, labelpad=0)
    ax3.set_xlabel(r'$U/J$', fontsize=12, labelpad=0)




    ax1.annotate('(a)', fontsize=10, xy=(-0.06, 1.05), xycoords='axes fraction')
    ax2.annotate('(b)', fontsize=10, xy=(-0.17, 0.65), xycoords='axes fraction')
    ax3.annotate('(c)', fontsize=10, xy=(-0.17, 0.65), xycoords='axes fraction')


    #===========================================================================
    path_fig.mkdir(parents=True, exist_ok=True)
    path_cwd = os.getcwd()
    os.chdir(path_fig)
    plt.savefig(fig_name)
    plt.close()

    if fig_name[-3:] == 'png':
        subprocess.check_output(["convert", fig_name, "-trim", fig_name])
    elif fig_name[-3:] == 'pdf':
        subprocess.check_output(["pdfcrop", fig_name, fig_name])

    os.chdir(path_cwd)
