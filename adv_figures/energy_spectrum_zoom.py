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

    fig_name = f'energy_spectrum_zoom_L{L}_N{N}.pdf'


    theta_list = np.linspace(-np.pi, np.pi, 1601)


    #---------------------------------------------------------------------------
    def get_energy_spectrum(bc, U):

        degen_dict_coords = {}  # collect coordinates for degeneracies
        # {degeneracy:[[theta, eval], ...]}

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
            if np.abs(theta) < 0.4:
                thres = 1e-10
            else:
                thres = 1e-3


            dict_degen = other_tools.find_degeneracies(H_eval, threshold=thres)
            print(theta)
            for k, v in dict_degen.items():
                
                d = len(v)
                if len(v) > 1:
                    print(k, v)
                    d_key = f'{d}'
                    if d_key in degen_dict_coords.keys():
                        degen_dict_coords[d_key].append([theta, eval(k)])
                    else:
                        degen_dict_coords[d_key] = [[theta, eval(k)]]


            energy_lol.append(H_eval)
        energy_lol = np.array(energy_lol)

        degen_dict_coords_out = {}
        for d, d_lol in degen_dict_coords.items():
            d_mat = np.array(d_lol)
            degen_dict_coords_out[d] = np.array([d_mat[:, 0]/np.pi, d_mat[:, 1]])

        return energy_lol, degen_dict_coords_out


    E_open_U0, d_dict_open_U0 = get_energy_spectrum(bc='open', U=0)
    E_twisted_U0, d_dict_twisted_U0 = get_energy_spectrum(bc='twisted', U=0)



    #---------------------------------------------------------------------------
    # get data
    #---------------------------------------------------------------------------

    #===========================================================================
    """Define grid structure"""
    plt.rc('text', usetex=True)
    fig = plt.figure(figsize=(8.3/2, 4), dpi=300)
    # fig.subplots_adjust(left=0.2, right=0.8, top=0.9, bottom=0.17)

    #split vertical
    canv = gridspec.GridSpec(2, 1, height_ratios=[1, 1], hspace=0.2)

    canv_top = gridspec.GridSpecFromSubplotSpec(1, 2, canv[0, 0],
                                                width_ratios=[1, 0.7], wspace=0.3)
    canv_bot = gridspec.GridSpecFromSubplotSpec(1, 2, canv[1, 0],
                                                width_ratios=[1, 0.7], wspace=0.3)

    canv_top_r = gridspec.GridSpecFromSubplotSpec(2, 1, canv_top[0, 1],
                                                  height_ratios=[1, 1], hspace=0.4)

    canv_top_l = gridspec.GridSpecFromSubplotSpec(2, 1, canv_bot[0, 1],
                                                  height_ratios=[1, 1], hspace=0.4)

    ax1 = plt.subplot(canv_top[0, 0])
    ax2 = plt.subplot(canv_top_r[0, 0])
    ax3 = plt.subplot(canv_top_r[1, 0])
    ax4 = plt.subplot(canv_bot[0, 0])
    ax5 = plt.subplot(canv_top_l[0, 0])
    ax6 = plt.subplot(canv_top_l[1, 0])



    d2_style = {'marker':'s', 's':7, 'linewidths':0, 'color':'tomato'}
    d3_style = {'marker':'*', 's':9, 'linewidths':0, 'color':'cornflowerblue'}
    d5_style = {'marker':'D', 's':9, 'linewidths':0, 'color':'forestgreen'}


    #===========================================================================
    # top
    #===========================================================================
    ax1.scatter(*d_dict_open_U0['3'], **d3_style)
    ax2.scatter(*d_dict_open_U0['3'], **d3_style)
    ax3.scatter(*d_dict_open_U0['3'], **d3_style)

    for e_i in E_open_U0.T:
        ax1.scatter(theta_list/np.pi, e_i, zorder=0, c='black', marker='o', s=0.5, linewidths=0)
        ax2.scatter(theta_list/np.pi, e_i, zorder=0, c='black', marker='o', s=0.5, linewidths=0)
        ax3.scatter(theta_list/np.pi, e_i, zorder=0, c='black', marker='o', s=0.5, linewidths=0)


    ax2.set_xlim(0.9, 1.0)
    ax2.set_ylim(2.2, 2.4)

    ax3.set_xlim(0.65, 0.75)
    ax3.set_ylim(-2, -1.8)

    ax1.indicate_inset_zoom(ax2, edgecolor='dimgray')
    ax1.indicate_inset_zoom(ax3, edgecolor='dimgray')


    #===========================================================================
    # bot
    #===========================================================================
    for e_i in E_twisted_U0.T:
        ax4.scatter(theta_list/np.pi, e_i, c='black', marker='o', s=0.5, linewidths=0)
        ax5.scatter(theta_list/np.pi, e_i, c='black', marker='o', s=0.5, linewidths=0)
        ax6.scatter(theta_list/np.pi, e_i, c='black', marker='o', s=0.5, linewidths=0)





    ax4.scatter(*d_dict_twisted_U0['2'], **d2_style)
    ax5.scatter(*d_dict_twisted_U0['2'], **d2_style)
    l1 = ax6.scatter(*d_dict_twisted_U0['2'], **d2_style)

    ax4.scatter(*d_dict_twisted_U0['3'], **d3_style)
    ax5.scatter(*d_dict_twisted_U0['3'], **d3_style)
    l2 = ax6.scatter(*d_dict_twisted_U0['3'], **d3_style)

    ax4.scatter(*d_dict_twisted_U0['5'], **d5_style)
    ax5.scatter(*d_dict_twisted_U0['5'], **d5_style)
    l3 = ax6.scatter(*d_dict_twisted_U0['5'], **d5_style)


    ax5.set_xlim(0.38, 0.48)
    ax5.set_ylim(2.7, 3.1)

    ax6.set_xlim(0.35, 0.45)
    ax6.set_ylim(-1.95, -1.85)

    ax4.indicate_inset_zoom(ax5, edgecolor='dimgray')
    ax4.indicate_inset_zoom(ax6, edgecolor='dimgray')



    ax1.legend((l1, l2, l3),
               (r'$d_0=2$', r'$3$', r'$5$'),
        bbox_to_anchor=(0.5, 1.1), ncol=7, loc='center', fontsize=8,
        framealpha=0.9, borderpad=0.2, handlelength=1.5, handletextpad=0.3,
        labelspacing=0.2, columnspacing=0.5, frameon=True
    )


    #===========================================================================
    for ax in [ax1, ax4]:
        # ax.set_ylim(-0.5, 0.5)
        # ax.axvline(0.2, color='red', ls=':')
        ax.set_xlim(-1, 1)


    for ax in [ax2, ax3, ax5, ax6]:
        ax.yaxis.tick_right()
        ax.tick_params(labelsize=10)
    # ax1.set_title(r'open boundary $U=0$')
    # ax2.set_title(r'twisted boundary $U=0$')
    # ax3.set_title(r'open boundary $U=1$')
    # ax4.set_title(r'twisted boundary $U=1$')


    # ax1.set_ylabel(r'$E$', fontsize=12, labelpad=3)
    # ax3.set_ylabel(r'$E$', fontsize=12, labelpad=3)

    ax4.set_xlabel(r'$\theta/\pi$', fontsize=12, labelpad=0)
    ax6.set_xlabel(r'$\theta/\pi$', fontsize=12, labelpad=0)


    ax1.set_xticklabels([])
    # ax2.set_xticklabels([])


    ax1.annotate('(a)', fontsize=10, xy=(-0.06, 1.05), xycoords='axes fraction')
    ax2.annotate('(b)', fontsize=10, xy=(-0.17, 0.65), xycoords='axes fraction')
    ax3.annotate('(c)', fontsize=10, xy=(-0.17, 0.65), xycoords='axes fraction')
    ax4.annotate('(d)', fontsize=10, xy=(-0.06, 1.05), xycoords='axes fraction')
    ax5.annotate('(e)', fontsize=10, xy=(-0.17, 0.65), xycoords='axes fraction')
    ax6.annotate('(f)', fontsize=10, xy=(-0.17, 0.65), xycoords='axes fraction')



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
