import os
import subprocess
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec



from helper import path_dirs


def make_plot():

    path_fig = path_dirs.adv_fig


    L = 100
    N = 2
    U = 0

    U_ = f'U{round(U,8)}'.replace('.', '_')

    #---------------------------------------------------------------------------
    fig_name = f'level_statistics_obc_L{L}_N{N}_{U_}.pdf'

    # theta=0,pi take only eigenvalues which have the same eigenvalue of U*I which is 1


    #---------------------------------------------------------------------------
    data_dir = Path('/afs/physnet.uni-hamburg.de/users/zoq_t/ftheel/'
                    'Schreibtisch/project_ahm2/level_statistics/data_obc')

    E_max = 5

    def read_in_data(data_name):
        with open(data_dir/data_name,'r') as f:
            text = f.read().replace('{', '[').replace('}', ']')
            eval_m = eval(text.split('=')[1])
        return eval_m


    eval_1 = read_in_data(f'L{L}_N{N}_{U_}_thpi_0_0_histogramData.m')
    eval_2 = read_in_data(f'L{L}_N{N}_{U_}_thpi_0_2_histogramData.m')
    eval_3 = read_in_data(f'L{L}_N{N}_{U_}_thpi_0_5_histogramData.m')
    eval_4 = read_in_data(f'L{L}_N{N}_{U_}_thpi_0_8_histogramData.m')
    eval_5 = read_in_data(f'L{L}_N{N}_{U_}_thpi_1_0_histogramData.m')


    N = len(eval_1)

    x_vals = np.arange(0, E_max+1e-9, E_max/N)[:-1] + E_max/N/2
    x_fine = np.arange(0, E_max+1e-9, 1e-3)


    def GOE():
        y_vals = np.pi/2 * x_fine *np.exp(-np.pi/4 * x_fine**2 )
        return x_fine, y_vals


    def Poisson():
        return x_fine, np.exp(-x_fine)


    #===========================================================================
    """Define grid structure"""
    plt.rc('text', usetex=True)
    fig = plt.figure(figsize=(8.3, 1.7), dpi=300)
    fig.subplots_adjust(left=0.09, right=0.92, top=0.9, bottom=0.2)


    canv = gridspec.GridSpec(1, 5, width_ratios=[1, 1, 1, 1, 1], wspace=0.2)


    ax1 = plt.subplot(canv[0, 0])
    ax2 = plt.subplot(canv[0, 1])
    ax3 = plt.subplot(canv[0, 2])
    ax4 = plt.subplot(canv[0, 3])
    ax5 = plt.subplot(canv[0, 4])



    #===========================================================================
    # level statistics
    #===========================================================================

    def plot_statistics(ax, eval_m):

        ax.set_xlim(-0.08, 5)
        ax.set_ylim(0, 1)

        ax.bar(x_vals, eval_m, width=0.18, color='lightskyblue', 
               edgecolor='black', linewidth=0.5)

        ax.plot(*GOE(), color='white', lw=1.8)
        ax.plot(*GOE(), color='orange', lw=1.5)
        ax.plot(*Poisson(), color='white', lw=1.8)
        ax.plot(*Poisson(), color='tomato', lw=1.5)

        ax.set_xlabel(r'$\Delta E$', fontsize=12, labelpad=0)


    plot_statistics(ax1, eval_1)
    plot_statistics(ax2, eval_2)
    plot_statistics(ax3, eval_3)
    plot_statistics(ax4, eval_4)
    plot_statistics(ax5, eval_5)

    ax1.set_ylabel(r'Probability density', fontsize=12)


    ax2.set_yticklabels([])
    ax3.set_yticklabels([])
    ax4.set_yticklabels([])
    ax5.set_yticklabels([])

    ax1.annotate(r'$\theta=0$', fontsize=10, xy=(0.67, 0.83), xycoords='axes fraction')
    ax2.annotate(r'$\theta=0.2\pi$', fontsize=10, xy=(0.52, 0.83), xycoords='axes fraction')
    ax3.annotate(r'$\theta=\pi/2$', fontsize=10, xy=(0.55, 0.83), xycoords='axes fraction')
    ax4.annotate(r'$\theta=0.8\pi$', fontsize=10, xy=(0.52, 0.83), xycoords='axes fraction')
    ax5.annotate(r'$\theta=\pi$', fontsize=10, xy=(0.67, 0.83), xycoords='axes fraction')

    # ax3.annotate(r'$L=60$, $N=2$', fontsize=10, xy=(0.15, 1.05), xycoords='axes fraction')



    #===========================================================================
    ax1.annotate('(a)', fontsize=10, xy=(0.07, 0.86), xycoords='axes fraction')
    ax2.annotate('(b)', fontsize=10, xy=(0.07, 0.86), xycoords='axes fraction')
    ax3.annotate('(c)', fontsize=10, xy=(0.07, 0.86), xycoords='axes fraction')
    ax4.annotate('(d)', fontsize=10, xy=(0.07, 0.86), xycoords='axes fraction')
    ax5.annotate('(e)', fontsize=10, xy=(0.07, 0.86), xycoords='axes fraction')

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
