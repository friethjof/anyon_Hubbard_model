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
from propagation.scan_schmidt_echo import ScanClass_SchmidtEcho

def make_plot():


    path_fig = path_dirs.adv_fig


    L = 10
    N = 2
    U = 1
    # bc = 'twisted_gauge'
    bc = 'open'
    N_rand = 1000
    # N_rand = 'nstate'
    t_bin_size = 10
    t_bin_delta = 0.2

    av_ = f'Nrand_{N_rand}_'
    av_ += f'tbin_{t_bin_size}_tdelta_{round(t_bin_delta, 8)}'.replace('.', '_')

    fig_name = f'loschmidt_echo_{bc}_bc_L{L}_N{N}_U{U}_{av_}.pdf'


    time = np.logspace(-1, 7, 200)


    def get_Schmidt_Echo(U_1, th_1, U_2, th_2):

        temp_dir = path_fig/'temp_data'
        temp_dir.mkdir(parents=True, exist_ok=True)

        U_ = f'U1_{round(U_1,8)}_U2_{round(U_2,8)}'.replace('.', '_')
        th_ = f'thpi1_{round(th_1/np.pi,8)}_thpi2_{round(th_2/np.pi,8)}'.replace('.', '_')
        name_npz = temp_dir/f'loschmidt_echo_{bc}_bc_L{L}_N{N}_{av_}_{U_}_{th_}.npz'

        if name_npz.is_file():
            return [
                np.load(name_npz)['mean_loschmidt'],
                np.load(name_npz)['rand_vec']
            ]

        se_class = ScanClass_SchmidtEcho(
            bc = bc,
            J = 1,
            L = L,
            N = N,
        )
        se_class.time = time
        se_class.psi0_str = 'random_psi0'

        mean_loschmidt, rand_vec = se_class.get_Lochmidt_Echo_rand_av(
            N_rand=N_rand,
            t_bin_size=t_bin_size,
            t_bin_delta=t_bin_delta,
            U_forward=U_1,
            theta_forward=th_1,
            U_back=U_2,
            theta_back=th_2,
        )

        np.savez(name_npz, mean_loschmidt=mean_loschmidt, rand_vec=rand_vec)

        return mean_loschmidt, rand_vec



    #---------------------------------------------------------------------------
    # get data
    #---------------------------------------------------------------------------

    #===========================================================================
    """Define grid structure"""
    plt.rc('text', usetex=True)
    fig = plt.figure(figsize=(8.3/2, 4), dpi=300)
    fig.subplots_adjust(left=0.15, right=0.9,)

    #split vertical
    canv = gridspec.GridSpec(2, 1, height_ratios=[1, 1], hspace=0.2)

    canv_top = gridspec.GridSpecFromSubplotSpec(
        1, 2, canv[0, 0], width_ratios=[1, 1], wspace=0.3)
    canv_bot = gridspec.GridSpecFromSubplotSpec(
        1, 2, canv[1, 0], width_ratios=[1, 1], wspace=0.3)


    ax1 = plt.subplot(canv_top[0, 0])
    ax2 = plt.subplot(canv_top[0, 1])
    ax3 = plt.subplot(canv_bot[0, 0])
    ax4 = plt.subplot(canv_bot[0, 1])



    styles =[
        {'markevery':60, 'marker':'s', 'lw':1, 'ms':2.5, 'color':'tomato'},
        {'markevery':55, 'marker':'*', 'lw':1, 'ms':2.5, 'color':'cornflowerblue'},
        {'markevery':58, 'marker':'D', 'lw':1, 'ms':2.5, 'color':'forestgreen'},
        {'markevery':65, 'marker':'<', 'lw':1, 'ms':2.5, 'color':'orange'},
        {'markevery':70, 'marker':'o', 'lw':1, 'ms':2.5, 'color':'gold'},

    ] 



    #===========================================================================
    # top
    #===========================================================================
    th_list = np.array([0.0, 0.2, 0.5, 0.8, 1.0])*np.pi

    d_th1 = 0.005 * np.pi
    for i, th in enumerate(th_list):
        mean_loschmidt, rand_vec = get_Schmidt_Echo(
            U_1=U,
            U_2=U,
            th_1=th,
            th_2=th+d_th1
        )
        # plt.close()
        # plt.scatter(np.abs(rand_vec)**2)
        # plt.show()
        # exit()

        ax1.plot(time, mean_loschmidt, **styles[i])


    d_th2 = 0.05 * np.pi
    l_list = []
    for i, th in enumerate(th_list):
        mean_loschmidt, _  = get_Schmidt_Echo(
            U_1=U,
            U_2=U,
            th_1=th,
            th_2=th+d_th2
        )
        l1, = ax2.plot(time, mean_loschmidt, **styles[i])
        l_list.append(l1)


    ax1.legend(
        l_list,
        (r'$\theta/\pi=0$', r'$0.2$', r'$0.5$', r'$0.8$', r'$1$'),
        bbox_to_anchor=(1.2, 1.2), ncol=7, loc='center', fontsize=8,
        framealpha=0.9, borderpad=0.2, handlelength=1.5, handletextpad=0.3,
        labelspacing=0.2, columnspacing=0.5, frameon=True
    )


    ax1.set_title(r'$\Delta \theta = 0.005 $', fontsize=9, pad=2)
    ax2.set_title(r'$\Delta \theta = 0.05 $', fontsize=9, pad=2)

    #===========================================================================
    # bottom
    #===========================================================================
    th_list = np.array([0.0, 0.2, 0.5, 0.8, 1.0])*np.pi

    d_U = 0.05
    for i, th in enumerate(th_list):
        mean_loschmidt, _ = get_Schmidt_Echo(
            U_1=U,
            U_2=U+d_U,
            th_1=th,
            th_2=th
        )
        ax3.plot(time, mean_loschmidt, **styles[i])


    d_U = 0.5
    for i, th in enumerate(th_list):
        mean_loschmidt, _ = get_Schmidt_Echo(
            U_1=U,
            U_2=U+d_U,
            th_1=th,
            th_2=th
        )
        ax4.plot(time, mean_loschmidt, **styles[i])
        l_list.append(l1)


    ax3.set_title(r'$\Delta U = 0.05 $', fontsize=9, pad=2)
    ax4.set_title(r'$\Delta U = 0.5 $',  fontsize=9, pad=2)



    # #===========================================================================
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_ylim(3e-3, 1.05)
        # ax.axvline(0.2, color='red', ls=':')
        ax.set_xscale('log')
        # ax.set_yscale('log')
        ax.set_xlim(time[0], time[-1])



    ax1.set_ylabel(r'$\mathcal{F}$', fontsize=12, labelpad=3)
    ax3.set_ylabel(r'$\mathcal{F}$', fontsize=12, labelpad=3)

    ax3.set_xlabel(r'$t$', fontsize=12, labelpad=0)
    ax4.set_xlabel(r'$t$', fontsize=12, labelpad=0)


    ax1.set_xticklabels([])
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])
    ax4.set_yticklabels([])

    ax1.annotate('(a)', fontsize=10, xy=(0.02, 0.05), xycoords='axes fraction')
    ax2.annotate('(b)', fontsize=10, xy=(0.02, 0.05), xycoords='axes fraction')
    ax3.annotate('(c)', fontsize=10, xy=(0.02, 0.05), xycoords='axes fraction')
    ax4.annotate('(d)', fontsize=10, xy=(0.02, 0.05), xycoords='axes fraction')



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
