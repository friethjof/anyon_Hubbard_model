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
from propagation.fidelity_class import FidelityAveraged


def make_plot(**args_in):
    
    L = args_in.get('L', 40)
    U = args_in.get('U', 0)
    t_bin_delta = args_in.get('t_bin_delta', 40)

    theta_set = args_in.get('theta_set', 1)


    ini_type = args_in.get('type')
    if ini_type == 'randstates_uniform_U0_th0':
        ini_states_dict = {'type':'randstates_uniform_U0_th0'}

    elif ini_type == 'randstates_uniform':
        ini_states_dict={'type':'randstates_uniform'}

    elif ini_type == 'randstates_gauss':
        ini_states_dict={
            'type':'randstates_gauss',
            'E_mid':args_in.get('E_mid', 0.0),
            'E_range':args_in.get('E_range', 0.1)
        }


    N_rand = 200


    #---------------------------------------------------------------------------
    N = 2
    bc = 'open'


    def get_fidelity_data(U, theta):

        fid_class = FidelityAveraged(
            bc = bc,
            J = 1,
            L = L,
            N = N,
            U = U,
            theta = theta,
            Tprop=1,
            dtprop=1,
            bool_save=False,
            create_fid_new=False,
            #
            N_rand=N_rand,
            t_bin_size=10,
            t_bin_delta=t_bin_delta,
            ini_states_dict=ini_states_dict,
        )

        time = fid_class.time
        fidelity, _ = fid_class.get_fidelity_rand_av()
        inf_t_av = fid_class.get_infinite_time_av()

        return time, fidelity, inf_t_av





    #---------------------------------------------------------------------------
    # name
    tbin_ = f'tdelta_{round(t_bin_delta, 8)}'.replace('.', '_')
    U_ = f'{float(U)}'.replace('.', '_')
    Nr_ = f'Nrand_{N_rand}'
    if  ini_type == 'randstates_gauss':
        E_mid = ini_states_dict['E_mid']
        E_range = ini_states_dict['E_range']
        ini_type = f"randstates_gauss_Emid_{E_mid}_Erange_{E_range}"
        ini_type = ini_type.replace('.', '_')

    path_fig = path_dirs.adv_fig/'fidelity_v2'/ini_type

    fig_name = f'fidelity_{bc}_bc_L{L}_N{N}_U{U_}_{ini_type}_v{theta_set}_{tbin_}_{Nr_}.png'


    #===========================================================================
    """Define grid structure"""
    plt.rc('text', usetex=True)
    fig = plt.figure(figsize=(8.3*0.5, 4.2), dpi=300)
    fig.subplots_adjust(left=0.2, right=0.94, top=0.94, bottom=0.1)

    #split vertical
    canv = gridspec.GridSpec(4, 1, height_ratios=[1, 1, 1, 1], hspace=0.3)



    ax1 = plt.subplot(canv[0, 0])
    ax2 = plt.subplot(canv[1, 0])
    ax3 = plt.subplot(canv[2, 0])
    ax4 = plt.subplot(canv[3, 0])



    styles =[
        {'lw':1.5, 'color':'tomato'},
        {'lw':1, 'ls':'--', 'color':'gray'},
    ] 


    #===========================================================================
    # top
    #===========================================================================
    if theta_set == 1:
        time, fidelity_av, inf_t_av = get_fidelity_data(U=U, theta=0)
        ax1.plot(time, fidelity_av, **styles[0])
        ax1.axhline(inf_t_av, **styles[1])
        ax1.set_title(f'$U={U}$'+r', $\theta=0$', fontsize=9, pad=2)

        time, fidelity_av, inf_t_av = get_fidelity_data(U=U, theta=0.1*np.pi)
        ax2.plot(time, fidelity_av, **styles[0])
        ax2.axhline(inf_t_av, **styles[1])
        ax2.set_title(f'$U={U}$'+r', $\theta=0.1\pi$', fontsize=9, pad=2)

        time, fidelity_av, inf_t_av = get_fidelity_data(U=U, theta=0.2*np.pi)
        ax3.plot(time, fidelity_av, **styles[0])
        ax3.axhline(inf_t_av, **styles[1])
        ax3.set_title(f'$U={U}$'+r', $\theta=0.2\pi$', fontsize=9, pad=2)

        time, fidelity_av, inf_t_av = get_fidelity_data(U=U, theta=0.3*np.pi)
        ax4.plot(time, fidelity_av, **styles[0])
        ax4.axhline(inf_t_av, **styles[1])
        ax4.set_title(f'$U={U}$'+r', $\theta=0.3\pi$', fontsize=9, pad=2)


    #---------------------------------------------------------------------------
    elif theta_set == 2:
        time, fidelity_av, inf_t_av = get_fidelity_data(U=U, theta=0.7)
        ax1.plot(time, fidelity_av, **styles[0])
        ax1.axhline(inf_t_av, **styles[1])
        ax1.set_title(f'$U={U}$'+r', $\theta=0.7$', fontsize=9, pad=2)

        time, fidelity_av, inf_t_av = get_fidelity_data(U=U, theta=0.8*np.pi)
        ax2.plot(time, fidelity_av, **styles[0])
        ax2.axhline(inf_t_av, **styles[1])
        ax2.set_title(f'$U={U}$'+r', $\theta=0.8\pi$', fontsize=9, pad=2)

        time, fidelity_av, inf_t_av = get_fidelity_data(U=U, theta=0.9*np.pi)
        ax3.plot(time, fidelity_av, **styles[0])
        ax3.axhline(inf_t_av, **styles[1])
        ax3.set_title(f'$U={U}$'+r', $\theta=0.9\pi$', fontsize=9, pad=2)


        time, fidelity_av, inf_t_av = get_fidelity_data(U=U, theta=1*np.pi)
        ax4.plot(time, fidelity_av, **styles[0])
        ax4.axhline(inf_t_av, **styles[1])
        ax4.set_title(f'$U={U}$'+r', $\theta=\pi$', fontsize=9, pad=2)


    # #---------------------------------------------------------------------------
    # elif theta_set == 3:
    #     time, fidelity_av, inf_t_av = get_fidelity_data(U=U, theta=1.0*np.pi)
    #     ax1.plot(time, fidelity_av, **styles[0])
    #     ax1.axhline(inf_t_av, **styles[1])
    #     ax1.set_title(f'$U={U}$'+r', $\theta=\pi$', fontsize=9, pad=2)

    #     time, fidelity_av, inf_t_av = get_fidelity_data(U=U, theta=1.1*np.pi)
    #     ax2.plot(time, fidelity_av, **styles[0])
    #     ax2.axhline(inf_t_av, **styles[1])
    #     ax2.set_title(f'$U={U}$'+r', $\theta=1.1\pi$', fontsize=9, pad=2)

    #     time, fidelity_av, inf_t_av = get_fidelity_data(U=U, theta=1.2*np.pi)
    #     ax3.plot(time, fidelity_av, **styles[0])
    #     ax3.axhline(inf_t_av, **styles[1])
    #     ax3.set_title(f'$U={U}$'+r', $\theta=1.2\pi$', fontsize=9, pad=2)


    #     time, fidelity_av, inf_t_av = get_fidelity_data(U=U, theta=1.3*np.pi)
    #     ax4.plot(time, fidelity_av, **styles[0])
    #     ax4.axhline(inf_t_av, **styles[1])
    #     ax4.set_title(f'$U={U}$'+r', $\theta=1.3\pi$', fontsize=9, pad=2)



    # #===========================================================================
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_ylim(0, 1.05)
        ax.set_ylim(1e-4, 1.05)
        # ax.axvline(0.2, color='red', ls=':')
        ax.set_xscale('log')
        ax.set_yscale('log')
        # ax.set_yscale('log')


        ax.set_xlim(time[0], time[-1])
        ax.set_ylabel(r'$S(t)$', fontsize=12, labelpad=3)


    ax4.set_xlabel(r'$t$', fontsize=12, labelpad=0)


    ax1.set_xticklabels([])
    ax2.set_xticklabels([])
    ax3.set_xticklabels([])


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

    # if fig_name[-3:] == 'png':
    #     subprocess.check_output(["convert", fig_name, "-trim", fig_name])
    # elif fig_name[-3:] == 'pdf':
    #     subprocess.check_output(["pdfcrop", fig_name, fig_name])

    os.chdir(path_cwd)
