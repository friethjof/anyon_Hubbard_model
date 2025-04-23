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



def make_plot():

    bc = 'open'
    L = 40
    N = 2
    U = 0
    theta = 0.2*np.pi

    average_set = {'type':'randstates_uniform', 'N_rand':550}

    t_bin_size = 10
    t_bin_delta = 0.05

    texp_1 = -2
    texp_2 = 8
    N_t = 300
    time = np.logspace(texp_1, texp_2, N_t)

    #---------------------------------------------------------------------------
    # name
    if  average_set['type'] == 'randstates_uniform':
        av_ = f"randstates_uniform"
        N_rand_ = f"_Nr{average_set['N_rand']}_"


    tbin_ = f'tbin_{t_bin_size}_tdelta_{round(t_bin_delta, 8)}'.replace('.', '_')
    U_ = f'U_{round(U,8)}'.replace('.', '_')
    th_ = f'thpi_{round(theta/np.pi,8)}'.replace('.', '_')

    path_fig = path_dirs.adv_fig

    fig_name = f'fidelity_{bc}_bc_L{L}_N{N}_{U_}_{th_}_{av_}{N_rand_}_{tbin_}.png'




    temp_dir = Path('/afs/physnet.uni-hamburg.de/project/zoq_t/'
                    'harm_osc_coll/13/project_ahm_integrability')
    if bc == 'open':
        temp_dir = temp_dir/'open_boundary_conditions'
    else:
        raise NotImplementedError(bc)

    temp_dir = temp_dir/'fidelity'/av_/f'time_t10x{texp_1}_10x{texp_2}_N_t_{N_t}'
    name_npz = temp_dir/(f'fidelity_{bc}_bc_L{L}_N{N}_{tbin_}_{U_}_{th_}_{av_}{N_rand_}.npz')

    if name_npz.is_file() and len(np.load(name_npz)['e_list'])>0:
        e_list = np.load(name_npz)['e_list']
    else:

        prop_class = PropagationStatic(
            bc = bc,
            J = 1,
            L = L,
            N = N,
            U = U,
            theta = theta,
            Tprop=1,
            dtprop=1,
            bool_save=False,
            psi0_str=None,
        )
        prop_class.time = time

        fidelity_av, e_list = prop_class.get_fidelity_rand_av(
            average_set=average_set,
            t_bin_size=t_bin_size,
            t_bin_delta=t_bin_delta,
            # store_dict=None
            store_dict={
                'store_dir':temp_dir/f'fidelity_{bc}_bc_L{L}_N{N}_{tbin_}_{U_}_{th_}',
                'dum_name': f'fidelity_time_av'
            }
        )

        ind_inf = np.argmin(np.abs(time - 1e5))
        inf_t_av = np.mean(fidelity_av[ind_inf:])

        np.savez(
            name_npz,
            fidelity_av=fidelity_av,
            e_list=e_list,
            inf_t_av=inf_t_av
        )





    #---------------------------------------------------------------------------
    # get data
    #---------------------------------------------------------------------------

    #===========================================================================
    """Define grid structure"""
    plt.rc('text', usetex=True)
    fig = plt.figure(figsize=(8.3*0.5, 2.2), dpi=300)
    fig.subplots_adjust(left=0.2, right=0.94, top=0.94, bottom=0.1)

    #split vertical
    canv = gridspec.GridSpec(1, 1)



    ax1 = plt.subplot(canv[0, 0])




    #===========================================================================


    # plt.plot(e_list)
    # plt.show()
    # exit()
    # ax1.hist(energies, range=(np.min(energies), np.max(energies)))
    ax1.hist(e_list, bins=100, color= 'tomato', zorder=100)

    ax1t = ax1.twinx()
    ax1t.scatter(energies, range(energies.shape[0]), c='cornflowerblue', s=1, zorder=1)

    ax1.set_zorder(ax1t.get_zorder() + 1)
    ax1.patch.set_visible(False)



    # ax.set_xlim(time[0], time[-1])
    ax1.set_ylabel(r'bin', fontsize=12, labelpad=3)
    ax1t.set_ylabel(r'$i$', fontsize=12, labelpad=3)
    ax1.set_xlabel(r'$E$', fontsize=12, labelpad=3)




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
