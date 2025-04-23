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


    path_fig = path_dirs.adv_fig/'loschmidt_echo_fft'


    L = 10
    N = 2


    U = 0.0
    delta_U = 0.5

    theta1 = 0 * np.pi
    theta2 = 0.2 * np.pi

    delta_th = 0.1 * np.pi

    bc = 'twisted_gauge'
    # bc = 'open'
    eigstate_str = 'eigstate_0'


    #---------------------------------------------------------------------------
    # name
    U_ = f'U_{round(U, 8)}'.replace('.', '_')
    if delta_U != 0:
        U_ += f'dU_{round(delta_U, 8)}'.replace('.', '_')

    if delta_th != 0:
        th_ = f'_dthpi_{round(delta_th/np.pi,8)}_'.replace('.', '_')
    else:
        th_ = ''


    fig_name = f'loschmidt_echo_fft_{bc}_bc_L{L}_N{N}_{U_}{th_}_{eigstate_str}.pdf'


    time = np.arange(0, 1e4, 0.1)





    se_class = ScanClass_SchmidtEcho(
        bc = bc,
        J = 1,
        L = L,
        N = N,
    )
    se_class.time = time
    se_class.psi0_str = eigstate_str

    loschmidt_1, _ = se_class.get_Schmidt_Echo(
        U_forward     = U,
        U_back        = U,
        theta_forward = theta1,
        theta_back    = theta1 + delta_th,
    )

    loschmidt_2, _ = se_class.get_Schmidt_Echo(
        U_forward     = U,
        U_back        = U,
        theta_forward = theta2,
        theta_back    = theta2 + delta_th,
    )


    loschmidt_3, _ = se_class.get_Schmidt_Echo(
        U_forward     = U,
        U_back        = U + delta_U,
        theta_forward = theta1,
        theta_back    = theta1,
    )

    loschmidt_4, _ = se_class.get_Schmidt_Echo(
        U_forward     = U,
        U_back        = U + delta_U,
        theta_forward = theta2,
        theta_back    = theta2,
    )


    from scipy.fft import rfft, rfftfreq, irfft
    from scipy.signal import find_peaks

    # print(fs, 1/fs)
    def make_fft(x, fs=time.shape[0]):
        freq = rfftfreq(fs, 0.1)
        u_fft = rfft(x)
        u_ifft = irfft(u_fft, n=x.shape[0])

        freq = freq[1:]
        u_fft = np.abs(u_fft[1:])
        return freq, u_fft


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
        {'lw':1, 'color':'tomato'},
        {'lw':1, 'ls':'--', 'color':'gray'},
    ] 



    #===========================================================================
    # top
    #===========================================================================
    ax1.plot(*make_fft(loschmidt_1), **styles[0])#, **styles[0])

    ax2.plot(*make_fft(loschmidt_2), **styles[0])

    ax3.plot(*make_fft(loschmidt_3), **styles[0])

    ax4.plot(*make_fft(loschmidt_4), **styles[0])


    str_U = f'$U={U}$'
    str_dU = r'$\Delta' + f' U={delta_U}$'
    str_th1 = r'$\theta' + f'={round(theta1/np.pi,8)}' + r'\pi$'
    str_th2 = r'$\theta' + f'={round(theta2/np.pi,8)}' + r'\pi$'
    str_dth = r'$\Delta\theta' + f'={round(delta_th/np.pi,8)}' + r'\pi$'


    ax1.set_title(f'{str_U}, {str_th1}, {str_dth}', fontsize=6, pad=2)
    ax2.set_title(f'{str_U}, {str_th2}, {str_dth}', fontsize=6, pad=2)
    ax3.set_title(f'{str_U}, {str_th1}, {str_dU}', fontsize=6, pad=2)
    ax4.set_title(f'{str_U}, {str_th2}, {str_dU}', fontsize=6, pad=2)


    # #===========================================================================
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_ylim(1e-3, 300)
        # ax.axvline(0.2, color='red', ls=':')
        # ax.set_xscale('log')
        ax.set_yscale('log')
        # ax.set_yscale('log')


    #     ax.set_xlim(time[0], time[-1])



    ax1.set_ylabel(r'$A(\omega)$', fontsize=12, labelpad=3)
    ax3.set_ylabel(r'$A(\omega)$', fontsize=12, labelpad=3)

    ax3.set_xlabel(r'$\omega$', fontsize=12, labelpad=0)
    ax4.set_xlabel(r'$\omega$', fontsize=12, labelpad=0)


    # ax1.set_xticklabels([])
    # ax2.set_xticklabels([])
    # ax2.set_yticklabels([])
    # ax4.set_yticklabels([])

    ax1.annotate('(a)', fontsize=10, xy=(0.82, 0.85), xycoords='axes fraction')
    ax2.annotate('(b)', fontsize=10, xy=(0.82, 0.85), xycoords='axes fraction')
    ax3.annotate('(c)', fontsize=10, xy=(0.82, 0.85), xycoords='axes fraction')
    ax4.annotate('(d)', fontsize=10, xy=(0.82, 0.85), xycoords='axes fraction')





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
