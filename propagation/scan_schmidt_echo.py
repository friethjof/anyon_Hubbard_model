import math
import gc
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from numba import njit

from hamiltonian.solve_hamilt import AnyonHubbardHamiltonian
from propagation.propagate_static import PropagationStatic


from propagation import initialize_prop
from helper import path_dirs
from helper import other_tools
from helper import plot_helper
from helper import operators




@njit
def get_loschmidt(time, psi0, evals_forw, evecs_forw, evals_back, evecs_back):
    length = psi0.shape[0]
    loschmidt = np.zeros(time.shape[0], dtype=float)
    for i, t in enumerate(time):
        # print(f't = {t:.1f}')
        psi_forw = np.zeros(length, dtype=np.complex128)
        for eval_n, evec_n in zip(evals_forw, evecs_forw):

            # n_n = np.outer(evec_n, np.conjugate(evec_n))
            # psi_tstep += np.exp(-1j*eval_n*t)* n_n.dot(psi0)

            psi_forw += np.exp(-1j*eval_n*t)*evec_n*np.vdot(evec_n, psi0)

        # backward propagation
        psi_back = np.zeros(length, dtype=np.complex128)
        for eval_m, evec_m in zip(evals_back, evecs_back):

            psi_back += np.exp(1j*eval_m*t)*evec_m*np.vdot(evec_m, psi_forw)

            # assert np.abs(np.vdot(psi_back, psi_back) - 1+0j) < 1e-8

        loschmidt[i] = (np.abs(np.vdot(psi_back, psi0))**2).real

    return loschmidt



#-------------------------------------------------------------------------------
class ScanClass_SchmidtEcho():
    """Organize exact diagonalization method, plotting, scaning of paramters.
    """

    def __init__(self, **args_in):
        """Initialize parameters.

        """

        self.bc = args_in.get('bc', None)
        self.L = args_in.get('L', None)
        self.N = args_in.get('N', None)
        self.U = args_in.get('U', None)
        self.J = args_in.get('J', None)
        self.theta = args_in.get('theta', None)
        self.d_th = args_in.get('d_th', 0.0)
        self.d_U = args_in.get('d_U', 0.0)

        # self.psi0_class = args_in.get('psi0_class', None)
        self.psi0_str = args_in.get('psi0_str', None)
        # self.Tprop = args_in.get('Tprop', None)
        # self.dtprop = args_in.get('dtprop', None)

        # vary over these theta values
        self.d_th_list = [self.theta]
        self.d_U_list = [self.U]

        self.theta_list = [self.theta]
        self.U_list = [self.U]

        self.time = None # np.arange(0, self.Tprop+self.dtprop, self.dtprop)
        # self.dum_Tf_dt = path_dirs.get_dum_Tf_dt(self.Tprop, self.dtprop)


    #===========================================================================
    def get_path_fig(self, **args_in):
        path_fig = path_dirs.get_path_fig_top_dyn(
            bc=self.bc,
            L=self.L,
            N=self.N,
            J=args_in.get('J', self.J),
            U=None,
            psi0_str=args_in.get('psi0_name', self.psi0_str),
            # Tprop=self.Tprop,
            # dtprop=self.dtprop
        )
        return path_fig


    def get_name_pref(self, N_rand=None, t_bin_size=None, t_bin_delta=None):
        name_ = []
        if self.U is not None:
            name_.append(f'U_{round(self.U, 8)}')
        if self.theta is not None:
            name_.append(f'thpi_{round(self.theta/np.pi, 8)}')
        if self.d_U != 0.0:
            name_.append(f'dU_{round(self.d_U, 8)}')
        if self.d_th != 0.0:
            name_.append(f'dth_{round(self.d_th/np.pi, 8)}')

        if N_rand is not None:
            name_.append(f'N_rand_{N_rand}')
        if t_bin_size is not None:
            name_.append(f't_bin_{t_bin_size}')
        if t_bin_delta is not None:
            name_.append(f'td_{round(t_bin_delta, 8)}')

        name_ = '_'.join(name_)
        return name_.replace('.', '_')



    #===========================================================================
    def get_hamil_backforth(self, **args_in):

        hamil_class_forward = AnyonHubbardHamiltonian(
            bc=self.bc,
            L=self.L,
            N=self.N,
            J=args_in.get('J_forward', self.J),
            U=args_in.get('U_forward', self.U),
            theta=args_in.get('theta_forward', self.theta)
        )

        hamil_class_back = AnyonHubbardHamiltonian(
            bc=self.bc,
            L=self.L,
            N=self.N,
            J=args_in.get('J_back', self.J),
            U=args_in.get('U_back', self.U),
            theta=args_in.get('theta_back', self.theta)
        )

        return hamil_class_forward, hamil_class_back


    def get_Schmidt_Echo(self, **args_in):
        """Calculate = |<psi0 | exp(iH't) exp(-iHt) | psi0>|^2"""
        assert isinstance(self.bc, str)
        assert isinstance(self.L, int)
        assert isinstance(self.N, int)

        if 'hamil_class_forward' in args_in.keys():
            hamil_class_forward = args_in["hamil_class_forward"]
            hamil_class_back = args_in["hamil_class_back"]
        else:
            hamil_class_forward, hamil_class_back = \
                self.get_hamil_backforth(**args_in)

        if 'psi0' in args_in.keys():
            psi0 = args_in['psi0']
            nstate0_str = ''
        else:
            psi0, nstate0_str = initialize_prop.get_psi0(
                psi0_str=self.psi0_str,
                basis_list=hamil_class_forward.basis.basis_list,
                eigstates=hamil_class_forward.evecs()
            )

        #-----------------------------------------------------------------------
        # make propagation
        #   |<psi0 | exp(i H' t) exp(-i H t) | psi0>|^2
        # = |<psi0 | sum_n' exp(i E_n' t) |n'><n'| sum_n exp(-i E_n t) |n><n| | psi0>|^2
        #-----------------------------------------------------------------------


        loschmidt_echo = get_loschmidt(
            time=self.time,
            psi0=psi0.astype(np.complex128),
            evals_forw=hamil_class_forward.evals(),
            evecs_forw=hamil_class_forward.evecs(),
            evals_back=hamil_class_back.evals(),
            evecs_back=hamil_class_back.evecs(),
        )

        # loschmidt_echo_test = []
        # for t in self.time:
        #     # print(f't = {t:.1f}')
        #     psi_tstep = np.zeros(hamil_class_forward.basis.length, dtype=complex)

        #     #-------------------------------------------------------------------
        #     # forward propagation
        #     for eval_n, evec_n in zip(hamil_class_forward.evals(), hamil_class_forward.evecs()):

        #         psi_tstep += (
        #             np.exp(-1j*eval_n*t)
        #             *evec_n*np.vdot(evec_n, psi0)
        #         )
            
        #     psi_back = np.zeros(hamil_class_forward.basis.length, dtype=complex)
        #     #-------------------------------------------------------------------
        #     # backward propagation
        #     for eval_m, evec_m in zip(hamil_class_back.evals(), hamil_class_back.evecs()):

        #         psi_back += (
        #             np.exp(1j*eval_m*t)
        #             *evec_m*np.vdot(evec_m, psi_tstep)
        #         )

        #     assert np.abs(np.vdot(psi_back, psi_back) - 1+0j) < 1e-8

        #     loschmidt_echo_test.append(np.abs(np.vdot(psi_back, psi0))**2)

        # print(np.allclose(loschmidt_echo_test, loschmidt_echo))
        # exit()
        return loschmidt_echo, nstate0_str



    def get_loschmidt_time_av(self, t_bin_size, t_bin_delta, **loschmidt_args):

        time_out = np.copy(self.time)
        time_av = np.sort(np.array([
            np.linspace(t-t_bin_delta, t+t_bin_delta, t_bin_size+1)
            for t in time_out
        ]).flatten())
        time_av = np.array([el for el in time_av if el >= 0])

        self.time = time_av

        loschmidt_echo, _ = self.get_Schmidt_Echo(
            **loschmidt_args,
        )

        lo_echo_time_av = []
        for t in time_out:
            t_min = max(0, t - t_bin_delta)
            t_max = t + t_bin_delta
            ind_min = np.argmin(np.abs(time_av - t_min))
            ind_max = np.argmin(np.abs(time_av - t_max))

            lo_bin = loschmidt_echo[ind_min:ind_max]
            lo_echo_time_av.append(np.mean(lo_bin))

        self.time = time_out

        return np.array(lo_echo_time_av)



    def get_Lochmidt_Echo_rand_av(self, **args_in):
            
        N_rand = args_in['N_rand']

        hamil_class_forward, hamil_class_back = self.get_hamil_backforth(**args_in)

        ini_size = hamil_class_forward.basis.length

        if N_rand == 'nstate':
            n_range = ini_size  # all number states
        else:
            n_range = N_rand

        loschmidt_echo_list = []
        rand_vec_all = 0
        for i in range(n_range):
            print(f'rand {i+1}')

            if N_rand == 'nstate':
                rand_vec = np.zeros(ini_size, dtype=np.complex128)
                rand_vec[i] = 1
            else:
                rand_vec = np.random.uniform(0, 1, ini_size) + 1.j * np.random.uniform(0, 1, ini_size)
                rand_vec = np.ones(ini_size)

            rand_vec /= np.sqrt(np.vdot(rand_vec, rand_vec))
            rand_vec_all += rand_vec


            lo_echo_time_av = self.get_loschmidt_time_av(
                t_bin_size=args_in['t_bin_size'],
                t_bin_delta=args_in['t_bin_delta'],
                psi0=rand_vec,
                hamil_class_forward=hamil_class_forward,
                hamil_class_back=hamil_class_back
            )

            loschmidt_echo_list.append(lo_echo_time_av)


        mean_loschmidt = np.mean(loschmidt_echo_list, axis=0)        
        # plt.plot(self.time, mean_loschmidt, c='black')
        # plt.xscale('log')
        # plt.show()
        return mean_loschmidt, rand_vec_all/n_range



    #===========================================================================
    # vary theta
    #===========================================================================
    def vary_delta_theta(self):
        "Propagate with theta forward and with theta+d back"


        for d_th in self.d_th_list:

            path_fig = self.get_path_fig()

            th_ = f'thpi_{round(self.theta/np.pi, 8)}_dth_{round(d_th/np.pi, 8)}'.replace('.', '_')
            fig_name = path_fig/'loschmidt_echo_vary_th'/f'{th_}'
            

            #-------------------------------------------------------------------
            # get Loschmidt echo
            loschmidt_echo, nstate0_str = self.get_Schmidt_Echo(
                theta_forward = self.theta,
                theta_back    = self.theta + d_th
            )

            #-------------------------------------------------------------------
            # make plot
            title = plot_helper.make_title(
                L=self.L,
                N=self.N,
                U=self.U,
                theta=self.theta,
                delta_theta=d_th,
                psi0=self.psi0_str
            )

            plot_helper.plot_lines(
                fig_name=fig_name,
                x_lists=[self.time],
                y_lists=[loschmidt_echo],
                fig_args={
                    'xlabel':'time',
                    'ylabel':'Loschmidt echo',
                    'title':title,
                    'xlog':True
                }
            )  


    def vary_delta_U(self):
        "Propagate with theta forward and with theta+d back"


        for d_U in self.d_U_list:

            path_fig = self.get_path_fig()

            th_ = f'thpi_{round(self.theta/np.pi, 8)}'.replace('.', '_')
            U_ = f'U_{round(self.U, 8)}_dU_{round(d_U, 8)}'.replace('.', '_')
            fig_name = path_fig/'loschmidt_echo_vary_U'/f'{th_}_{U_}'
            

            #-------------------------------------------------------------------
            # get Loschmidt echo
            loschmidt_echo, nstate0_str = self.get_Schmidt_Echo(
                U_forward = self.U,
                U_back    = self.U + d_U
            )

            #-------------------------------------------------------------------
            # make plot
            title = plot_helper.make_title(
                L=self.L,
                N=self.N,
                U=self.U,
                theta=self.theta,
                delta_U=d_U,
                psi0=self.psi0_str
            )

            plot_helper.plot_lines(
                fig_name=fig_name,
                x_lists=[self.time],
                y_lists=[loschmidt_echo],
                fig_args={
                    'xlabel':'time',
                    'ylabel':'Loschmidt echo',
                    'title':title,
                    'xlog':True
                }
            )


    #===========================================================================
    # vary theta, random initial state
    #===========================================================================
    def vary_theta_rand(self, N_rand, t_bin_size, t_bin_delta):
        "Propagate with theta forward and with theta+d back"

        path_fig = self.get_path_fig()
        name_ = self.get_name_pref(N_rand, t_bin_size, t_bin_delta)

        fig_name = path_fig/'loschmidt_echo_averaged_vary_theta'/name_


        schmidt_echo_rand = []
        label_list = []

        for theta in self.theta_list:

            #-------------------------------------------------------------------
            # get Loschmidt echo
            loschmidt_echo, _ = self.get_Lochmidt_Echo_rand_av(
                theta_forward = theta,
                theta_back = theta + self.d_th,
                U_forward = self.U,
                U_back = self.U + self.d_U,
                N_rand = N_rand,
                t_bin_size = t_bin_size,
                t_bin_delta = t_bin_delta
            )

            schmidt_echo_rand.append(loschmidt_echo)

            label_list.append(r'$\theta/\pi='+ f'{round(theta/np.pi, 8)}$')


        plot_helper.plot_lines(
            fig_name=fig_name,
            x_lists=[self.time],
            y_lists=schmidt_echo_rand,
            label_list=label_list,
            fig_args={
                'xlabel':'time',
                'ylabel':'Loschmidt echo',
                'xlog':True,
            }
        ) 


    #===========================================================================
    # vary U, random initial state
    #===========================================================================
    def vary_U_rand(self, N_rand):
        "Propagate with theta forward and with theta+d back"


        path_fig = self.get_path_fig()

        # name_ = self.get_name_pref()

        delta_ = f'thpi_{round(self.theta/np.pi,8)}_dU_{round(self.d_U, 8)}'.replace('.', '_')

        fig_name = path_fig/'loschmidt_echo_averaged_vary_U'/f'{delta_}_N_rand_{N_rand}'

        assert self.d_th == 0.0

        schmidt_echo_rand = []
        label_list = []

        for U in self.U_list:

            #-------------------------------------------------------------------
            # get Loschmidt echo
            loschmidt_echo, _ = self.get_Lochmidt_Echo_rand_av(
                theta_forward = self.theta,
                theta_back = self.theta + self.d_th,
                U_forward = U,
                U_back = U + self.d_U,
                N_rand = N_rand
            )

            schmidt_echo_rand.append(loschmidt_echo)

            label_list.append(f'$U={round(U, 8)}$')


        plot_helper.plot_lines(
            fig_name=fig_name,
            x_lists=[self.time],
            y_lists=schmidt_echo_rand,
            label_list=label_list,
            fig_args={
                'xlabel':'time',
                'ylabel':'Loschmidt echo',
                'xlog':True,
            }
        ) 

