import math
from pathlib import Path
from collections.abc import Iterable

import numpy as np
import matplotlib.pyplot as plt

from hamiltonian.solve_hamilt import AnyonHubbardHamiltonian
from propagation.propagate_td_ham import PropagationTD

from helper import path_dirs



class AdiabTimePropagation():
    """Make adiabatic evoltion from theta=0 -> 2pi
    """

    def __init__(self, **args_in):
        """Initialize parameters.

        Args:
            bc (str): boundary conditions of lattice, open (hard-wall) | periodic
            L (int): number of lattice sites
            N (int): number of atoms
            J (float): hopping constant
            U (float): on-site interaction
         """

        self.bc = args_in.get('bc')
        self.L = args_in.get('L')
        self.N = args_in.get('N')
        self.U = args_in.get('U', 0)
        self.J = args_in.get('J', 1)
        self.theta_ini = args_in.get('theta_ini', 0)
        self.vary_theta_sites = args_in.get('vary_theta_sites', list(range(1, self.L)))  # sum runs unitl L-1
        self.Tprop = args_in.get('Tprop')
        self.dt = args_in.get('dt')
        self.create_new = args_in.get('create_new', False)

        self.load_only = args_in.get('load_only', False)


        # assert self.U == 0
        # assert self.J == 1

        path_basis = path_dirs.get_path_basis(
            self.bc, self.L, self.N, self.U, self.theta_ini, self.J)
        self.path_dir = path_basis/'adiabatic_propagation'

        # # self.path_npz = self.get_path_npz()
        # self.path_fig = path_dirs.get_path_fig_top(
        #     self.bc, self.L, self.N)/'adiabatic_time_evolution'
        self.temp_dict = {}

        self.define_E0_state = None  # this feature should allow to propagate 
                                     # any state apart from E=0
        self.define_E0_state_str = None


    #===========================================================================
    def make_path_npz(self, pre_str=None, add_Tprop=False, add_dt=False,
                      add_thfct=None):

        name_npz = ''

        if pre_str is not None:
            name_npz += pre_str

        if add_Tprop:
            name_npz += f'_Tprop_{round(self.Tprop, 8)}'.replace('.', '_')

        if add_dt:
            name_npz += f'_dt_{round(self.dt, 8)}'.replace('.', '_')

        if add_thfct is not None:
            name_npz += f'_{add_thfct}'

        if self.define_E0_state is not None:
            name_npz += f'_{self.define_E0_state_str}'

        name_npz += '_vary_th_sites_' + '_'.join([str(el) for el in self.vary_theta_sites])


        name_npz += '.npz'

        return self.path_dir/name_npz



    #===========================================================================
    # gs adiabatic projection
    #===========================================================================
    def get_E0_subspace_th0(self):
        if 'E0_th0' in self.temp_dict.keys():
            return self.temp_dict['E0_th0']


        hamilt_class = AnyonHubbardHamiltonian(
            bc=self.bc,
            L=self.L,
            N=self.N,
            J=self.J,
            U=0,
            theta=0,
            # bool_recreate=True
        )


        E0_th0 = hamilt_class.get_E0_eigenstate_energy_basis()


        self.temp_dict['E0_th0'] = E0_th0

        return E0_th0


    #---------------------------------------------------------------------------
    def get_theta_fcts(self, type):
        """Specify time-dependent function for theta
            th(t): from 0 to 2pi within 0 <= t <= self.Tprop
        """

        if type == 'linear':

            # loop over sites, for each site
            theta_fct_list = []
            for site_i in range(self.L):
                
                if site_i+1 in self.vary_theta_sites:
                    theta_fct = lambda t: 2*np.pi/self.Tprop * t
                else:
                    theta_fct = lambda t: self.theta_ini

                theta_fct_list.append(theta_fct)


        if False:
            # plot theta
            time = np.linspace(0, self.Tprop, 1000)
            for i, th_fct in enumerate(theta_fct_list):
                plt.plot(time, [th_fct(t) for t in time], label=f'fct site {i+1}')
            plt.legend()
            plt.show()
            exit()
        # assert np.allclose(theta_fct(0), 0)
        # assert np.allclose(np.abs(theta_fct(self.Tprop)), 2*np.pi)

        return theta_fct_list


    #---------------------------------------------------------------------------
    def adiabatic_propagation(self, E0_th0_i, th_fct):
        """Propagate E0 states from theta = 0 --> 2\pi
        """

        theta_fct_list = self.get_theta_fcts(type=th_fct)

        theta_snaplist = np.array([i/257*(2*np.pi) for i in range(0, 257)])
        t_snaps = np.round(
            [th_i * self.Tprop / (2*np.pi) for th_i in theta_snaplist], 8
        )


        time_dt = np.round(np.arange(0, self.Tprop+self.dt, self.dt), 8)

        # new time list
        time = np.array(list(sorted(set(list(time_dt) + list(t_snaps)))))

        prop_class = PropagationTD(
            bc=self.bc,
            L=self.L,
            N=self.N,
            J=self.J,
            U=self.U,
            psi0_str=None,
            theta_fct_list=theta_fct_list,
            Tprop=self.Tprop,
            dtprop=self.dt,
        )
        prop_class.psi0 = E0_th0_i
        prop_class.time = time
        
        psi_t = prop_class.psi_t()
        theta_vals_list = np.array(
            [th_fct(t) for t in time]
            for th_fct in theta_fct_list
        )

        return time, theta_vals_list, psi_t


    #---------------------------------------------------------------------------
    def get_adiabatic_E0_propagations(self, th_fct='linear'):

        if 'psi_lol' in self.temp_dict.keys():
            return [
                self.temp_dict['time'],
                self.temp_dict['theta_vals_list'],
                self.temp_dict['psi_lol'] 
            ]

        E0_th0 = self.get_E0_subspace_th0()

        psi_lol = []
        for E0_th0_i in E0_th0:

            time, theta_vals_list, psi_t = self.adiabatic_propagation(E0_th0_i, th_fct)

            psi_lol.append(psi_t)

        self.temp_dict['time'] = time
        self.temp_dict['theta_vals_list'] = theta_vals_list
        self.temp_dict['psi_lol'] = np.array(psi_lol)

        return self.get_adiabatic_E0_propagations()


    #===========================================================================
    # analyze psi(t)
    #===========================================================================
    def get_holonomy_mat(self):

        path_npz = self.make_path_npz(pre_str='holo_mat', add_Tprop=True)

        if path_npz.is_file() and not self.create_new:
            return [
                np.load(path_npz)['evals'],
                np.load(path_npz)['evec'],
                np.load(path_npz)['holo_mat']
            ]

        if self.load_only:
            return None, None, None

        E0_th0 = self.get_E0_subspace_th0()

        time, theta_vals_list, psi_lol = self.get_adiabatic_E0_propagations()

        # print(psi_lol[0][0] - E0_th0[0])
        # print(psi_lol[0][1] - E0_th0[1])
        # print(psi_lol[0][2] - E0_th0[2])
        # print(np.sum(np.abs(psi_lol[0][0] - E0_th0[0])))
        # print(np.sum(np.abs(psi_lol[1][0] - E0_th0[1])))
        # print(np.sum(np.abs(psi_lol[2][0] - E0_th0[2])))

        n_vecs = psi_lol.shape[0]
        holo_mat = np.zeros((n_vecs, n_vecs), dtype=complex)
        for i in range(n_vecs):
            for j in range(n_vecs):
                # holo_mat[i, j] = np.vdot(E0_th0[i], psi_lol[j][-1])
                holo_mat[i, j] = np.vdot(E0_th0[i], psi_lol[j, -1, :])

        evals, evec = np.linalg.eig(holo_mat)

        path_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path_npz,
            evals=evals,
            evec=evec,
            holo_mat=holo_mat
        )

        return evals, evec, holo_mat
