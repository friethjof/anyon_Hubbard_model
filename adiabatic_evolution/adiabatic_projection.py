import math
import gc
from pathlib import Path
from collections.abc import Iterable

import numpy as np
import matplotlib.pyplot as plt

from hamiltonian.solve_hamilt import AnyonHubbardHamiltonian
from propagation.propagate_td_ham import PropagationTD

from helper import path_dirs



class AdiabProjection():
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
        self.N_th = args_in.get('N_th')
        self.create_new = args_in.get('create_new', False)

        self.load_only = args_in.get('load_only', False)


        # assert self.U == 0
        # assert self.J == 1

        path_basis = path_dirs.get_path_basis(
            self.bc, self.L, self.N, self.U, self.theta_ini, self.J)
        self.path_dir = path_basis/'adiabatic_projection'

        # # self.path_npz = self.get_path_npz()
        # self.path_fig = path_dirs.get_path_fig_top(
        #     self.bc, self.L, self.N)/'adiabatic_time_evolution'
        self.temp_dict = {}

        self.define_E0_state = None  # this feature should allow to propagate 
                                     # any state apart from E=0
        self.define_E0_state_str = None


    #===========================================================================
    def make_path_npz(self, pre_str=None):

        name_npz = ''

        if pre_str is not None:
            name_npz += pre_str

        name_npz += f'_Nth_{self.N_th}'

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
    def adiabatic_projection_Nth(self):
        """Project onto E0 subspace from theta = 0 --> 2\pi for a certain number
        of theta steps from 0->2pi
        """

        path_npz = self.make_path_npz(pre_str='res')

        if path_npz.is_file() and not self.create_new:
            return [
                np.load(path_npz)['theta_list'],
                np.load(path_npz)['vdot_lol'],
                np.load(path_npz)['theta_snaplist'],
                np.load(path_npz)['E0_state_snaplist'],
            ]

        if self.load_only:
            return None, None, None, None

        #-----------------------------------------------------------------------
        # initialize, get E0-states for theta=0
        #-----------------------------------------------------------------------
        E0_th0 = self.get_E0_subspace_th0()

        #-----------------------------------------------------------------------
        # project on E0-states
        #-----------------------------------------------------------------------
        E0_old = np.copy(E0_th0)


        theta_snaplist = np.array([i/257*(2*np.pi) for i in range(0, 258)])

        theta_linsp = [*theta_snaplist] + [*np.linspace(0*np.pi, 2*np.pi, self.N_th)]
        theta_linsp = np.array(sorted(list(set(theta_linsp))))
        theta_zeros = theta_linsp * 0

        theta_vec_list = []
        for site_i in range(1, self.L):
            if site_i in self.vary_theta_sites:
                theta_vec_list.append(theta_linsp)
            else:
                theta_vec_list.append(theta_zeros)
        theta_vec_list = np.array(theta_vec_list)
        
        E0_state_snaplist = [E0_old]



        vdot_lol = []
        for theta_vec in theta_vec_list.T:

            #---------------------------------------------------------------
            # solve hamilt
            if np.abs(theta_vec[self.vary_theta_sites[0]-1] - 2*np.pi) < 1e-9:
                print('here start', theta_vec/np.pi)
                E0_new = np.copy(E0_th0)
            else:
                hamilt_class = AnyonHubbardHamiltonian(
                    bc=self.bc,
                    L=self.L,
                    N=self.N,
                    J=self.J,
                    U=0,
                    theta=theta_vec,
                    bool_save=False
                )
                gc.collect()

                E0_new = hamilt_class.get_eigenstates_E0()


            #---------------------------------------------------------------
            # make projection
            E0_projected = []
            for E0_old_i in E0_old:
                E0_proj_i = 0
                for E0_new_i in E0_new:
                    E0_proj_i += E0_new_i * np.vdot(E0_new_i, E0_old_i)

                E0_projected.append(E0_proj_i)


            E0_old = np.array(E0_projected)
            # norm vectors
            # E0_dth = [el/np.linalg.norm(el) for el in E0_dth]

            if theta_vec[self.vary_theta_sites[0]-1] in theta_snaplist[1:]:
                E0_state_snaplist.append(E0_old)

            if theta_vec[self.vary_theta_sites[0]-1] in theta_snaplist:
                print('th/pi', np.round(theta_vec/np.pi, 8))


            vdot_list = []
            for E0_th0_i, E0_proj_i in zip(E0_th0, E0_projected):
                vdot_list.append(np.vdot(E0_th0_i, E0_proj_i))
            vdot_lol.append(vdot_list)



        # plt.plot(theta_list, vdot_list)
        # plt.plot(theta_list, ana_list)
        # plt.show()
        # exit()

        vdot_lol = np.array(vdot_lol)

        print(np.array(E0_state_snaplist).shape)

        E0_state_snaplist = np.array(E0_state_snaplist)

        path_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path_npz,
            theta_snaplist=theta_snaplist,
            E0_state_snaplist=E0_state_snaplist,
            vdot_lol=vdot_lol,
            theta_list=theta_vec_list[self.vary_theta_sites[0]-1]
        )

        return [
            np.load(path_npz)['theta_list'],
            np.load(path_npz)['vdot_lol'],
            np.load(path_npz)['theta_snaplist'],
            np.load(path_npz)['E0_state_snaplist'],
        ]




    #===========================================================================
    # analyze psi(t)
    #===========================================================================
    def get_holonomy_mat(self):
        """Calculate overlap between E0 states in energy basis
        M_ij = <phi_i(theta=0) | \Prod P(2pi)...P(0) phi_i(theta=0)>
        """

        _, _, th_snap_list, E0_state_snaplist = self.adiabatic_projection_Nth()

        if E0_state_snaplist is None:
            return None, None, None


        E0_th0 = self.get_E0_subspace_th0()

        assert th_snap_list[0] == 0 and th_snap_list[-1] == 2*np.pi

        E0_th0 = E0_state_snaplist[0]
        E0_th2pi = E0_state_snaplist[-1]

        n_vecs = E0_th0.shape[0]
        holo_mat = np.zeros((n_vecs, n_vecs), dtype=complex)
        for i in range(n_vecs):
            for j in range(n_vecs):
                holo_mat[i, j] = np.vdot(E0_th0[i], E0_th2pi[j])

        evals, evec = np.linalg.eig(holo_mat)

        return evals, evec, holo_mat

