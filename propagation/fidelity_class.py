import os
import itertools
import shutil
import math
import time
from pathlib import Path
import subprocess

import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.optimize
from numba import njit

from helper import path_dirs
from propagation import initialize_prop
from hamiltonian.solve_hamilt import AnyonHubbardHamiltonian
from propagation.propagate_static import PropagationStatic
from helper import operators



#===============================================================================
class FidelityAveraged(PropagationStatic):
    """Propagation of a given initial state.

    |psi(t)> = sum_n exp(-i*E_n*t) |n> <n|psi_0>
    n> : eigenvectors
    """

    def __init__(self, **args_in):
        """Initialize parameters.

        Args:
            bc (str): boundary conditions of lattice, open (hard-wall) | periodic
            L (int): number of lattice sites
            N (int): number of atoms
            J (float): hopping constant
            U (float): on-site interaction
            theta (float): statistical angle

            psi0_str (str): specifies initial state for propagation
            Tprop (float): final time of propagation
            dtprop (float): time steps of propagation
            _psi_t (arr): place holder for psi-file

        """

        super().__init__(
            bc=args_in.get('bc', None),
            L=args_in.get('L', None),
            N=args_in.get('N', None),
            J=args_in.get('J', None), 
            U=args_in.get('U', None),
            theta=args_in.get('theta', None),
            psi0_str=args_in.get('psi0_str', None),
            Tprop=args_in.get('Tprop', None),
            dtprop=args_in.get('dtprop', None),
            bool_save=args_in.get('bool_save', None),
            psi0_ini_class=args_in.get('psi0_ini_class', None),
        )

        self.create_fid_new = args_in.get('create_fid_new', False)

        # state avergae parameters
        self.N_rand = args_in.get('N_rand')
        assert self.N_rand < 5000

        # time average parameters
        self.t_bin_size = args_in.get('t_bin_size', 10)
        self.t_bin_delta = args_in.get('t_bin_delta', 0.05)


        # random states
        self.ini_states_dict = args_in.get('ini_states_dict')


        self.texp_1 = args_in.get('texp_1', -2)
        self.texp_2 = args_in.get('texp_2', 8)
        self.N_t = args_in.get('texp_2N_t', 300)
        self.time = np.logspace(self.texp_1, self.texp_2, self.N_t)

        self.path_dir_top = self.get_path_dir_top()

        self.temp_dict = {}



    #===========================================================================
    # define names and paths
    #===========================================================================
    def get_system_str(self, **args_in):
        theta = args_in.get('theta', self.theta)
        U = args_in.get('U', self.U)
        U_ = f'U_{round(float(self.U), 8)}'.replace('.', '_')
        th_ = f'thpi_{round(theta/np.pi, 8)}'.replace('.', '_')
        return U_, th_


    def get_ini_states_str(self):
        states_type = self.ini_states_dict['type']

        av_ = f"{states_type}"

        if  states_type == 'randstates_gauss':
            E_mid = self.ini_states_dict['E_mid']
            E_range = self.ini_states_dict['E_range']
            av_ = f"randstates_gauss_Emid_{E_mid}_Erange_{E_range}"
            av_ = av_.replace('.', '_')

        return av_


    def get_time_str(self):
        t_str = f'time_t10x{self.texp_1}_10x{self.texp_2}_N_t_{self.N_t}'
        tbin_ = f'tbin_{int(self.t_bin_size)}_tdelta_{round(self.t_bin_delta, 8)}'.replace('.', '_')
        return f'{t_str}_{tbin_}'


    def get_path_dir_top(self, **args_in):
        "Path in which data.npz is stored"

        dir_save = Path('/afs/physnet.uni-hamburg.de/project/zoq_t/'
                        'harm_osc_coll/13/project_ahm_integrability')
        if self.bc == 'open':
            dir_save = dir_save/'open_boundary_conditions'
        else:
            raise NotImplementedError(self.bc)

        av_ini_states_ = self.get_ini_states_str()

        U_, th_ = self.get_system_str(**args_in)
        t_str = self.get_time_str()

        dir_save = (
            dir_save/
            'fidelity_v2'/
            av_ini_states_/
            f'L{self.L}_N{self.N}_{U_}_{th_}'/
            t_str
        )
        dir_save.mkdir(parents=True, exist_ok=True)

        return dir_save


    def get_path_npz_fidelity_av(self):
        return self.get_path_dir_top()/f'fidelity_average_Nr_{self.N_rand}.npz'


    def get_path_npz_rand_state_i(self, i):
        """Define path where to store one time-averaged fidelity obtained for
        initial state i"""
        return self.path_dir_top/f'fidelity_time_av_i{i}.npz'


    #===========================================================================
    # helper fcts
    #===========================================================================
    def get_fidelity(self):
        "calculate fidelity"

        fidelity = []
        for t, psi_t in zip(self.time, self.psi_t()):
            f_t = np.abs(np.vdot(self.psi0, psi_t))**2
            # print('time:', t, 'fidelity:',  f_t)
            fidelity.append(f_t)

        return np.array(fidelity).real


    def get_expanded_time_array(self):
        """To do the time average, more time instances need to be evaluated.
        Create here the time expanded time array for the t-average.
        """
        time_points = np.copy(self.time)
        time_av = np.sort(np.array([
            np.linspace(t-t*self.t_bin_delta, t+t*self.t_bin_delta, self.t_bin_size+1)
            for t in time_points
        ]).flatten())

        time_av = np.array([el for el in time_av if el >= 0])
        return time_av


    def get_energy_window(self):
        "Restriction for the energy of the initial states"

        if self.ini_states_dict['type'] == 'randstates_gauss':
            E_range = self.ini_states_dict['E_range']*max(self.evals())
            E_mid = self.ini_states_dict['E_mid']*max(self.evals())
            energy_window = [
                E_mid - E_range,
                E_mid + E_range
            ]
        else:
            energy_window = [-0.1 + self.evals()[0], self.evals()[-1]+0.1]

        return energy_window


    def get_evecs_U0_th0(self):
        key_name = 'evecs_U0_th0'
        if key_name in self.temp_dict.keys():
            return self.temp_dict[key_name]

        hamil_class = AnyonHubbardHamiltonian(
            bc = self.bc,
            J = 1,
            L = self.L,
            N = self.N,
            U = 0,
            theta = 0,
            bool_save=True,
        )
        self.temp_dict[key_name] = hamil_class.evecs()
        return hamil_class.evecs()


    def get_uniform_random_vector_cmplx(self, i):
        "store random vector always in path(theta=0)"
        key_name = 'random_coeff_cmlpx'
        if key_name in self.temp_dict.keys():
            return self.temp_dict[key_name][i]

        path_rand_list_npz = self.get_path_dir_top(U=0, theta=0)/'random_coeff_cmlpx.npz'
        N_max = 5000  # number of random states

        if N_max <= i:
            raise ValueError('Too many random states considered')

        if path_rand_list_npz.is_file():
            rand_list = np.load(path_rand_list_npz)['rand_list']
            self.temp_dict[key_name] = rand_list
            return rand_list[i]

        rand_list = []
        for _ in range(N_max):
            rand_populations = (
                        np.random.uniform(0, 1, self.basis.length)
                + 1.j * np.random.uniform(0, 1, self.basis.length)
            )
            rand_list.append(rand_populations)

        np.savez(
            path_rand_list_npz,
            rand_list=rand_list
        )
        self.temp_dict[key_name] = rand_list
        return rand_list[i]


    def get_gauss_random_state(self, i):
        "store random vector always in path(theta=0)"
        key_name_energy = 'random_states_gauss_energy'
        key_name_state = 'random_states_gauss'
        key_name_coeff = 'random_coeffs_gauss'

        if key_name_state in self.temp_dict.keys():
            return [
                self.temp_dict[key_name_energy][i],
                self.temp_dict[key_name_state][i],
                self.temp_dict[key_name_coeff][i],
            ]

        path_rand_list_npz = self.get_path_dir_top(U=0, theta=0)/'random_states_gauss.npz'
        N_max = 5000  # number of random states

        if N_max <= i:
            raise ValueError('Too many random states considered')


        if path_rand_list_npz.is_file():
            e_list = np.load(path_rand_list_npz)['e_list']
            rand_state_list = np.load(path_rand_list_npz)['rand_state_list']
            rand_coeff_list = np.load(path_rand_list_npz)['rand_coeff_list']

            self.temp_dict[key_name_energy] = e_list
            self.temp_dict[key_name_state] = rand_state_list
            self.temp_dict[key_name_coeff] = rand_coeff_list

            return e_list[i], rand_state_list[i], rand_coeff_list[i]


        energy_window = self.get_energy_window()

        e_list, rand_state_list, rand_coeff_list = [], [], []
        j = 0
        while j < N_max:

            rand_loc = np.random.uniform(-1, 1, 1)[0]
            x = np.linspace(-1-rand_loc, 1-rand_loc, self.basis.length)
            sigma = 1
            window = 1/sigma*np.exp(-x**2/sigma**2)
            rand_pop_real = np.random.normal(loc=0, scale=window, size=self.basis.length)
            rand_pop_imag = np.random.normal(loc=0, scale=window, size=self.basis.length)
            rand_populations = rand_pop_real + 1.j*rand_pop_imag
            rand_vec = self.evecs().T.dot(rand_populations)
            rand_vec /= np.sqrt(np.vdot(rand_vec, rand_vec))

            energy = np.vdot(rand_vec, self.hamilt().dot(rand_vec)).real


            if energy_window[0] <= energy and energy <= energy_window[1]:
                j += 1
                e_list.append(energy)
                rand_coeff_list.append(rand_populations)
                rand_state_list.append(rand_vec)
            else:
                continue

        # plt.close()
        # plt.hist(e_list, bins=50, color='forestgreen')
        # plt.xlabel('energy')
        # plt.ylabel('bin')
        # plt.show()

        np.savez(
            path_rand_list_npz,
            rand_state_list=rand_state_list,
            rand_coeff_list=rand_coeff_list,
            e_list=e_list
        )

        self.temp_dict[key_name_energy] = e_list
        self.temp_dict[key_name_state] = rand_state_list
        self.temp_dict[key_name_coeff] = rand_coeff_list

        return e_list[i], rand_state_list[i], rand_coeff_list[i]



    def get_random_initial_state(self, i=None):
        "Get random initial state"

        states_type = self.ini_states_dict['type']

        # if states_type == 'nstates':
        #     rand_vec = np.zeros(ini_size, dtype=np.complex128)
        #     rand_vec[i] = 1
        # elif average_set['type'] == 'eigstates':
        #     rand_vec = self.evecs()[i]

        if states_type == 'randstates_uniform':
            rand_coeff = self.get_uniform_random_vector_cmplx(i)
            rand_vec = self.evecs().T.dot(rand_coeff)
            rand_vec /= np.sqrt(np.vdot(rand_vec, rand_vec))
            energy = np.vdot(rand_vec, self.hamilt().dot(rand_vec)).real


        elif states_type == 'randstates_uniform_U0_th0':
            rand_coeff = self.get_uniform_random_vector_cmplx(i)
            evecs_theta0 = self.get_evecs_U0_th0()
            rand_vec = evecs_theta0.T.dot(rand_coeff)
            rand_vec /= np.sqrt(np.vdot(rand_vec, rand_vec))
            energy = np.vdot(rand_vec, self.hamilt().dot(rand_vec)).real


        elif states_type == 'randstates_gauss':
            energy, rand_vec, rand_coeff = self.get_gauss_random_state(i)



        else:
            raise ValueError(states_type)

        # plt.close()
        # plt.hist(np.abs(rand_vec)**2)
        # plt.scatter(range(self.basis.length), np.abs(rand_coeff)**2)
        # plt.show()
        # exit()

        return energy, rand_vec


    #===========================================================================
    # main routine
    #===========================================================================
    def get_fidelity_rand_av(self):
        """"Calculate the fidelity averaged over N random progations return
        the moving time average, i.e., <<S(t)>>"""


        fidelity_av_npz = self.get_path_npz_fidelity_av()
        if not self.create_fid_new and fidelity_av_npz.is_file():
            np_load = np.load(fidelity_av_npz)
            return [
                np_load['fidelity_av'],
                np_load['e_list']
            ]

        #=======================================================================
        ini_size = self.basis.length
        time_points = np.copy(self.time)
        self.time = self.get_expanded_time_array()



        if self.N_rand is None:
            self.N_rand = ini_size


        fidelity_av_list = []
        e_list = []
        i = 0
        for i in range(self.N_rand):
            print(f'i={i+1}')

            path_npz_fid_i = self.get_path_npz_rand_state_i(i=i)

            if not self.create_fid_new and path_npz_fid_i.is_file():

                fid_tav_i = np.load(path_npz_fid_i)['fidelity_time_av']
                energy_i = np.load(path_npz_fid_i)['energy']

                fidelity_av_list.append(fid_tav_i)
                e_list.append(energy_i)

                i += 1
                continue




            energy, rand_vec = self.get_random_initial_state(i=i)

            #-------------------------------------------------------------------
            # calculate fieldity
            #-------------------------------------------------------------------
            self._psi_t = None
            self.psi0 = rand_vec

            fidelity_rand = self.get_fidelity()

            # make moving time avergae
            fidelity_time_av = []
            for t in time_points:
                t_min = max(0, t - t*self.t_bin_delta)
                t_max = t + t*self.t_bin_delta
                ind_min = np.argmin(np.abs(self.time - t_min))
                ind_max = np.argmin(np.abs(self.time - t_max))
                # t_slice = self.time[ind_min:ind_max+1]
                # print(t, (t_slice[-1] - t_slice[0])/2/t)
                f_bin = fidelity_rand[ind_min:ind_max+1]
                fidelity_time_av.append(np.mean(f_bin))


            np.savez(
                path_npz_fid_i,
                rand_vec=rand_vec,
                time=time_points,
                fidelity_time_av=fidelity_time_av,
                energy=energy
            )

            fidelity_av_list.append(fidelity_time_av)


        # plot energy
        if False:
            print(np.max(e_list) - np.min(e_list))
            plt.hist(e_list, bins=50, color='tomato')
            plt.show()
            plt.close()
            # # plt.hlines(energy_window, xmin=0, xmax=n_range, color='gray', ls='--')
            # plt.scatter(range(n_range), e_list)
            # plt.scatter(range(self.basis.length), self.evals())
            # plt.ylabel('energy')
            # plt.xlabel('i')
            # # plt.savefig('/home/friethjof/Downloads/rand_states_energy.png')
            # plt.show()

            # plt.close()
            # plt.scatter(e_list, np.ones(n_range))
            # plt.vlines(energy_window, ymin=0.5, ymax=1.5, color='gray', ls='--')
            # plt.show()
            # exit()

        # reset time
        self.time = time_points

        # calculate mean
        fidelity_av = np.mean(fidelity_av_list, axis=0)

        np.savez(
            fidelity_av_npz,
            fidelity_av=fidelity_av,
            e_list=e_list
        )

        return fidelity_av, e_list
    

    def get_infinite_time_av(self):
        fidelity_av, _ = self.get_fidelity_rand_av()

        ind_inf = np.argmin(np.abs(self.time - 1e5))
        inf_t_av = np.mean(fidelity_av[ind_inf:])

        return inf_t_av