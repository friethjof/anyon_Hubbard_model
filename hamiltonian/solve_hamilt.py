import os
import itertools
import shutil
import math
import time
from pathlib import Path
import subprocess
from collections.abc import Iterable

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.linalg

from hamiltonian.basis import Basis

from helper import path_dirs
from helper import operators
from helper import other_tools
from helper import palindromic_number_states

# author: Friethjof Theel
# date: 15.06.2022
# last modified: Feb 2023


#===============================================================================
class AnyonHubbardHamiltonian():
    """
    H = - sum_j^L-1 J_j b_j^t exp(i*theta*n_j) b_j+1
                    + b_j+1^t exp(-i*theta*n_j) b_j
        + 1/2 sum_j^L U_j n_j(n_j-1)

    bosonic operators
    b_j   | n > = sqrt(n)   | n-1 >
    b_j^t | n > = sqrt(n+1) | n+1 >
    """

    def __init__(self, **args_in):
        
        self.bc = args_in.get('bc', 'open')
        self.L = args_in.get('L')
        self.N = args_in.get('N')
        self.J = args_in.get('J')
        self.U = args_in.get('U')
        self.theta = args_in.get('theta')
        self.basis = Basis(self.L, self.N)
        self.bool_save = args_in.get('bool_save', True)

        self._hamilt = None
        self._evals = None
        self._evecs = None

        self.path_basis = path_dirs.get_path_basis(
            self.bc, self.L, self.N, self.U, self.theta, self.J,
            bool_mkdir=self.bool_save)


    #===========================================================================
    # build and solve hamiltonian
    #===========================================================================
    def make_diagonalization(self):
        path_hamil_npz = self.path_basis/'hamilt_spectrum.npz'

        if path_hamil_npz.is_file():
            self._hamilt = np.load(path_hamil_npz)['hamilt_mat']
            self._evals = np.load(path_hamil_npz)['evals']
            self._evecs = np.load(path_hamil_npz)['evecs']
            return None

        #=======================================================================
        # do calculation

        # measure ellapsed time
        if self.bool_save:
            print('basis size:', self.basis.length)
            clock_1 = time.time()
        #-------------------------------------------------------------------
        # calculate all matrix elements: H_mn = <m| H | n>


        if isinstance(self.theta, Iterable):
            theta_list = self.theta
        else:
            if self.bc == 'open':
                theta_list = [self.theta]*(self.L-1)
            else:
                theta_list = [self.theta]*self.L

        if isinstance(self.J, Iterable):
            J_list = self.J
        else:
            if self.bc == 'open':
                J_list = [self.J]*(self.L-1)
            else:
                J_list = [self.J]*self.L

        if isinstance(self.U, Iterable):
            U_list = self.U
        else:
            U_list = [self.U]*self.L

        self._hamilt = operators.get_hamilt_mat(
            bc = self.bc,
            L = self.L,
            J_list = np.array(J_list),
            U_list = np.array(U_list),
            N = self.N,
            theta_list = np.array(theta_list),
            basis_list = self.basis.basis_list
        )

        # for i, b_m in enumerate(self.basis.basis_list):
        #     for j, b_n in enumerate(self.basis.basis_list):

        #         self._hamilt[i, j] = operators.get_hamilt_mn(
        #             bc = self.bc,
        #             L = self.L,
        #             J_list = J_list,
        #             U_list = U_list,
        #             N = self.N,
        #             theta_list = theta_list,
        #             b_m = b_m,
        #             b_n = b_n,
        #         )


        # check whether hamiltonian is hermitian
        assert np.allclose(self._hamilt, np.conjugate(self._hamilt.T))
        if self.bool_save:
            clock_2 = time.time()
            print('matrix hamiltonian created, '
                f'time: {other_tools.time_str(clock_2-clock_1)}')
        # Do not use np.linalg.eig here!
        # --> it might be that eigenvectors of degenerate eigenvalues
        # are not orthogonal!
        eval, evec = np.linalg.eigh(self._hamilt)
        idx = eval.argsort()
        self._evals = eval[idx]
        self._evecs = (evec[:, idx]).T

        # assert all([(np.vdot(el, el) - 1) < 1e-8 for el in self._evecs])
        assert np.max(np.abs(self._evals.imag)) < 1e-8
        self._evals = self._evals.real

        if self.bool_save:
            clock_3 = time.time()
            print(f'matrix hamiltonian has been diagonalized, '
                f'time: {other_tools.time_str(clock_3-clock_2)}')
        # check orthonormality
        ortho_bool = True
        for i in range(self.basis.length):
            for j in range(self.basis.length):
                sp = np.vdot(self._evecs[i], self._evecs[j])
                if i == j and np.abs(sp - 1) > 1e-10:
                    ortho_bool = False

                if i != j and np.abs(sp) > 1e-10:
                    ortho_bool = False
        assert ortho_bool


        if self.bool_save:
            clock_4 = time.time()
            print(f'orthonormality has been checked, '
                f'time: {other_tools.time_str(clock_4-clock_3)}')


        if self.bool_save:
            np.savez(path_hamil_npz,
                hamilt_mat=self._hamilt,
                evals=self._evals,
                evecs=self._evecs,
                basis_list=self.basis.basis_list,
                basis_length=self.basis.length,
                L=self.L,
                N=self.N,
                J=self.J,
                U=self.U,
                theta=self.theta
            )
            print(path_hamil_npz.stat().st_size)
            print(f'File size: {path_hamil_npz.stat().st_size/(1024**2):.3f} MB')

        if self.bool_save:
            clock_5 = time.time()
            print(f'Total time consumption: '
                f'{other_tools.time_str(clock_5-clock_1)}')


    def hamilt(self):
        if self._hamilt is None:
            self.make_diagonalization()
        return self._hamilt


    def evals(self):
        if self._evals is None:
            self.make_diagonalization()
        return self._evals


    def evecs(self):
        if self._evecs is None:
            self.make_diagonalization()
        return self._evecs


    #===========================================================================
    # E0 degeneracy
    #===========================================================================
    def energy_degeneracy(self):
        """Determine the degeneracies of the energies.

        Returns
        dict : degeneracies
        """

        return other_tools.find_degeneracies(self.evals())


    def get_eigenstates_E0(self):
        """Get eigenstates which are degenerated with an eigenenergy E=0
        """

        dict_degen = other_tools.find_degeneracies(self.evals())
        E0_found = False
        for k, v in dict_degen.items():
            if abs(eval(k)) < 1e-10:
                ind_list = v
                E0_found = True
                break

        if not E0_found:
            raise ValueError
        evevs_E0 = [self.evecs()[i] for i in ind_list]

        return np.array(evevs_E0)


    def get_E0_chirality(self, bool_any_E0=False):
        "Calculate chirality of E0-states"

        if self.theta == 0 and not bool_any_E0:
            evevs_E0 = self.get_E0_eigenstate_energy_basis()
        else:
            evevs_E0 = self.get_eigenstates_E0()

        S_operator = operators.get_chiral_operator(self.basis.basis_list)

        chiral_exp = []
        for psi in evevs_E0:
            c_val = np.vdot(psi, (S_operator.dot(psi)))
            assert np.max(np.abs(c_val.imag)) < 1e-9
            chiral_exp.append(c_val.real)

        return chiral_exp


    #===========================================================================
    # Energy (chiral pair basis)
    #===========================================================================
    def get_energy_occup_basis(self, occupation_list):
        """Obtain energy occupation number basis
        Single particle eigenstates:
            c_{\nu,j} = sqrt(2/(L+1)) sin(pi*\nu*j/(L+1))
            b'_\nu = \sum_{j=1}^L c_{\nu,j} b_j
            |{\eta_1, ..., \eta_L}> =
                    \Prod_{\nu=1}^L (b'^dagger)^{\eta_\nu}/sqrt(\eta_\nu!) |0>

            Then return |{\eta_1, ..., \eta_L}> in number basis
        """

        assert len(occupation_list) == self.L
        assert sum(occupation_list) == self.N

        C_tensor = np.zeros((self.basis.length, self.basis.length))
        for i in range(self.basis.length):
            for j in range(self.basis.length):
                c_ij = np.sqrt(2/(self.L + 1)) * np.sin(np.pi*(i+1)*(j+1)/(self.L + 1))
                C_tensor[i, j] = c_ij


        #-----------------------------------------------------------------------
        def apply_cij_bj(i, j, coeff, number_list):
            n_list_out = number_list.copy()
            coeff *= C_tensor[i, j]*math.sqrt(number_list[j] + 1)
            n_list_out[j] += 1
            return coeff, n_list_out


        #-----------------------------------------------------------------------
        def apply_b_tilde(i, n_vec_in, basis_in):
            """Apply \tilde{b}
                --> creates one particle
                --> update number state basis
            """

            N_new = basis_in.N + 1
            basis_new = Basis(self.L, N_new)
            n_vec_new = np.zeros(basis_new.length)
            for coeff, n_list in zip(n_vec_in, basis_in.basis_list):
                for j in range(self.L):
                    coeff_new, n_list_new = apply_cij_bj(i, j, coeff, n_list)

                    # add new coeff to new basis state
                    ind = basis_new.nstate_index(n_list_new)
                    n_vec_new[ind] += coeff_new
            return n_vec_new, basis_new


        #-----------------------------------------------------------------------
        basis = Basis(self.L, 0)
        n_vec = np.array([1.0])
        for i, n in enumerate(occupation_list):
            for _ in range(n):
                n_vec, basis = apply_b_tilde(i, n_vec, basis)
            n_vec *= 1/np.sqrt(math.factorial(n))

        assert basis.length == self.basis.length
        assert abs(1 - np.vdot(n_vec, n_vec)) < 1e-9
        assert np.sum(np.abs(self.hamilt().dot(n_vec))) < 1e-9

        # print(np.vdot(n_vec, self.hamilt().dot(n_vec)))

        return n_vec


    def ovlp_energy_occup_basis_E0_subspace(self):
        """get all |{\eta_1, ..., \eta_L}> in nstate representation and
        calculate overlap with E0 subspace"""

        evecs_E0 = self.get_eigenstates_E0()

        E0_overlap = []
        for nstate in self.basis.basis_list:

            ebasis = self.get_energy_occup_basis(nstate)

            E0_overlap.append(
                np.sum([np.abs(np.vdot(ebasis, E0_evec))**2 for E0_evec in evecs_E0])
            )


        return np.array(E0_overlap)


    def get_E0_eigenstate_energy_basis(self):

        num_states_lol = palindromic_number_states.get_lists(L=self.L, N=self.N)

        E0_state_ebasis = []
        for num_states in num_states_lol:
            E0_state_ebasis.append(self.get_energy_occup_basis(num_states))

        return np.array(E0_state_ebasis)


    #===========================================================================
    # Translational operator
    #===========================================================================
    def diag_H_and_transl_op(self):
        """
        Find basis C which diagonalizes simultaneously the Hamiltonian H and the 
        translation operator U, make first sure that they commute:
        HU - UH = 0
        Return lists of diagonal elements:
        D_h = C^-1 H C
        D_u = C^-1 U C
        """

        path_npz = self.path_basis/'eigenvalues_Hamil_transl_op_same_basis.npz'
        if path_npz.is_file():
            return np.load(path_npz)['diag_H'], np.load(path_npz)['diag_U']


        H_mat = self.hamilt()
        H_evec = self.evecs()



        U_mat = operators.get_translation_op(self.basis.basis_list)


        assert np.allclose(U_mat@H_mat - H_mat@U_mat, np.zeros_like(H_mat))


        dict_degen = other_tools.find_degeneracies(self.evals())
        basis = []
        for k, v in dict_degen.items():
            # v holds list of indices corresponding to eigenvalue k
            if len(v) == 1:  # -> degenerate
                basis.append(H_evec[v[0]])
            else:
                # diagonalize set of degenerate eigenvectors in K
                X_mat = np.array([H_evec[i] for i in v])
                C_mat = X_mat.conjugate() @ U_mat @ X_mat.T
                C_eval, C_evec = scipy.linalg.eig(C_mat)
                C_evec = (C_evec[:, C_eval.argsort()]).T
                Y_mat = C_evec @ X_mat
                for y_arr in Y_mat:
                    basis.append(y_arr)

        basis = np.array(basis).T
        basis_inv = np.linalg.inv(basis)

        D_mat_U = basis_inv @ U_mat @ basis
        D_mat_H = basis_inv @ H_mat @ basis

        if not np.allclose(D_mat_U, np.diag(np.diagonal(D_mat_U)), rtol=1e-05, atol=1e-06):
            print(D_mat_U)
            print(np.sum(np.abs(D_mat_U - np.diag(np.diagonal(D_mat_U)))))
            print(self.theta/np.pi)
            raise ValueError('Dmat U is not diagonal!')
        if not np.allclose(D_mat_H, np.diag(np.diagonal(D_mat_H)), rtol=1e-05, atol=1e-06):
            print(D_mat_H)
            print(np.sum(np.abs(D_mat_U - np.diag(np.diagonal(D_mat_U)))))
            print(self.theta/np.pi)
            raise ValueError('Dmat H is not diagonal!')

        # np.savez(
        #     path_npz,
        #     diag_U=np.diagonal(D_mat_U),
        #     diag_H=np.diagonal(D_mat_H)
        # )

        # length = self.basis.length
        # x, y = np.meshgrid(np.arange(length), np.arange(length))
        # im = plt.pcolormesh(x, y, np.abs(D_mat_H.T), shading='auto')
        # plt.colorbar(im)
        # plt.show()

        return np.diagonal(D_mat_H), np.diagonal(D_mat_U)



    def save_evals_in_same_U_sector(self):
        diag_H, diag_U = self.diag_H_and_transl_op()
        U_dict_degen = other_tools.find_degeneracies(diag_U)


        # combine U which correspond to 

        # print(len(U_dict_degen.keys()))
        # angle_list = []
        # for k, v in U_dict_degen.items():
        #     u_angle = np.round(np.angle(eval(k))/np.pi, 8)
        #     if u_angle == -1.0:
        #         u_angle = 1.0
        #     angle_list.append(u_angle)

        # # find index corresponding to maximum degeneracy of U eigenvalues
        # ind_max_len = np.argmax([len(el) for el in U_dict_degen.values()])

        # get eigenvalue of U with value of one!
        ind_max_len = np.argmin([np.abs(eval(el)-1) for el in U_dict_degen.keys()])
        print(ind_max_len)

        # determine index list with the positions where diag_U has this eigenvalues
        ind_list = [v for i, (k, v) in enumerate(U_dict_degen.items()) if i == ind_max_len][0]
        # find corresponding eigenvalues of H which have the same u
        h_list = diag_H[ind_list]
        assert np.max(np.abs(h_list.imag)) < 1e-10
        h_list = h_list.real

        print('length', len(ind_list), 'eig val:', list(U_dict_degen.keys())[ind_max_len])

        assert self.bc == 'twisted_gauge'
        path_txt = Path('/afs/physnet.uni-hamburg.de/users/zoq_t/'
                        'ftheel/Schreibtisch/project_ahm2/level_statistics/'
                        'data_gtbc')
        path_name = f'L{self.L}_N{self.N}_U{round(self.U,8)}_thpi_{round(self.theta/np.pi, 8)}'.replace('.', '_')
        with open(path_txt/path_name, 'w+') as f:
            for energy in h_list:
                f.write(str(round(energy, 16)) + '\n')
        # exit()


    #===========================================================================
    # U operator
    #===========================================================================
    def diag_H_and_U_I_op(self):
        """
        Find basis C which diagonalizes simultaneously the Hamiltonian H and the 
        translation operator U, make first sure that they commute:
        HU - UH = 0
        Return lists of diagonal elements:
        D_h = C^-1 H C
        D_u = C^-1 U C
        """

        path_npz = self.path_basis/'eigenvalues_Hamil_U_I_op_same_basis.npz'
        if path_npz.is_file():
            return np.load(path_npz)['diag_H'], np.load(path_npz)['diag_U']


        H_mat = self.hamilt()
        H_evec = self.evecs()

        I_op = operators.get_inv_op(self.basis.basis_list)
        U_op = operators.get_U_op(self.basis.basis_list, self.theta)

        U_I_mat = U_op@I_op

        assert np.allclose(U_I_mat@H_mat - H_mat@U_I_mat, np.zeros_like(H_mat))


        dict_degen = other_tools.find_degeneracies(self.evals())
        basis = []
        for k, v in dict_degen.items():
            # v holds list of indices corresponding to eigenvalue k
            if len(v) == 1:  # -> degenerate
                basis.append(H_evec[v[0]])
            else:
                # diagonalize set of degenerate eigenvectors in K
                X_mat = np.array([H_evec[i] for i in v])
                C_mat = X_mat.conjugate() @ U_I_mat @ X_mat.T
                C_eval, C_evec = scipy.linalg.eig(C_mat)
                C_evec = (C_evec[:, C_eval.argsort()]).T
                Y_mat = C_evec @ X_mat
                for y_arr in Y_mat:
                    basis.append(y_arr)

        basis = np.array(basis).T
        basis_inv = np.linalg.inv(basis)

        D_mat_U = basis_inv @ U_I_mat @ basis
        D_mat_H = basis_inv @ H_mat @ basis

        if not np.allclose(D_mat_U, np.diag(np.diagonal(D_mat_U)), rtol=1e-05, atol=1e-06):
            print(D_mat_U)
            print(np.sum(np.abs(D_mat_U - np.diag(np.diagonal(D_mat_U)))))
            print(self.theta/np.pi)
            raise ValueError('Dmat U is not diagonal!')
        if not np.allclose(D_mat_H, np.diag(np.diagonal(D_mat_H)), rtol=1e-05, atol=1e-06):
            print(D_mat_H)
            print(np.sum(np.abs(D_mat_U - np.diag(np.diagonal(D_mat_U)))))
            print(self.theta/np.pi)
            raise ValueError('Dmat H is not diagonal!')

        np.savez(
            path_npz,
            diag_U=np.diagonal(D_mat_U),
            diag_H=np.diagonal(D_mat_H)
        )

        # length = self.basis.length
        # x, y = np.meshgrid(np.arange(length), np.arange(length))
        # im = plt.pcolormesh(x, y, np.abs(D_mat_H.T), shading='auto')
        # plt.colorbar(im)
        # plt.show()
        # exit()

        return np.diagonal(D_mat_H), np.diagonal(D_mat_U)


    def save_evals_in_same_UI_sector(self, path_txt):
        diag_H, diag_U = self.diag_H_and_U_I_op()
        U_dict_degen = other_tools.find_degeneracies(diag_U)

        # combine U which correspond to 

        # print(len(U_dict_degen.keys()))
        # angle_list = []
        # for k, v in U_dict_degen.items():
        #     u_angle = np.round(np.angle(eval(k))/np.pi, 8)
        #     if u_angle == -1.0:
        #         u_angle = 1.0
        #     angle_list.append(u_angle)

        # find index corresponding to maximum degeneracy of U eigenvalues
        index_key = np.argmax([len(el) for el in U_dict_degen.values()])
        # index_key = np.argmin([len(el) for el in U_dict_degen.values()])


        # determine index list with the positions where diag_U has this eigenvalues
        ind_list = [v for i, (k, v) in enumerate(U_dict_degen.items()) if i == index_key][0]
        # find corresponding eigenvalues of H which have the same u
        h_list = diag_H[ind_list]
        assert np.max(np.abs(h_list.imag)) < 1e-10
        h_list = h_list.real

        print('length', len(ind_list), 'eig val:', list(U_dict_degen.keys())[index_key])

        assert self.bc == 'open'
        # path_txt = Path('/afs/physnet.uni-hamburg.de/users/zoq_t/'
        #                 'ftheel/Schreibtisch/project_ahm2/level_statistics/'
        #                 'data_obc')
        path_name = f'L{self.L}_N{self.N}_U{round(self.U,8)}_thpi_{round(self.theta/np.pi, 8)}'.replace('.', '_')
        with open(path_txt/path_name, 'w+') as f:
            for energy in h_list:
                f.write(str(round(energy, 16)) + '\n')



    #===========================================================================
    # hopping operator
    #===========================================================================
    def get_hopping_EEV(self, site_i):
        """
        return list of energy expectation values
        <psi_n | 
            b_i^\dagger exp(i*theta*n_j) b_{i+1} 
            + b_{i+1}^\dagger exp(-i*theta*n_j) b_i | psi_n>
        """


        # if self.bc == 'open':
        assert site_i < self.L -1

        hop_op_i = operators.get_hop_op_i(
            basis_list=self.basis.basis_list,
            site_i=site_i,
            theta=self.theta
        )


        eev_list = []
        for evec in self.evecs():
            expval = np.vdot(evec, np.dot(hop_op_i, evec))
            assert np.abs(expval.imag) < 1e-10
            eev_list.append(expval.real)

        return np.array(eev_list)
    

