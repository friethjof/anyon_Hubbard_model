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
from helper import operators



@njit
def get_psi_t(time, psi0, evals, evecs, length):
    psi_t = np.zeros((time.shape[0], length), dtype=np.complex128)
    for i, t in enumerate(time):
        # print(f't = {t:.1f}')
        psi_tstep = np.zeros(length, dtype=np.complex128)
        for eval_n, evec_n in zip(evals, evecs):

            # n_n = np.outer(evec_n, np.conjugate(evec_n))
            # psi_tstep += np.exp(-1j*eval_n*t)* n_n.dot(psi0)

            psi_tstep += np.exp(-1j*eval_n*t)*evec_n*np.vdot(evec_n, psi0)
        
        psi_t[i, :] = psi_tstep
    return psi_t


#===============================================================================
class PropagationStatic(AnyonHubbardHamiltonian):
    """Propagation of a given initial state.

    |psi(t)> = sum_n exp(-i*E_n*t) |n> <n|psi_0>
    n> : eigenvectors
    """

    def __init__(self, bc, L, N, J, U, theta, psi0_str, Tprop, dtprop,
                 bool_save=True, psi0_ini_class=None):
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
            bc=bc,
            L=L,
            N=N,
            J=J, 
            U=U,
            theta=theta,
            bool_save=bool_save
        )

        self.bool_save = bool_save

        if Tprop is not None:
            self.Tprop = Tprop
            self.dtprop = dtprop
            self.time = np.arange(0, Tprop+dtprop, dtprop)
            self.dum_Tf_dt = path_dirs.get_dum_Tf_dt(Tprop, dtprop)

        self.psi0_str = psi0_str
        self.psi0_ini_class = psi0_ini_class


        # get initial state
        if psi0_str is not None and self.psi0_ini_class is None:

            if bool_save:
                self.path_prop = self.path_basis/psi0_str
                self.path_prop.mkdir(parents=True, exist_ok=True)

            self.psi0, self.nstate0_str = initialize_prop.get_psi0(
                psi0_str, self.basis.basis_list, self.evecs())


        elif self.psi0_ini_class is not None:  # parameter quench implied!
            self.psi0_quench_str = initialize_prop.get_psi0_quench_str(
                psi0_str=psi0_str,
                hamil_ini = self,
                hamil_prop = psi0_ini_class
            )

            self.psi0, self.nstate0_str = initialize_prop.get_psi0(
                psi0_str, self.basis.basis_list, self.psi0_ini_class.evecs())

            if bool_save:
                self.path_prop = self.path_basis/self.psi0_quench_str
                self.path_prop.mkdir(parents=True, exist_ok=True)

        self._psi_t = None



    #===========================================================================
    # propagate intial state
    #===========================================================================
    def make_propagation(self):
        if self.bool_save:
            path_psi_npz = self.path_prop/f'psi_{self.dum_Tf_dt}.npz'

        # if False:
        if self.bool_save and (path_psi_npz.is_file() and
                np.load(path_psi_npz)['Tprop'] == self.Tprop and
                np.load(path_psi_npz)['dtprop'] == self.dtprop):
            self._psi_t = np.load(path_psi_npz)['psi_t']
        else:
            # make propagation
            length=self.basis.length

            psi_t = get_psi_t(
                time=self.time,
                psi0=self.psi0.astype(np.complex128),
                evals=self.evals(),
                evecs=self.evecs(),
                length=length
            )


            # psi_t_test = []
            # for t in self.time:
            #     # print(f't = {t:.1f}')
            #     psi_tstep = np.zeros(self.basis.length, dtype=complex)
            #     for eval_n, evec_n in zip(self.evals(), self.evecs()):

            #         psi_tstep += (
            #             np.exp(-1j*eval_n*t)
            #             *evec_n*np.vdot(evec_n, self.psi0)
            #         )
                
            #     psi_t_test.append(psi_tstep)
            #     assert np.abs(np.vdot(psi_tstep, psi_tstep) - 1+0j) < 1e-8
            # assert np.allclose(psi_t, psi_t_test)
            # exit()

            self._psi_t = np.array(psi_t)

            if self.bool_save:
                np.savez(
                    path_psi_npz,
                    time=self.time,
                    psi_t=self._psi_t,
                    psi0=self.psi0,
                    psi0_str=self.psi0_str,
                    nstate0_str=self.nstate0_str,
                    Tprop=self.Tprop,
                    dtprop=self.dtprop,
                    L=self.L,
                    N=self.N,
                    J=self.J,
                    U=self.U,
                    theta=self.theta,
                    basis_list=self.basis.basis_list
                )


    def psi_t(self):
        if self._psi_t is None:
            self.make_propagation()
        return self._psi_t


    #===========================================================================
    # number operator
    #===========================================================================
    def num_op_site(self, site_i):
        "psi_t = <psi(t)| b_i^t b^i |psi(t)>"
        path_nop_dict = self.path_prop/f'dict_numOp_{self.dum_Tf_dt}.npz'
        if path_nop_dict.is_file():
            nop_dict = dict(np.load(path_nop_dict))
            time = nop_dict['time']
            if abs(time[-1] - self.time[-1]) < 1e-6:
                if f'site_{site_i}' in nop_dict.keys():
                    return time, nop_dict[f'site_{site_i}']
            else:
                nop_dict = {}
        else:
            nop_dict = {}

        nop = []
        for psi_tstep in self.psi_t():
            nop_t = 0
            for psi_coeff, b_m in zip(psi_tstep, self.basis.basis_list):
                nop_t += np.abs(psi_coeff)**2*b_m[site_i]
            nop.append(nop_t)
        nop = np.array(nop)
        assert np.max(np.abs(nop.imag)) < 1e-8
        nop = nop.real

        nop_dict[f'site_{site_i}'] = nop
        nop_dict[f'time'] = self.time
        if self.bool_save:
            np.savez(path_nop_dict, **nop_dict)
        return self.time, nop


    def num_op_mat(self):
        """Return lists which are easy to plot"""
        L = self.L
        expV_list, label_list = [], []
        for i in range(L):
            time, expV = self.num_op_site(i)
            expV_list.append(expV)
        return time, np.array(expV_list)


    def get_nop_time(self, t_instance):
        time, expV_list = self.num_op_mat()
        time_ind = np.argmin(np.abs(time - t_instance))
        return np.array(expV_list)[:, time_ind]



    #===========================================================================
    # number state projection
    #===========================================================================
    def nstate_projection(self):
        """Get number state list: [2, 0, 0, 2]
        Get the overlap with psi_prop, i.e. return the time-dependent
        coefficient corresponding to this state"""

        nstate_mat = np.abs(self.psi_t())**2

        count_nstate = len([el for el in nstate_mat.T if np.max(np.abs(el))>1e-10])
        # print('number of number states which are nonzero', count_nstate)

        return nstate_mat


    def nstate_SVN(self):
        path_dict = self.path_prop/f'dict_nstate_SVN_{self.dum_Tf_dt}.npz'
        if path_dict.is_file():
            obs_dict = dict(np.load(path_dict))
            return obs_dict['svn'], obs_dict['svn_max_val']
        else:
            obs_dict = {}
        nstate_mat = self.nstate_projection()

        svn = []
        for i in range(nstate_mat.shape[0]):
            svn.append(sum([-el*np.log(el) for el in nstate_mat[i, :]
                if el > 0]))
        svn = np.array(svn)

        svn_max_val = np.log(nstate_mat.shape[1])

        obs_dict['time'] = self.time
        obs_dict['svn'] = svn
        obs_dict['svn_max_val'] = svn_max_val
        if self.bool_save:
            np.savez(path_dict, **obs_dict)

        return svn, svn_max_val


    def nstate_SVN_horizontal_fit(self, tmin=30, tmax=None):
        svn, svn_max_val = self.nstate_SVN()

        def hline(t, a):
            return a

        tind_min = np.argmin(np.abs(self.time - tmin))
        if tmax is not None:  # take last time step
            tind_max = np.argmin(np.abs(self.time - tmax))
        else:
            tind_max = None

        svn_cut = svn[tind_min:tind_max]
        popt, pcov = scipy.optimize.curve_fit(
            f=hline,
            xdata=self.time[tind_min:tind_max],
            ydata=svn_cut,
            p0=svn_max_val)
        svn_fit = popt[0]
        # svn_err = np.sqrt(np.sum(np.abs(svn_cut - svn_fit)**2)/svn_cut.shape[0])
        svn_err = np.sum(np.abs(svn_cut - svn_fit)**2)/svn_cut.shape[0]

        return svn_fit, svn_err, svn_max_val


    #===========================================================================
    # loschmidt echo:  M (t) = |<psi0 | exp(iHt) exp(-iHt) | psi0> |^2
    #===========================================================================
    # def loschmidt_echo(self):
    #     """Calculate M (t) = |<psi0 | exp(iHt) exp(-iHt) | psi0> |^2
    #     https://arxiv.org/pdf/1206.6348 Eq. (1)
    #     """


    #     lo_echo = []
    #     for t, psi in zip(self.time, self.psi_t()):
    
    #         psi_tstep = np.zeros(self.basis.length, dtype=complex)
    #         for eval_n, evec_n in zip(self.evals(), self.evecs()):

    #             # |psi'(t)> = |evec> <evec| exp(i E_n t) |psi(t)>
    #             psi_tstep += (
    #                 np.exp(1j*eval_n*t)
    #                 *evec_n*np.vdot(evec_n, psi)
    #             )

    #         # M = |<psi(0)|psi'(t)>|^2
    #         M = np.sum(np.abs(np.vdot(self.psi_t()[0], psi_tstep))**2)
        
    #         lo_echo.append(M)
        
    #     return np.array(lo_echo)

