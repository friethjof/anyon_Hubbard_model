import time
from pathlib import Path
from typing import Callable, List

import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate

from hamiltonian.solve_hamilt import AnyonHubbardHamiltonian
from propagation import initialize_prop

from helper import path_dirs
from helper import operators



#===============================================================================
class PropagationTD(AnyonHubbardHamiltonian):
    """Propagation of a given initial state."""

    def __init__(self, bc, L, N, J, U, theta_fct_list, psi0_str, Tprop, dtprop):
        """Initialize parameters.

        Args:
            bc (str): boundary conditions of lattice, open (hard-wall) | periodic
            L (int): number of lattice sites
            N (int): number of atoms
            J (float): hopping constant
            U (float|fct): on-site interaction
            theta (float|fct): statistical angle

            psi0_str (str): specifies initial state for propagation
            Tprop (float): final time of propagation
            dtprop (float): time steps of propagation
            _psi_t (arr): place holder for psi-file

        """

        if isinstance(U, Callable):
            self.U_fct = U
        else:
            self.U_fct = lambda t: U  # constact function  U_fct(t) = U

        if isinstance(J, Callable):
            self.J_fct = J
        else:
            self.J_fct = lambda t: J


        assert isinstance(theta_fct_list, List)
        assert all([isinstance(th_fct, Callable) for th_fct in theta_fct_list])
        self.theta_fct_list = theta_fct_list

        super().__init__(
            bc=bc,
            L=L,
            N=N,
            J=self.J_fct(0),
            U=self.U_fct(0),
            theta=[th_fct(0) for th_fct in theta_fct_list]
        )

        if Tprop is not None:
            self.Tprop = Tprop
            self.dtprop = dtprop
            self.time = np.arange(0, Tprop+dtprop, dtprop)
            self.dum_Tf_dt = path_dirs.get_dum_Tf_dt(Tprop, dtprop)

        self.psi0_str = psi0_str

        # get initial state
        if psi0_str is not None:
            self.path_prop = self.path_basis/psi0_str
            self.path_prop.mkdir(parents=True, exist_ok=True)
            self.psi0, self.nstate0_str = initialize_prop.get_psi0(
                psi0_str, self.basis.basis_list, self.evecs())


        self._psi_t = None




    #===========================================================================
    # propagate intial state
    #===========================================================================
    def make_td_rhs(self):
        "RHS: -i*H(t) | psi>"

        def func(t, psi):
            hamilt = np.zeros(
                (self.basis.length, self.basis.length),
                dtype=complex
            )
            for i, b_m in enumerate(self.basis.basis_list):
                for j, b_n in enumerate(self.basis.basis_list):

                    hamilt[i, j] = operators.get_hamilt_mn(
                        bc = self.bc,
                        L = self.L,
                        J_list = [self.J_fct(t)]*(self.L-1),
                        U_list = [self.U_fct(t)]*self.L,
                        N = self.N,
                        theta_list = [th_fct(t) for th_fct in self.theta_fct_list],
                        b_m = b_m,
                        b_n = b_n,
                    )

            return (-1j)*hamilt.dot(psi)


        return func


    def make_propagation(self):
        # path_psi_npz = self.path_prop/f'psi_{self.dum_Tf_dt}.npz'

        if False:
        # if (path_psi_npz.is_file() and
        #         np.load(path_psi_npz)['Tprop'] == self.Tprop and
        #         np.load(path_psi_npz)['dtprop'] == self.dtprop):
            self._psi_t = np.load(path_psi_npz)['psi_t']
        else:
            # make propagation
            integrator = integrate.complex_ode(self.make_td_rhs())
            integrator.set_integrator('dop853', nsteps=1e8,
                atol=1e-10, rtol=1e-10)
            integrator.set_initial_value(self.psi0, self.time[0])


            psi_t = [self.psi0]
            for t in self.time[1:]:
                print(f't = {t:.3f}')
                psi_tstep = integrator.integrate(t)
                psi_t.append(psi_tstep)

            self._psi_t = np.array(psi_t)

            # np.savez(
            #     path_psi_npz,
            #     time=self.time,
            #     psi_t=self._psi_t,
            #     psi0=self.psi0,
            #     psi0_str=self.psi0_str,
            #     nstate0_str=self.nstate0_str,
            #     Tprop=self.Tprop,
            #     dtprop=self.dtprop,
            #     L=self.L,
            #     N=self.N,
            #     J=self.J,
            #     U=self.U,
            #     theta=self.theta,
            #     basis_list=self.basis.basis_list
            # )
            #
            # path_logf = self.path_prop/f'psi_{self.dum_Tf_dt}_log_file.txt'
            # initialize_prop.write_log_file(
            #     path_logf, self.L, self.N, self.J, self.U, self.theta,
            #     self.psi0_str, self.nstate0_str, self.Tprop, self.dtprop)


    def psi_t(self):
        if self._psi_t is None:
            self.make_propagation()
        return self._psi_t


    #===========================================================================
    # number operator
    #===========================================================================
    def num_op_site(self, site_i):
        "psi_t = <psi(t)| b_i^t b^i |psi(t)>"
        nop = []
        for psi_tstep in self.psi_t():
            nop_t = 0
            for psi_coeff, b_m in zip(psi_tstep, self.basis.basis_list):
                nop_t += np.abs(psi_coeff)**2*b_m[site_i]
            nop.append(nop_t)
        nop = np.array(nop)
        assert np.max(np.abs(nop.imag)) < 1e-8
        nop = nop.real

        return self.time, nop


    def num_op_mat(self):
        """Return lists which are easy to plot"""
        L = self.L
        expV_list = []
        for i in range(L):
            time, expV = self.num_op_site(i)
            expV_list.append(expV)
        return time, np.array(expV_list)



    def get_E0_subpace_U0_theta0_cost_list(self, path_basis=None, version=1):
        """Estimate the cost for a given input psi

        version 1: min || Ax - psi_fin ||

        version 2: 1 - || A psi_fin ||
        
        """

        if path_basis is not None:
            path_npz = path_basis/'E0_subpace_U0_theta0_cost_list.npz'
            if path_npz.is_file():
                return np.load(path_npz)['cost_list']

        # assert self.U_fct(self.time[-1]) == 0.0 and self.theta_fct(self.time[-1]) == 0.0

        hamil_class = AnyonHubbardHamiltonian(
            bc=self.bc,
            L=self.L,
            N=self.N,
            J=self.J,
            U=0,
            theta=0
        )

        evecs_E0 = hamil_class.get_E0_eigenstate_energy_basis()

        cost_list = []
        for psi in self.psi_t():

            if version == 1:
                x = np.linalg.lstsq(evecs_E0.T, psi, rcond=None)[0]
                vec_min = np.abs((evecs_E0.T).dot(x) - psi)
                length_vec = np.abs(np.vdot(vec_min, vec_min))**2
                cost = length_vec

            elif version == 2:
                ovlp_sum = sum([np.abs(np.vdot(el, psi))**2 for el in evecs_E0])
                cost = 1 - ovlp_sum

            elif version == 3:
                cost = 1 - np.abs(np.vdot(psi, evecs_E0[2]))**2

            elif version == 4:
                cost = 1 - np.abs(np.vdot(psi, evecs_E0[1]))**2

            elif version == 5:
                cost = 1 - np.abs(np.vdot(psi, evecs_E0[0]))**2


            else:
                raise ValueError(f'version={version}')

            cost_list.append(cost)
            print(cost)

        if path_basis is not None:
            np.savez(path_npz, cost_list=cost_list)

        return cost_list


    def td_energy(self):
        "Calculate time-dependent energy"

        psi_list = self.psi_t()

        e_list = []
        for t, psi in zip(self.time, psi_list):
            hamilt = np.zeros(
                (self.basis.length, self.basis.length),
                dtype=complex
            )
            for i, b_m in enumerate(self.basis.basis_list):
                for j, b_n in enumerate(self.basis.basis_list):

                    hamilt[i, j] = operators.get_hamilt_mn(
                        bc = self.bc,
                        L = self.L,
                        J = self.J,
                        U = self.U_fct(t),
                        N = self.N,
                        theta = self.theta_fct(t),
                        b_m = b_m,
                        b_n = b_n
                    )

            energy = np.vdot(psi, hamilt.dot(psi))
            e_list.append(energy)
        return np.array(e_list)


    def td_chirality(self):
        "Calculate time-dependent energy"

        S_operator = operators.get_chiral_operator(self.basis.basis_list)

        chiral_exp = []
        for psi in self.psi_t():
            c_val = np.vdot(psi, (S_operator.dot(psi)))
            chiral_exp.append(c_val)

        return np.array(chiral_exp)