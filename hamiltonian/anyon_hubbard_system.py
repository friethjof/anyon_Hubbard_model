import time
from pathlib import Path
from typing import Optional, Union, Any, Iterable

import numpy as np

from hamiltonian.basis import Basis
from hamiltonian.plot_helper import PlotObservables
from helper import construct_hamilt, path_dirs, other_tools


class AnyonHubbardHamiltonian:
    """
    Defines and solves the 1D anyon-Hubbard Hamiltonian.

    The Hamiltonian for a system with open boundary conditions reads as:
        H = - sum_j^{L-1} J_j b_j^† exp(i*theta*n_j) b_{j+1}
            + b_{j+1}^† exp(-i*theta*n_j) b_j
            + 1/2 sum_j^{L} U_j n_j(n_j - 1)
    where the bosonic operators act as:
        b_j |n> = sqrt(n) |n-1>
        b_j^† |n> = sqrt(n+1) |n+1>
    Here, we construct the Hamiltonian in a number state basis and provide
    functions to calculate the most basic observables, such as the energy.

    Keyword Args:
        bc (str): Boundary condition type, default 'open'.
        L (int): Number of sites.
        N (int): Number of particles.
        J (float or Iterable): Hopping terms, can be site-dependent.
        U (float or Iterable): Interaction terms, can be site-dependent.
        theta (float or Iterable): Anyonic phase, can be site-dependent.
        bool_save (bool): Flag for saving computations.

    Notes:
        J, U and theta can be site-dependent.
    """

    def __init__(self, **args_in: Any) -> None:
        """
        Initialize the AnyonHubbardHamiltonian with provided parameters.

        Args:
            **args_in: Arbitrary keyword arguments for parameters described
                in the class docstring keyword args section.
        """
        self.bc: str = args_in.get('bc', 'open')
        if 'L' not in args_in:
            raise ValueError("Argument 'L' (lattice sites) is required.")
        self.L: int = args_in['L']
        if 'N' not in args_in:
            raise ValueError("Argument 'N' (number of particles) is required.")
        self.N: int = args_in['N']
        if 'J' not in args_in:
            raise ValueError("Argument 'J' (hopping amplitude(s)) required.")
        self.J: Union[float, Iterable[float]] = args_in['J']
        if 'U' not in args_in:
            raise ValueError("Argument 'U' (interaction term(s)) required.")
        self.U: Union[float, Iterable[float]] = args_in['U']
        if 'theta' not in args_in:
            raise ValueError("Argument 'theta' (stat. phase(s)) required.")
        self.theta: Union[float, Iterable[float]] = args_in['theta']

        self.basis: Basis = Basis(self.L, self.N)

        self.bool_save: bool = args_in.get('bool_save', False)

        self._hamilt: Optional[np.ndarray] = None
        self._evals: Optional[np.ndarray] = None
        self._evecs: Optional[np.ndarray] = None

        if self.bool_save:
            self.path_basis: Path = path_dirs.get_path_basis(
                bc=self.bc,
                L=self.L,
                N=self.N,
                J=self.J,
                U=self.U,
                theta=self.theta
            )
        
        # plotting module
        self.plot = PlotObservables(self)


    def make_diagonalization(self) -> None:
        """
        Builds and diagonalizes the Hamiltonian.

        Loads cached Hamiltonian if available, otherwise computes it from
        parameters and saves the result if bool_save is enabled.

        Raises:
            AssertionError: If...
            ... the Hamiltonian is not Hermitian.
            ... eigenvalues have significant imaginary parts.
            ... eigenvectors are not orthonormal.
        """

        if self.bool_save:
            path_hamil_npz = self.path_basis / 'hamilt_spectrum.npz'

        if self.bool_save and path_hamil_npz.is_file():
            self._hamilt = np.load(path_hamil_npz)['hamilt_mat']
            self._evals = np.load(path_hamil_npz)['evals']
            self._evecs = np.load(path_hamil_npz)['evecs']
            return

        if self.bool_save:
            print('basis size:', self.basis.length)
            clock_1 = time.time()

        #----------------------------------------------------------------------
        # Prepare θ, J, U as lists if they're not Iterable
        if isinstance(self.theta, Iterable):
            theta_list = self.theta
        elif self.bc == 'open':
            # theta is not iterable and open bc 
            theta_list = [self.theta] * (self.L - 1)
        else:
            # theta is not iterable and periodic/gauge bc 
            theta_list = [self.theta] * self.L

        if isinstance(self.J, Iterable):
            J_list = self.J
        elif self.bc == 'open':
            J_list = [self.J] * (self.L - 1)
        else:
            J_list = [self.J] * self.L

        if isinstance(self.U, Iterable):
            U_list = self.U
        else:
            U_list = [self.U] * self.L

        #----------------------------------------------------------------------
        # create hamiltonian matrix
        self._hamilt = construct_hamilt.get_hamilt_mat(
            bc=self.bc,
            J_list=np.array(J_list),
            U_list=np.array(U_list),
            theta_list=np.array(theta_list),
            basis_list=self.basis.basis_list,
        )

        # Check Hermitian
        assert np.allclose(self._hamilt, np.conjugate(self._hamilt.T))

        if self.bool_save:
            clock_2 = time.time()
            print(
                'matrix hamiltonian created, time:',
                other_tools.time_str(clock_2 - clock_1)
            )

        # Diagonalize using eigh for Hermitian matrix
        evals, evecs = np.linalg.eigh(self._hamilt)
        idx = evals.argsort()
        self._evals = evals[idx]
        self._evecs = (evecs[:, idx]).T

        # Check eigenvalues are real
        assert np.max(np.abs(self._evals.imag)) < 1e-8
        self._evals = self._evals.real

        if self.bool_save:
            clock_3 = time.time()
            print(
                'matrix hamiltonian has been diagonalized, time:',
                other_tools.time_str(clock_3 - clock_2)
            )

        # Check orthonormality
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
            print(
                'orthonormality has been checked, time:',
                other_tools.time_str(clock_4 - clock_3)
            )

        # Save computed data
        if self.bool_save:
            np.savez(
                path_hamil_npz,
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
            file_size_bytes = path_hamil_npz.stat().st_size
            file_size_MB = file_size_bytes / (1024 ** 2)
            print(f'File size (bytes):', file_size_bytes)
            print(f'File size (MB): {file_size_MB:.3f}')

        if self.bool_save:
            clock_5 = time.time()
            print('Total time consumption:', other_tools.time_str(clock_5 - clock_1))


    def hamilt(self) -> np.ndarray:
        """
        Returns the Hamiltonian matrix from the memory or creates it.

        Returns:
            np.ndarray: Hamiltonian matrix.
        """
        if self._hamilt is None:
            self.make_diagonalization()

        return self._hamilt


    def evals(self) -> np.ndarray:
        """
        Loads eigenvalues from the memory or cimputes them.

        Returns:
            np.ndarray: Eigenvalues.
        """
        if self._evals is None:
            self.make_diagonalization()

        return self._evals


    def evecs(self) -> np.ndarray:
        """
        Loads eigenvectors or computes them if necessary.

        Returns:
            np.ndarray: Eigenvectors.
        """
        if self._evecs is None:
            self.make_diagonalization()

        return self._evecs


    def gs_psi(self) -> np.array:
        """
        Returns:
            Ground state psi (np.array)
        """
        return self.evecs()[0]


    #===========================================================================
    def one_site_density(self):
        """
        Calculate and return the one-site density as 1D-array
        rho_i = <psi| b_i^t b^i |psi>
        where b_i^t is the creation operator of site i

        Returns:
            (real) one-site density (np.array)
        """

        psi0 = self.gs_psi()
        obd = []
        for i in range(self.L):
            obd.append(
                np.vdot(psi0, self.basis.op_bi_bi(i).dot(psi0))
            )
        obd_arr = np.array(obd)

        if np.max(np.abs(obd_arr.imag)) > 1e-10:
            raise ValueError('Imaginary part is larger than threshold', obd_arr)

        return obd_arr.real
