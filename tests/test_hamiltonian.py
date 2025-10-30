import unittest

import numpy as np
from hamiltonian.anyon_hubbard_system import AnyonHubbardHamiltonian


class TestHamilt(unittest.TestCase):
    """
    Test the construction of the Hamiltonian by comparing to analytic results.


    Methods:
    - _get_analytic_hamilt_obc
    - _get_analytic_hamilt_tbc
    - test_hamiltion_construction_obc: compares the numerically obtained
                                       Hamiltonian to an analytic solution
                                       for open boundaries.
    """

    # ---------------------------------------------------------------------------
    # open boundary conditions
    # ---------------------------------------------------------------------------
    def _get_analytic_hamilt_obc(self, L: int, N: int, theta: float) -> np.ndarray:
        """Return the analytic results of a Hamiltonian in number state basis.

        Assumes J = 1 and U = 0 and OPEN boundary conditions (obc).

        Args:
            L (int): number of lattice sites.
            N (int): number of particles.
            theta (float): statistical parameter.

        Returns:
            Hamiltonian (np.ndarray) obeying open boundary conditions.

        """

        if L == 3 and N == 2:

            hamilt = np.array(
                [
                    [0, -1 * np.sqrt(2) * np.exp(1j * theta), 0, 0, 0, 0],
                    [
                        -1 * np.sqrt(2) * np.exp(-1j * theta),
                        0,
                        -1,
                        -1 * np.sqrt(2),
                        0,
                        0,
                    ],
                    [0, -1, 0, 0, -1, 0],
                    [0, -1 * np.sqrt(2), 0, 0, -1 * np.sqrt(2) * np.exp(1j * theta), 0],
                    [
                        0,
                        0,
                        -1,
                        -1 * np.sqrt(2) * np.exp(-1j * theta),
                        0,
                        -1 * np.sqrt(2),
                    ],
                    [0, 0, 0, 0, -1 * np.sqrt(2), 0],
                ]
            )

        elif L == 2 and N == 4:

            hamilt = np.array(
                [
                    [0, -2 * np.exp(3j * theta), 0, 0, 0],
                    [
                        -2 * np.exp(-3j * theta),
                        0,
                        -np.sqrt(6) * np.exp(2j * theta),
                        0,
                        0,
                    ],
                    [
                        0,
                        -np.sqrt(6) * np.exp(-2j * theta),
                        0,
                        -np.sqrt(6) * np.exp(1j * theta),
                        0,
                    ],
                    [0, 0, -np.sqrt(6) * np.exp(-1j * theta), 0, -2],
                    [0, 0, 0, -2, 0],
                ]
            )

        elif L == 2 and N == 6:
            hamilt = np.array(
                [
                    [0, -np.sqrt(6) * np.exp(5j * theta), 0, 0, 0, 0, 0],
                    [
                        -np.sqrt(6) * np.exp(-5j * theta),
                        0,
                        -np.sqrt(10) * np.exp(4j * theta),
                        0,
                        0,
                        0,
                        0,
                    ],
                    [
                        0,
                        -np.sqrt(10) * np.exp(-4j * theta),
                        0,
                        -np.sqrt(12) * np.exp(3j * theta),
                        0,
                        0,
                        0,
                    ],
                    [
                        0,
                        0,
                        -np.sqrt(12) * np.exp(-3j * theta),
                        0,
                        -np.sqrt(12) * np.exp(2j * theta),
                        0,
                        0,
                    ],
                    [
                        0,
                        0,
                        0,
                        -np.sqrt(12) * np.exp(-2j * theta),
                        0,
                        -np.sqrt(10) * np.exp(1j * theta),
                        0,
                    ],
                    [0, 0, 0, 0, -np.sqrt(10) * np.exp(-1j * theta), 0, -np.sqrt(6)],
                    [0, 0, 0, 0, 0, -np.sqrt(6), 0],
                ]
            )

        else:
            raise NotImplementedError(f"System size L={L}, N={N} not implemented.")

        return hamilt

    def test_hamiltion_construction_obc(self):
        """Check the construction of the Hamiltonian for open boundaries (obc).

        The correctness of the Hamiltonian is tested by comparing the numeric
        results to analytically calculated ones. A Hamiltonian is considered
        consisting of L lattice sites with open boundaries and N particles.
        Compare the results for various values of theta.

        Raises:
            AssertionError: When numeric and analytic Hamiltonians do not match.
        """

        L_N_vals = [(3, 2), (2, 4), (2, 6)]

        # 'random' theta values
        theta_list = [0, 0.1, np.sqrt(2), 0.1 * np.pi, 0.5 * np.pi, np.pi, 3 * np.pi]

        for L, N in L_N_vals:
            for theta in theta_list:

                # get analytically calculated Hamiltonian
                hamilt_ana = self._get_analytic_hamilt_obc(L, N, theta)

                # numerically obtain hamiltonian
                anyon_class = AnyonHubbardHamiltonian(
                    bc="open",
                    L=L,
                    N=N,
                    J=1,
                    U=0,
                    theta=theta,
                    bool_save=False,  # calculate Hamiltonian from scratch
                )

                hamilt_num = anyon_class.hamilt()

                info_msg = "\n\nconstruction of Hamiltonian failed for: "
                info_msg += f"N={N}, L={L}, theta={theta} and open boundaries!"

                assert np.allclose(hamilt_ana, hamilt_num), info_msg

    # ---------------------------------------------------------------------------
    # twisted boundary conditions
    # ---------------------------------------------------------------------------
    def _get_analytic_hamilt_tbc(self, L: int, N: int, theta: float) -> np.ndarray:
        """Return the analytic results of a Hamiltonian in number state basis.

        Assumes J = 1 and U = 0 and TWISTED boundary conditions (tbc).

        Args:
            L (int): number of lattice sites.
            N (int): number of particles.
            theta (float): statistical parameter.

        Returns:
            Hamiltonian (np.ndarray) obeying twisted boundary conditions.
        """

        if L == 3 and N == 1:

            hamilt = np.array(
                [
                    [0, -1, -np.exp(1j * theta)],
                    [-1, 0, -1],
                    [-np.exp(-1j * theta), -1, 0],
                ]
            )

        elif L == 3 and N == 2:

            hamilt = np.array(
                [
                    [
                        0,
                        -np.sqrt(2) * np.exp(1j * theta),
                        -np.sqrt(2) * np.exp(2j * theta),
                        0,
                        0,
                        0,
                    ],
                    [
                        -np.sqrt(2) * np.exp(-1j * theta),
                        0,
                        -1,
                        -np.sqrt(2),
                        -np.exp(2j * theta),
                        0,
                    ],
                    [
                        -np.sqrt(2) * np.exp(-2j * theta),
                        -1,
                        0,
                        0,
                        -1,
                        -np.sqrt(2) * np.exp(1j * theta),
                    ],
                    [0, -np.sqrt(2), 0, 0, -np.sqrt(2) * np.exp(1j * theta), 0],
                    [
                        0,
                        -np.exp(-2j * theta),
                        -1,
                        -np.sqrt(2) * np.exp(-1j * theta),
                        0,
                        -np.sqrt(2),
                    ],
                    [0, 0, -np.sqrt(2) * np.exp(-1j * theta), 0, -np.sqrt(2), 0],
                ]
            )

        elif L == 2 and N == 2:

            hamilt = np.array(
                [
                    [0, -np.sqrt(2) * (np.exp(1j * theta) + np.exp(2j * theta)), 0],
                    [
                        -np.sqrt(2) * (np.exp(-1j * theta) + np.exp(-2j * theta)),
                        0,
                        -np.sqrt(2) * (1 + np.exp(1j * theta)),
                    ],
                    [0, -np.sqrt(2) * (1 + np.exp(-1j * theta)), 0],
                ]
            )

        else:
            raise NotImplementedError(f"System size L={L}, N={N} not implemented.")

        return hamilt

    def test_hamiltion_construction_tbc(self):
        """Check the Hamiltonian with twisted boundaries (tbc).

        The correctness of the Hamiltonian is tested by comparing the numeric
        results to analytically calculated ones. A Hamiltonian is considered
        consisting of L lattice sites with twisted boundaries and N particles.
        Compare the results for various values of theta.

        Raises:
            AssertionError: When numeric and analytic Hamiltonians do not match.
        """

        L_N_vals = [(3, 1), (3, 2), (2, 2)]

        # 'random' theta values
        theta_list = [0, 0.1, np.sqrt(2), 0.1 * np.pi, 0.5 * np.pi, np.pi, 3 * np.pi]

        for L, N in L_N_vals:
            for theta in theta_list:

                # get analytically calculated Hamiltonian
                hamilt_ana = self._get_analytic_hamilt_tbc(L, N, theta)

                # numerically obtain hamiltonian
                anyon_class = AnyonHubbardHamiltonian(
                    bc="twisted",
                    L=L,
                    N=N,
                    J=1,
                    U=0,
                    theta=theta,
                    bool_save=False,  # calculate Hamiltonian from scratch
                )

                hamilt_num = anyon_class.hamilt()

                info_msg = "\n\nconstruction of Hamiltonian failed for: "
                info_msg += f"N={N}, L={L}, theta={theta} and twisted BC!"

                assert np.allclose(hamilt_ana, hamilt_num), info_msg
