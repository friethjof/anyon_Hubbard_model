from typing import Union
import math
import numpy as np
from numba import njit


@njit
def nstate_allclose(arr1: Union[np.ndarray, list], arr2: Union[np.ndarray, list]) -> bool:
    """
    Compare two lists/arrays element-wise within a tolerance.

    This function checks if all corresponding elements in the two input arrays
    are equal within an absolute tolerance (1e-10). This is required for
    older versions of numba.

    Args:
        arr1 (Union[np.ndarray, list]): First list/array
        arr2 (Union[np.ndarray, list]): Second list/array

    Returns:
        bool: True if all elements are close within a tolerance, False otherwise
    """

    all_close = True
    for el1, el2 in zip(arr1, arr2):
        if np.abs(el1 - el2) > 1e-10:
            all_close = False
            break

    return all_close


# -------------------------------------------------------------------------------
# hopping terms for all boundary conditions
# -------------------------------------------------------------------------------
@njit
def hop_term_j(
    j: int, N: int, theta: float, basis1: np.ndarray, basis2: np.ndarray, bc: str
) -> complex:
    """
    Calculate the hopping term of sites j and j+1.

    Calcualte the jopping amplitude, note here the density dependent statistical
    phase theta which distinguishes this model from the Bose-Hubbard model.
    - open, periodic, twisted: b_j^t exp(i*theta*n_j) b_j+1
    - twisted gauge:           b_j^t exp(i*theta*n_j) b_j+1 * exp(-i*theta*N/L)

    Args:
        j (int): Index of lattice site
        N (int): Number of particles
        theta (float): Statistical phase
        basis1 (np.ndarray): Bra basis state
        basis2 (np.ndarray): Ket basis state
        bc (str): Boundary cond ('open', 'periodic', 'twisted', 'twisted_gauge')

    Returns:
        complex: contribution of the hopping term
    """

    b2 = basis2.copy()

    # b_j+1
    if b2[j + 1] == 0:
        return 0

    hop_jp1 = math.sqrt(b2[j + 1])
    b2[j + 1] -= 1

    if bc == "twisted_gauge":
        L = len(basis2)
        exp_j = np.exp(1j * theta * b2[j]) * np.exp(-1j * theta * N / L)
    else:
        exp_j = np.exp(1j * theta * b2[j])

    # b_j+1^t
    hop_j = math.sqrt(b2[j] + 1)
    b2[j] += 1

    if not nstate_allclose(basis1, b2):
        return 0

    return hop_j * exp_j * hop_jp1


@njit
def hop_term_j_dagger(
    j: int, N: int, theta: float, basis1: np.ndarray, basis2: np.ndarray, bc: str
) -> complex:
    """
    Calculate the complex conjugate hopping term of sites j and j+1.

    Calcualte the jopping amplitude, note here the density dependent statistical
    phase theta which distinguishes this model from the Bose-Hubbard model.
    - open, periodic, twisted: b_j+1^t exp(-i*theta*n_j) b_j
    - twisted gauge:           b_j+1^t exp(-i*theta*n_j) b_j * exp(i*theta*N/L)

    Args:
        j (int): Index of lattice site
        N (int): Number of particles
        theta (float): Statistical phase
        basis1 (np.ndarray): Bra basis state
        basis2 (np.ndarray): Ket basis state
        bc (str): Boundary cond ('open', 'periodic', 'twisted', 'twisted_gauge')

    Returns:
        complex: contribution of the hopping term
    """

    b2 = basis2.copy()

    # b_j
    if b2[j] == 0:
        return 0

    hop_j = math.sqrt(b2[j])
    b2[j] -= 1

    # exp(-i * theta * n_j)
    if bc == "twisted_gauge":
        L = len(basis2)
        exp_j = np.exp(-1j * theta * b2[j]) * np.exp(1j * theta * N / L)
    else:
        exp_j = np.exp(-1j * theta * b2[j])

    # b_j+1^t
    hop_jp1 = math.sqrt(b2[j + 1] + 1)
    b2[j + 1] += 1

    if not nstate_allclose(basis1, b2):
        return 0

    return hop_jp1 * hop_j * exp_j


# -------------------------------------------------------------------------------
# onesite interaction
# -------------------------------------------------------------------------------
@njit
def onsite_int(j: int, basis1: np.ndarray, basis2: np.ndarray) -> float:
    """
    Calculate the onsite interaction term of site j.

    Calculate the onsite interaction strength by counting the number of
    particles on site j: U/2 * n_j (n_j - 1)

    Args:
        j (int): Index of the lattice site
        basis1 (np.ndarray): Bra basis state
        basis2 (np.ndarray): Ket basis state

    Returns:
        float: The onsite interaction contribution for site j.
    """

    if not nstate_allclose(basis1, basis2):
        return 0

    if basis1[j] == 0:
        return 0

    return basis1[j] * (basis1[j] - 1)


# -------------------------------------------------------------------------------
# for twisted boundary conditions
# -------------------------------------------------------------------------------
@njit
def hop_term_L(
    N: int, theta: float, basis1: np.ndarray, basis2: np.ndarray, bc: str
) -> complex:
    """
    Calculate the hopping term at site L for a lattice ring potential.

    This function is only for periodic or twisted boundary conditions as it
    connects the L-th site with the first site:
        Periodic bc: b_L^t exp(i*theta*n_L) b_1
        Twisted  bc: b_L^t exp(i*theta*n_L) b_1 * exp(-i*theta*N)
        Twisted gauge bc: b_L^t exp(-i * theta * n_L) b_1 * exp(-i*theta*N/L)

    Args:
        N (int): Number of particles
        theta (float): Statistical phase
        basis1 (Union[np.ndarray, list]): Bra basis state
        basis2 (Union[np.ndarray, list]): Ket basis state
        bc (str): Boundary condition ('periodic', 'twisted', 'twisted_gauge')

    Returns:
        complex number, contribution of the hopping term
    """

    b2 = basis2.copy()

    # b_j+1
    if b2[0] == 0:
        return 0

    hop_1 = math.sqrt(b2[0])
    b2[0] -= 1

    # exp(i * theta * n_j)
    if bc == "periodic":
        exp_L = np.exp(1j * theta * b2[-1])
    elif bc == "twisted":
        exp_L = np.exp(1j * theta * b2[-1]) * np.exp(-1j * theta * N)
    elif bc == "twisted_gauge":
        L = len(basis2)
        exp_L = np.exp(1j * theta * b2[-1]) * np.exp(-1j * theta * N / L)
    else:
        raise NotImplementedError

    # b_j^t
    if b2[-1] == N:
        return 0

    hop_L = math.sqrt(b2[-1] + 1)
    b2[-1] += 1

    if not nstate_allclose(basis1, b2):
        return 0

    return hop_L * exp_L * hop_1


@njit
def hop_term_L_dagger(
    N: int, theta: float, basis1: np.ndarray, basis2: np.ndarray, bc: str
) -> complex:
    """
    Calculate the complex conjugated hopping term at site L.

    This function is only for periodic or twisted boundary conditions as it
    connects the L-th site with the first site (ring potential):
        Periodic bc: b_1^† exp(-i * theta * n_L) b_L
        Twisted  bc: b_1^† exp(-i * theta * n_L) b_L * exp(i * theta * N)
        Twisted gauge bc: b_1^† exp(-i * theta * n_L) b_L * exp(i * theta * N/L)

    Args:
        N (int): Number of particles
        theta (float): Statistical phase
        basis1 (Union[np.ndarray, list]): Bra basis state
        basis2 (Union[np.ndarray, list]): Ket basis state
        bc (str): Boundary condition ('periodic', 'twisted', 'twisted_gauge')

    Returns:
        complex number, contribution of the hopping term
    """

    b2 = basis2.copy()

    # b_j
    if b2[-1] == 0:
        return 0

    hop_L = math.sqrt(b2[-1])
    b2[-1] -= 1

    # exp(-i * theta * n_j)
    if bc == "periodic":
        exp_L = np.exp(-1j * theta * b2[-1])
    elif bc == "twisted":
        exp_L = np.exp(-1j * theta * b2[-1]) * np.exp(1j * theta * N)
    elif bc == "twisted_gauge":
        L = len(basis2)
        exp_L = np.exp(-1j * theta * b2[-1]) * np.exp(1j * theta * N / L)
    else:
        raise NotImplementedError

    # b_j+1^t
    if b2[0] == N:
        return 0

    hop_1 = math.sqrt(b2[0] + 1)
    b2[0] += 1

    if not nstate_allclose(basis1, b2):
        return 0

    return hop_1 * exp_L * hop_L


# ===============================================================================
# get element of Hamiltonian
# ===============================================================================
@njit
def get_hamilt_mn(
    bc: str,
    L: int,
    J_list: Union[np.ndarray, list],
    U_list: Union[np.ndarray, list],
    N: int,
    theta_list: Union[np.ndarray, list],
    b_m: np.ndarray,
    b_n: np.ndarray,
) -> float:
    """
    Calculate one entry of the Hamiltonian matrix.

    Sum the contributions of the hopping and interaction terms:
    H_mn = <b_m | H | b_n>
         = <b_m | J_hop | b_n> + <b_m | U_int | b_n>
    Note that the hopping term and interaction term denote a sum over the
    lattice sites.
    For instance see: https://arxiv.org/abs/2306.01737 Eq. (1) for open
    boundary conditions (bc). Allow here also periodic or twisted bc.

    Args:
        bc (str): Boundary condition ('open', 'periodic', 'twisted(_gauge)')
        L (int): Number of lattice sites
        J_list (Union[np.ndarray, list]): Hopping amplitudes
        U_list (Union[np.ndarray, list]): Interaction strengths for each site.
        N (int): Number of particles.
        theta_list (Union[np.ndarray, list]): Anyonic phase for each site.
        basis_list (np.ndarray): Array of basis states.

    Returns:
        np.ndarray: Complex Hamiltonian matrix of shape (basis_len, basis_len).
    """

    H_mn = 0

    for ind in range(L):

        # -----------------------------------------------------------------------
        if bc == "open":

            # add hopping terms
            if ind < L - 1:
                # - sum_j^L-1 J_j b_j^t exp(i*theta*n_j) b_j+1
                H_mn -= J_list[ind] * hop_term_j(ind, N, theta_list[ind], b_m, b_n, bc)
                # - sum_j^L-1 J_j b_j+1^t exp(-i*theta*n_j) b_j
                H_mn -= J_list[ind] * hop_term_j_dagger(
                    ind, N, theta_list[ind], b_m, b_n, bc
                )

            # add interaction term
            # 1/2 sum_j^L U_j n_j(n_j-1)
            H_mn += U_list[ind] / 2.0 * onsite_int(ind, b_m, b_n)

        elif bc == "twisted" or bc == "periodic" or bc == "twisted_gauge":

            # add hopping terms
            if ind < L - 1:
                # - sum_j^L-1 J_j b_j^t exp(i*theta*n_j) b_j+1
                H_mn -= J_list[ind] * hop_term_j(ind, N, theta_list[ind], b_m, b_n, bc)
                # - sum_j^L-1 J_j b_j+1^t exp(-i*theta*n_j) b_j
                H_mn -= J_list[ind] * hop_term_j_dagger(
                    ind, N, theta_list[ind], b_m, b_n, bc
                )

            if ind == L - 1:
                H_mn -= J_list[ind] * hop_term_L(N, theta_list[ind], b_m, b_n, bc)
                H_mn -= J_list[ind] * hop_term_L_dagger(N, theta_list[ind], b_m, b_n, bc)

            # add interaction term
            # 1/2 sum_j^L U_j n_j(n_j-1)
            H_mn += U_list[ind] / 2.0 * onsite_int(ind, b_m, b_n)

        else:

            raise NotImplementedError

    return H_mn


@njit
def get_hamilt_mat(
    bc: str,
    J_list: Union[np.ndarray, list],
    U_list: Union[np.ndarray, list],
    theta_list: Union[np.ndarray, list],
    basis_list: np.ndarray,
) -> np.ndarray:
    """
    Construct the Hamiltonian matrix in number state basis.

    Loops through all number states and calculate for each pair the element of
    the Hamiltonian matrix.

    Args:
        bc (str): Boundary condition ('open', 'periodic', 'twisted(_gauge)')
        J_list (Union[np.ndarray, list]): Hopping amplitudes
        U_list (Union[np.ndarray, list]): Interaction strengths for each site.
        theta_list (Union[np.ndarray, list]): Anyonic phase for each site.
        basis_list (np.ndarray): Array of basis states.

    Returns:
        np.ndarray: Complex Hamiltonian matrix of shape (basis_len, basis_len).
    """
    basis_len = len(basis_list)  # size of basis
    N = np.sum(basis_list[0])  # number of particles
    L = len(basis_list[0])  # number of lattice sites

    hamilt_mat = np.zeros((basis_len, basis_len), dtype=np.complex128)

    for i, b_m in enumerate(basis_list):
        for j, b_n in enumerate(basis_list):
            hamilt_mat[i, j] = get_hamilt_mn(
                bc=bc,
                L=L,
                J_list=J_list,
                U_list=U_list,
                N=N,
                theta_list=theta_list,
                b_m=b_m,
                b_n=b_n,
            )

    return hamilt_mat
