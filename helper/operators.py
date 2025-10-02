import math
import numpy as np
from numba import njit


@njit
def nstate_allclose(arr1, arr2):
    r'Provide an alternative to compare two number states'

    all_close = True
    for el1, el2 in zip(arr1, arr2):
        if np.abs(el1 - el2) > 1e-10:
            all_close = False
            break

    return all_close


@njit
def hop_term_j(j, N, theta, basis1, basis2, bc):
    r"""
    open, periodic, twisted: b_j^t exp(i*theta*n_j) b_j+1
    twisted gauge:           b_j^t exp(i*theta*n_j) b_j+1 * exp(-i*theta*N/L)
    """

    b2 = basis2.copy()

    # b_j+1
    if b2[j+1] == 0:
        return 0

    hop_jp1 = math.sqrt(b2[j+1])
    b2[j+1] -= 1


    if bc == 'twisted_gauge':
        L = len(basis2)
        exp_j = np.exp(1j*theta*b2[j]) * np.exp(-1j*theta*N/L)
    else:
        exp_j = np.exp(1j*theta*b2[j])

    # b_j^t
    if b2[j] == N:
        return 0

    hop_j = math.sqrt(b2[j]+1)
    b2[j] += 1

    if not nstate_allclose(basis1, b2):
        return 0

    return hop_j*exp_j*hop_jp1



@njit
def hop_term_j_dagger(j, N, theta, basis1, basis2, bc):
    r"""    
    open, periodic, twisted: b_j+1^t exp(-i*theta*n_j) b_j
    twisted gauge:           b_j+1^t exp(-i*theta*n_j) b_j * exp(i*theta*N/L)
    """


    b1 = basis1.copy()
    b2 = basis2.copy()

    # b_j
    if b2[j] == 0:
        return 0

    hop_j = math.sqrt(b2[j])
    b2[j] -= 1


    # exp(-i * theta * n_j)
    if bc == 'twisted_gauge':
        L = len(basis2)
        exp_j = np.exp(-1j*theta*b2[j]) * np.exp(1j*theta*N/L)
    else:
        exp_j = np.exp(-1j*theta*b2[j]) 


    # b_j+1^t
    if b2[j+1] == N:
        return 0

    hop_jp1 = math.sqrt(b2[j+1]+1)
    b2[j+1] += 1

    if not nstate_allclose(b1, b2):
        return 0

    return hop_jp1*hop_j*exp_j


@njit
def onsite_int(j, basis1, basis2):
    r"U/2 sum_j^L  n_j(n_j-1)"

    if not nstate_allclose(basis1, basis2):
        return 0


    if basis1[j] == 0:
        return 0

    return basis1[j]*(basis1[j] - 1)



#-------------------------------------------------------------------------------
# for twisted boundary conditions
#-------------------------------------------------------------------------------
@njit
def hop_term_L(N, theta, basis1, basis2, bc):
    r"""
    periodic bc: b_L^t exp(i*theta*n_L) b_1
    twisted  bc: b_L^t exp(i*theta*n_L) b_1 * exp(-i*theta*N)
    """

    b1 = basis1.copy()
    b2 = basis2.copy()

    # b_j+1
    if b2[0] == 0:
        return 0

    hop_1 = math.sqrt(b2[0])
    b2[0] -= 1

    # exp(i * theta * n_j)
    if bc == 'periodic':
        exp_L = np.exp(1j*theta*b2[-1])
    elif bc == 'twisted':
        exp_L = np.exp(1j*theta*b2[-1])*np.exp(-1j*theta*N)
    elif bc == 'twisted_gauge':
        L = len(basis2)
        exp_L = np.exp(1j*theta*b2[-1]) * np.exp(-1j*theta*N/L)
    else:
        raise NotImplementedError


    # b_j^t
    if b2[-1] == N:
        return 0

    hop_L = math.sqrt(b2[-1]+1)
    b2[-1] += 1

    if not nstate_allclose(b1, b2):
        return 0

    return hop_L*exp_L*hop_1

@njit
def hop_term_L_dagger(N, theta, basis1, basis2, bc):
    r"""
    periodic bc: b_1^t exp(-i*theta*n_L) b_L
    twisted  bc: b_1^t exp(-i*theta*n_L) b_L * exp(i*theta*N)
    """

    b1 = basis1.copy()
    b2 = basis2.copy()

    # b_j
    if b2[-1] == 0:
        return 0

    hop_L = math.sqrt(b2[-1])
    b2[-1] -= 1

    # exp(-i * theta * n_j)
    if bc == 'periodic':
        exp_L = np.exp(-1j*theta*b2[-1])
    elif bc == 'twisted':
        exp_L = np.exp(-1j*theta*b2[-1])*np.exp(1j*theta*N)
    elif bc == 'twisted_gauge':
        L = len(basis2)
        exp_L = np.exp(-1j*theta*b2[-1]) * np.exp(1j*theta*N/L)
    else:
        raise NotImplementedError

    # b_j+1^t
    if b2[0] == N:
        return 0

    hop_1 = math.sqrt(b2[0]+1)
    b2[0] += 1

    if not nstate_allclose(b1, b2):
        return 0

    return hop_1*exp_L*hop_L



#===============================================================================
# get element of Hamiltonian
#===============================================================================
@njit
def get_hamilt_mn(bc, L, J_list, U_list, N, theta_list, b_m, b_n):

    H_mn = 0

    for ind in range(L):

        #-----------------------------------------------------------------------
        if bc == 'open':

            # add hopping terms
            if ind < L-1:
                # - sum_j^L-1 J_j b_j^t exp(i*theta*n_j) b_j+1
                H_mn -= J_list[ind] * hop_term_j(ind, N, theta_list[ind], b_m, b_n, bc)
                 # - sum_j^L-1 J_j b_j+1^t exp(-i*theta*n_j) b_j
                H_mn -= J_list[ind] * hop_term_j_dagger(ind, N, theta_list[ind], b_m, b_n, bc)

            # add interaction term
            # 1/2 sum_j^L U_j n_j(n_j-1)
            H_mn += U_list[ind] / 2.0 * onsite_int(ind, b_m, b_n)



        elif bc == 'twisted' or bc == 'periodic' or bc == 'twisted_gauge':

            # add hopping terms
            if ind < L-1:
                # - sum_j^L-1 J_j b_j^t exp(i*theta*n_j) b_j+1
                H_mn -= J_list[ind] * hop_term_j(ind, N, theta_list[ind], b_m, b_n, bc)
                 # - sum_j^L-1 J_j b_j+1^t exp(-i*theta*n_j) b_j
                H_mn -= J_list[ind] * hop_term_j_dagger(ind, N, theta_list[ind], b_m, b_n, bc)


            if ind == L-1:
                H_mn -= J_list[ind] * hop_term_L(N, theta_list[ind], b_m, b_n, bc)
                H_mn -= J_list[ind] * hop_term_L_dagger(N, theta_list[ind], b_m, b_n, bc)


            # add interaction term
            # 1/2 sum_j^L U_j n_j(n_j-1)
            H_mn += U_list[ind] / 2.0 * onsite_int(ind, b_m, b_n)


        else:
            
            raise NotImplementedError

        #-----------------------------------------------------------------------

    return H_mn


@njit
def get_hamilt_mat(bc, L, J_list, U_list, N, theta_list, basis_list):

    basis_len = len(basis_list)

    hamilt_mat = np.zeros((basis_len, basis_len), dtype=np.complex128)

    for i, b_m in enumerate(basis_list):
        for j, b_n in enumerate(basis_list):

            hamilt_mat[i, j] = get_hamilt_mn(
                bc = bc,
                L = L,
                J_list = J_list,
                U_list = U_list,
                N = N,
                theta_list = theta_list,
                b_m = b_m,
                b_n = b_n,
            )

    return hamilt_mat

