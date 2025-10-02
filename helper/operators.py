import math
import numpy as np
from numba import njit
from collections.abc import Iterable


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

    b1 = basis1.copy()
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

    if not nstate_allclose(b1, b2):
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



#===============================================================================



#-----------------------------------------------------------------------
@njit
def bi_dagger_bj(basis1, basis2, i, j, N):
    r"<1,2,0,1| b_i^t b_j | 2,2,0,0>"

    b1 = basis1.copy()
    b2 = basis2.copy()

    # b_j |2>
    if b2[j] == 0:
        return 0
    else:
        bj = math.sqrt(b2[j])
        b2[j] -= 1

    # b_i^t b_j |2>
    if b2[i] == N:
        return 0
    else:
        bi_dagger = math.sqrt(b2[i] + 1)
        b2[i] += 1

    if nstate_allclose(b1, b2):
        return bi_dagger*bj
    else:
        return 0



#-----------------------------------------------------------------------
@njit
def bj_dagger_bi_dagger_bi_bj(basis1, basis2, i, j, N):
    r"<1,2,0,1| b_j^t b_i^t b_i b_j | 2,2,0,0>"

    b1 = basis1.copy()
    b2 = basis2.copy()

    # b_j |2>
    if b2[j] == 0:
        return 0
    else:
        bj = math.sqrt(b2[j])
        b2[j] -= 1

    # b_i |2>'
    if b2[i] == 0:
        return 0
    else:
        bi = math.sqrt(b2[i])
        b2[i] -= 1

    # b_i^t |2>''
    if b2[i] == N:
        return 0
    else:
        bi_dagger = math.sqrt(b2[i] + 1)
        b2[i] += 1

    # b_j^t |2>'''
    if b2[j] == N:
        return 0
    else:
        bj_dagger = math.sqrt(b2[j] + 1)
        b2[j] += 1

    if nstate_allclose(b1, b2):
        return bj_dagger*bi_dagger*bj*bj
    else:
        return 0


#-----------------------------------------------------------------------
def get_2b_ninj(state, L, basis_list):
    r"Calculate <n_i n_j>"
    num_2b_mat = np.zeros((L, L))
    for m in range(0, L):
        for n in range(0, L):
            for psi_coeff, b_m in zip(state, basis_list):
                num_2b_mat[m, n] += (
                    np.abs(psi_coeff)**2
                    *b_m[m]
                    *b_m[n]
                )
    return num_2b_mat



#===============================================================================
# get hopping operator
#===============================================================================
@njit
def get_hop_op_i(basis_list, site_i, theta):
    r"""
        b_i^\dagger exp(i*theta*n_j) b_{i+1} 
        + b_{i+1}^\dagger exp(-i*theta*n_j) b_i 
    """
    length = len(basis_list)
    N = sum(basis_list[0])

    hop_op_j = np.zeros((length, length), dtype=np.complex128)

    for i, b_m in enumerate(basis_list):
        for j, b_n in enumerate(basis_list):

            hop_op_j[i, j] = (
                hop_term_j(site_i, N, theta, b_m, b_n)
                + hop_term_j_dagger(site_i, N, theta, b_m, b_n)
            )

    return hop_op_j

#===============================================================================
# momentum operator
#===============================================================================
def get_bibj_mat(basis_list, i, j, N):
    r"""\langle b_i^\dagger b_j \rangle
        = <psi | b_i^\dagger b_j | psi >
        = (<4,0,0,0|c_1 + <3,1,0,0|c_2 + ... ) | b_i^\dagger b_j | psi >
        """

    bibj_mat = np.zeros((len(basis_list), len(basis_list)), dtype=complex)
    for ind_1, basis1 in enumerate(basis_list):
        for ind_2, basis2 in enumerate(basis_list):
            # apply bi^dagger b_j
            coeff_bibj = bi_dagger_bj(basis1, basis2, i, j, N)

            bibj_mat[ind_1, ind_2] = coeff_bibj

    return bibj_mat


def get_bibj_mats(basis_list, L):

    N = sum(basis_list[0])

    bibj_mats = np.zeros(length = len(basis_list)
        (L, L, len(basis_list), len(basis_list)),
        dtype=complex
    )

    for m in range(0, L):
        for n in range(0, L):
            bibj_mats[m, n, :, :] = get_bibj_mat(basis_list, m, n, N)

    return bibj_mats


#===============================================================================
def get_bjbibibj_mat(basis_list, i, j, N):

    bjbibibj_mat = np.zeros((len(basis_list), len(basis_list)), dtype=complex)
    for ind_1, basis1 in enumerate(basis_list):
        for ind_2, basis2 in enumerate(basis_list):
            # apply bi^dagger b_j
            coeff_bibj = bj_dagger_bi_dagger_bi_bj(basis1, basis2, i, j, N)

            bjbibibj_mat[ind_1, ind_2] = coeff_bibj

    return bjbibibj_mat


def get_bjbibibj_mats(basis_list, L):

    N = sum(basis_list[0])

    bjbibibj_mats = np.zeros(
        (L, L, len(basis_list), len(basis_list)),
        dtype=complex
    )

    for m in range(0, L):
        for n in range(0, L):
            bjbibibj_mats[m, n, :, :] = get_bjbibibj_mat(basis_list, m, n, N)

    return bjbibibj_mats


def calc_bjbibibj(psi, L, basis_list):
    r"""Calculate the two-body density for the eigenstates"""

    bjbibibj_mats = get_bjbibibj_mats(basis_list, L)


    corr_mat = np.zeros((L, L), dtype=complex)
    for m in range(0, L):
        for n in range(0, L):
            corr_mat[m, n] = np.vdot(
                psi,
                bjbibibj_mats[m, n, :, :].dot(psi)
            )

        assert np.max(np.abs(corr_mat.imag)) < 1e-8

    return corr_mat.real



#===============================================================================
def get_chiral_operator(basis_list):

    b_length = np.shape(basis_list)[0]

    S_mat = np.zeros((b_length, b_length), dtype=complex)

    for j, b_j in enumerate(basis_list):
        S_mat[j, j] = np.exp(1j*np.pi*sum([(k+1)*el for k, el in enumerate(b_j)]))
        # print(b_j,  f'--> {round(S_mat[j, j].real, 0)}')

    return S_mat


#===============================================================================
# translation operator U
#===============================================================================
LOOKUP_TABLE = np.array([
    1, 1, 2, 6, 24, 120, 720, 5040, 40320,
    362880, 3628800, 39916800, 479001600,
    6227020800, 87178291200, 1307674368000,
    20922789888000, 355687428096000, 6402373705728000,
    121645100408832000, 2432902008176640000], dtype='int64')


@njit
def fast_factorial(n):
    if n > 20:
        raise ValueError
    return LOOKUP_TABLE[n]


@njit
def get_U_nm(basis_m, basis_n):
    r""" U_L |nvec> = sum_i^L (b_i^dagger)^(n_i+1) / sqrt(n_i!) |0>_i"""

    # L = basis_m.shape[0]
    L = len(basis_m)
  
    pref = 1
    U_n = np.zeros_like(basis_n)


    for i, n_i in enumerate(basis_n):

        if i+1 == L:
            ip1 = 0
        else:
            ip1 = i+1
            


        # apply ((b_i^dagger)^(n_(i+1)))
        for _ in range(basis_n[ip1]):
            pref = pref * np.sqrt(U_n[i]+1)
            U_n[i] += 1
        

        pref = pref * 1/np.sqrt(fast_factorial(n_i))


    if nstate_allclose(basis_m, U_n):
        return pref
    else:
        return 0


@njit
def get_translation_op(basis_list):
    r""" U_L |Psi> = sum_i^L (b_i^dagger)^(n_i+1) / sqrt(n_i!) |0>_i"""

    # L = basis_list.shape[1]
    # length = basis_list.shape[0]
    length = len(basis_list)

    # N = int(np.sum(basis_list[0]))

    # U_mat = np.zeros((length, length), dtype=np.complex128)

    # for m, basis_m in enumerate(basis_list):
    #     for n, basis_n in enumerate(basis_list):

    #         U_mat[m, n] = get_U_nm(basis_m, basis_n)
    # return U_mat
    U_mat_2 = np.zeros((length, length), dtype=np.complex128)
    for n, basis_n in enumerate(basis_list):
        U_basis_n = np.roll(basis_n, -1)


        for m, basis_m in enumerate(basis_list):
            if nstate_allclose(basis_m, U_basis_n):
                U_mat_2[m, n] = 1
            else:
                pass
    # print(U_mat)
    # print(U_mat_2)
    # print(np.allclose(U_mat_2, U_mat))

    # exit()

    return U_mat_2


@njit
def get_inv_op(basis_list):
    r""" I : inversion operator: |1, 2, 3, 4>  -->  |4, 3, 2, 1> """

    length = len(basis_list)
    inv_op = np.zeros((length, length))

    for i, basis1 in enumerate(basis_list):
        for j, basis2 in enumerate(basis_list):

            if nstate_allclose(basis1, np.flip(basis2)):
                inv_op[i, j] = 1
            else:
                continue
    
    return inv_op


# @njit
def get_U_op(basis_list, theta):
    r""" U(theta): exp(i \theta \sum_{k=1}^L n_k(n_k -1)/2) """

    length = len(basis_list)
    L = len(basis_list[0])
    U_op = np.zeros((length, length), np.complex128)

    for i, basis in enumerate(basis_list):
        exp_sum = 0
        for j in range(L):
            exp_sum += basis[j]*(basis[j]-1)
        U_op[i, i] = np.exp(1j*theta*exp_sum/2)

    return U_op