import math
import gc
from pathlib import Path

from numba import njit
import numpy as np
import matplotlib.pyplot as plt

from helper import operators
from hamiltonian.solve_hamilt import AnyonHubbardHamiltonian



@njit
def nstate_allclose(arr1, arr2):
    'Provide an alternative to compare two number states'

    all_close = True
    for el1, el2 in zip(arr1, arr2):
        if np.abs(el1 - el2) > 1e-10:
            all_close = False
            break

    return all_close


@njit
def get_b_i_mat(basis_left, basis_right, ind_i):
    """ <m | b_i | n>
        with |n> corresponds to N   particles
        with |m> corresponds to N-1 particles
    """


    b_i_mat = np.zeros((basis_left.shape[0], basis_right.shape[0]), dtype=np.complex128)

    for j, n_vec in enumerate(basis_right):
        nvec = n_vec.copy()

        if nvec[ind_i] == 0:
            continue
        else:
            bi_n = math.sqrt(nvec[ind_i])
            nvec[ind_i] -= 1

        for i, m_vec in enumerate(basis_left):
            if not nstate_allclose(m_vec, nvec):
                continue

            b_i_mat[i, j] = bi_n

    return b_i_mat


@njit
def get_b_i_dagger_mat(basis_left, basis_right, ind_i):
    """ <m | b_i^t | n>
        with |n> corresponds to N   particles
        with |m> corresponds to N+1 particles
    """


    b_i_mat = np.zeros((basis_left.shape[0], basis_right.shape[0]), dtype=np.complex128)

    for j, n_vec in enumerate(basis_right):
        nvec = n_vec.copy()

        bi_n = math.sqrt(nvec[ind_i]+1)
        nvec[ind_i] += 1



        for i, m_vec in enumerate(basis_left):
            if not nstate_allclose(m_vec, nvec):
                continue

            b_i_mat[i, j] = bi_n

    return b_i_mat



@njit
def apply_H(t, evals, evecs, psi0):
    "sum_i exp(-i * E_n * t) |n><n| psi>"
    psi_t = np.zeros((psi0.shape[0]), dtype=np.complex128)
    for eval_n, evec_n in zip(evals, evecs):
        psi_t += np.exp(-1j*eval_n*t)*evec_n*np.vdot(evec_n, psi0)
        # n_n = np.outer(evec_n, np.conjugate(evec_n))
        # psi_t += np.exp(-1j*eval_n*t)* n_n.dot(psi0)
    return psi_t



#===========================================================================
# out-of-time-ordered correlator (OTOC)
#===========================================================================
def get_otoc_ij_beta(hamil_class, site_i, site_j, beta, time):
    """https://link.aps.org/doi/10.1103/PhysRevLett.121.250404 conclusion

    F_jk(t) = <b_j^t(t) b_k^t(0) b_j(t) b_k(0)>_beta

    with
    b_j(t) = |n><n| exp(i E_n t) b_j exp(-i E_m t) |m><m|
            = exp(i E_n t) * exp(-i E_m t)  *  |n><n| b_j |m><m|
    
    <.>_beta = Tr(O exp(-beta H )) / Tr(exp(-beta H))
                = sum_n <n| O exp(-beta E_n) |n> / ...

    Tr(exp(-beta H)) = sum_n <n| exp(-beta E_n) | n> = sum_n exp(-beta E_n)

    energy E_n   and basis |n>   corresponds to particle number N
    energy E_n'  and basis |n'>  corresponds to particle number N-1
    energy E_n'' and basis |n''> corresponds to particle number N-2

    F_jk(t) = sum_n <n|
                    sum_n exp(i E_n t) * |n><n| b_j^t sum_n' exp(-i E_n' t)
                    sum_n' |n'><n'| b_k^t 
                    sum_n'' exp(i E_n'' t) |n''><n''| b_j (sum_n' exp(-i E_n' t) |n'><n'|)
                    b_k
                    |n> / ...
    """



    evals_N0 = hamil_class.evals()
    evecs_N0 = hamil_class.evecs()


    hamil_class_N1 = AnyonHubbardHamiltonian(
        bc=hamil_class.bc,
        L=hamil_class.L,
        N=hamil_class.N-1,
        J=hamil_class.J,
        U=hamil_class.U,
        theta=hamil_class.theta
    )

    evals_N1 = hamil_class_N1.evals()
    evecs_N1 = hamil_class_N1.evecs()

    hamil_class_N2 = AnyonHubbardHamiltonian(
        bc=hamil_class.bc,
        L=hamil_class.L,
        N=hamil_class.N-2,
        J=hamil_class.J,
        U=hamil_class.U,
        theta=hamil_class.theta
    )

    evals_N2 = hamil_class_N2.evals()
    evecs_N2 = hamil_class_N2.evecs()


    # action of b_j on eigenstates |n> in basis (N-1)
    b_k_rect_mat = get_b_i_mat(
        basis_left=hamil_class_N1.basis.basis_list,
        basis_right=hamil_class.basis.basis_list,
        ind_i=site_i-1
    )

    # action of b_j on eigenstates |n> in basis (N-1)
    b_j_rect_mat = get_b_i_mat(
        basis_left=hamil_class_N2.basis.basis_list,
        basis_right=hamil_class_N1.basis.basis_list,
        ind_i=site_j-1
    )

    # action of b_k^t on eigenstates |n-2> in basis (N-1)
    bk_t_rect_mat = get_b_i_dagger_mat(
        basis_left=hamil_class_N1.basis.basis_list,
        basis_right=hamil_class_N2.basis.basis_list,
        ind_i=site_i-1
    )

    bj_t_rect_mat = get_b_i_dagger_mat(
        basis_left=hamil_class.basis.basis_list,
        basis_right=hamil_class_N1.basis.basis_list,
        ind_i=site_j-1
    )
    

    tr_H = np.sum(np.exp( -1*beta*hamil_class.evals()))
    otoc_list = []
    for t in time:
        sum_t = 0
        for eigval, evec in zip(hamil_class.evals(), hamil_class.evecs()):

            # b_k |n>
            step1 = b_k_rect_mat.dot(evec)

            # |n'><n'| [ ...|n> ]
            step2 = apply_H(t, evals_N1, evecs_N1, step1)

            # b_j [ ...|n> ]
            step3 = b_j_rect_mat.dot(step2)

            # exp(i E_n'' t) |n''><n''| [ ...|n> ]
            step4 = apply_H((-1)*t, evals_N2, evecs_N2, step3)

            # b_k^t [ ...|n> ]
            step5 = bk_t_rect_mat.dot(step4)

            # exp(-i E_n' t) |n'><n'| [ ... ]
            step6 = apply_H(t, evals_N1, evecs_N1, step5)
 
            # b_j^t [  ]
            step7 = bj_t_rect_mat.dot(step6)

            # exp(i E_n t) |n><n| [ ... ]
            step8 = apply_H((-1)*t, evals_N0, evecs_N0, step7)

            n_jkjk_n = np.vdot(evec, step8)

            sum_t += n_jkjk_n * np.exp(-beta * eigval )

        #     op_bjbibibj = operators.get_bjbibibj_mat(
        #         basis_list=hamil_class.basis.basis_list,
        #         i=site_i-1,
        #         j=site_j-1,
        #         N=hamil_class.N
        #     )
        #     print('op  ', np.vdot(evec, op_bjbibibj.dot(evec)))
        #     print('otoc', n_jkjk_n)
        #     print()
        # exit()
        
        otoc_list.append(sum_t/tr_H)
    
    print(np.abs(np.array(otoc_list).imag))
    return np.abs(np.array(otoc_list))

    # plt.plot(time, np.array(otoc_list).real)
    # plt.plot(time, np.array(otoc_list).imag)
    # plt.show()
    # plt.close()
