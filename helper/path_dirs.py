import os
import math
from pathlib import Path
import decimal
from typing import Callable
from collections.abc import Iterable


adv_fig = Path('/afs/physnet.uni-hamburg.de/users/zoq_t/ftheel/'
        'Schreibtisch/project_ahm2/nice_figures')


def get_path_names(type, bc):
    if type == 'basis':
        path_basis = Path(f'/afs/physnet.uni-hamburg.de/project/zoq_t/harm_osc_coll/13/project_ahm_integrability')
        if bc == 'open':
            return path_basis/'open_boundary_conditions'
        elif bc == 'periodic':
            return path_basis/'periodic_boundary_conditions'
        elif bc == 'twisted':
            return path_basis/'twisted_boundary_conditions'
        elif bc == 'twisted_gauge':
            return path_basis/'twisted_gauge_boundary_conditions'
        else:
            raise NotImplementedError

    elif type == 'fig':
        path_proj = Path(f'/afs/physnet.uni-hamburg.de/users/zoq_t/ftheel/Schreibtisch/project_ahm2')
        if bc == 'open':
            return path_proj/'figures_obc'
        elif bc == 'periodic':
            return path_proj/'figures_pbc'
        elif bc == 'twisted':
            return path_proj/'figures_tbc'
        elif bc == 'twisted_gauge':
            return path_proj/'figures_tgbc'
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError


def get_theta_str(theta):
    if isinstance(theta, Iterable):
        if all([theta[0]==el for el in theta]):
            th_ = f'thpi_{round(theta[0]/math.pi, 8)}'.replace('.', '_')
        else:
            th_ = f'thpiVec'
            for th_el in theta:
                th_ += f'_{round(th_el/math.pi, 8)}'.replace('.', '_')
    else:
        th_ = f'thpi_{round(theta/math.pi, 8)}'.replace('.', '_')
    return th_


def get_J_str(J):
    if isinstance(J, Iterable):
        if all([J[0]==el for el in J]):
            J_ = f'J_{round(J[0], 8)}'.replace('.', '_')
        else:
            J_ = f'Jvec'
            for J_el in J:
                J_ += f'_{round(J_el, 8)}'.replace('.', '_')
    else:
        J_ = f'J_{round(J, 8)}'.replace('.', '_')
    return J_


def get_U_str(U):
    if isinstance(U, Iterable):
        if all([U[0]==el for el in U]):
            U_ = f'U_{round(U[0], 8)}'.replace('.', '_')
        else:
            U_ = f'Uvec'
            for U_el in U:
                U_ += f'_{round(U_el, 8)}'.replace('.', '_')
    else:
        U_ = f'U_{round(U, 8)}'.replace('.', '_')
    return U_


def get_path_basis(bc, L, N, U, theta, J, bool_mkdir=True):
    """Create path of data from input pars. J is assumed to be always 1

    Args:
        path_data_top (Path): top path of data folder
        L (int): number of lattice sites
        N (int): number of atoms
        U (float): on-site interaction
        theta (float): statistical angle

    Returns:
        path_basis (Path): path where solved hamilt spectrum is stored
    """

    assert isinstance(bc, str)
    th_ = get_theta_str(theta)
    J_ = get_J_str(J)
    U_ = get_U_str(U)

    path_basis = get_path_names('basis', bc)/f'L{L}_N{N}/{U_}_{J_}'/f'{th_}'
    if bool_mkdir:
        path_basis.mkdir(parents=True, exist_ok=True)

    return path_basis


def get_path_fig_top(bc, L, N, J, U):
    J_ = get_J_str(J)
    fig_dir = f'L{L}_N{N}_{J_}'
    if U is not None:
        U_ = get_U_str(U)
        fig_dir += f'_{U_}'

    return get_path_names('fig', bc)/fig_dir


def get_path_fig_top_dyn(bc, L, N, J, U, psi0_str, Tprop=None, dtprop=None):
    path_fig_top = get_path_fig_top(bc, L, N, J, U)
    if Tprop is None:
        dum_Tf_dt = ''
    else:
        dum_Tf_dt = '_' + get_dum_Tf_dt(Tprop, dtprop)

    return path_fig_top/f'{psi0_str}{dum_Tf_dt}'


def get_path_fig_top_dyn_multi(bc, L, N, psi0s_str, Tprop=None, dtprop=None):
    path_fig_top = get_path_fig_top(bc, L, N)
    if Tprop is None:
        dum_Tf_dt = ''
    else:
        dum_Tf_dt = '_' + get_dum_Tf_dt(Tprop, dtprop)

    return path_fig_top/f'scan_psi0_{psi0s_str}{dum_Tf_dt}'


def get_dum_Tf_dt(Tprop, dtprop):
    "Determine string which specifies the propagation limits"
    if Tprop % 1 != 0:
        raise ValueError(f'Tprop={Tprop} has to be integer!')
    dum_Tf_dt = f'Tf_{round(Tprop)}'
    if dtprop != 0.1:
        if (dtprop - round(dtprop, 3)) != 0:
            err_msg = f'self.dtprop={dtprop} only 3 digits allowed!'
            raise ValueError(err_msg)
        dtprop_ = f'{round(dtprop, 3)}'.replace('.', '_')
        dum_Tf_dt += f'_dt_{dtprop_}'
    return dum_Tf_dt


# def get_dum_name_U_theta(U, theta, J=None):
#     '''Determine part of figure name specifying U, theta, J, if J is None
#     assume J=1 and don't mention it. If J=0 and U=1, use U='inf'
#     '''
#     theta_ = f'{theta/math.pi:.3f}'.replace('.', '_')
#     U_ = f'{U:.3f}'.replace('.', '_')
#     if J is not None and J != 1:
#         if J == 0 or U == 1:
#             U_theta_ = f'U_inf_thpi_{theta_}'
#         else:
#             err_msg = f'U={U}, J={J}; J should be 0 and U=1 for U=inf!'
#             raise ValueError(err_msg)
#     else:
#         U_theta_ = f'U_{U_}_thpi_{theta_}'
#     return U_theta_

