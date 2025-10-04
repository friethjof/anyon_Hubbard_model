import os
import math
from pathlib import Path
from typing import Union, Iterable
#from collections.abc import Iterable



path_data = Path('../data_ahm')
path_fig = Path(f'../figures_ahm')


def get_path_bc(parent_dir: str, bc: str) -> Path:
    """
    Construct main path based on boundary conditions
    
    Args:
        parent_dir (str): specifies whether data or figures directory is sought
        bc (str): boundary conditions

    Returns:
        Path-object
    """

    if parent_dir == 'basis':
        if bc == 'open':
            return path_data/'open_boundary_conditions'
        elif bc == 'periodic':
            return path_data/'periodic_boundary_conditions'
        elif bc == 'twisted':
            return path_data/'twisted_boundary_conditions'
        elif bc == 'twisted_gauge':
            return path_data/'twisted_gauge_boundary_conditions'
        else:
            raise NotImplementedError

    elif parent_dir == 'fig':

        if bc == 'open':
            return path_fig/'figures_obc'
        elif bc == 'periodic':
            return path_fig/'figures_pbc'
        elif bc == 'twisted':
            return path_fig/'figures_tbc'
        elif bc == 'twisted_gauge':
            return path_fig/'figures_tgbc'
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError


def get_theta_str(theta: Union[float, Iterable[float]]) -> str:
    """
    Converts theta-value to string used for storing.
    
    Args:
        theta (float or Iterable): value of statistical pahse

    Returns:
        Name of paramter and value for storing. 
    """

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


def get_J_str(J: Union[float, Iterable[float]]) -> str:
    """
    Converts J-value to string used for storing.
    
    Args:
        J (float or Iterable): value of hopping parameter

    Returns:
        Name of paramter and value for storing. 
    """

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


def get_U_str(U: Union[float, Iterable[float]]) -> str:
    """
    Converts U-value to string used for storing.
    
    Args:
        theta (float or Iterable): value of onsite interaction

    Returns:
        Name of paramter and value for storing. 
    """

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


def get_path_basis(
        bc: str,
        L: int,
        N: int,
        J: Union[float, Iterable[float]],
        U: Union[float, Iterable[float]],
        theta: Union[float, Iterable[float]],
        bool_mkdir: bool=True
    ):
    """
    Create path of data from input pars. J is assumed to be always 1

    Args:
        path_data_top (Path): top path of data folder
        L (int): number of lattice sites
        N (int): number of atoms
        J (float or Iterable): hopping parameter
        U (float or Iterable): on-site interaction
        theta (float or Iterable): statistical angle

    Returns:
        path_basis (Path): path where solved hamilt spectrum is stored
    """

    assert isinstance(bc, str)
    th_ = get_theta_str(theta)
    J_ = get_J_str(J)
    U_ = get_U_str(U)

    path_basis = get_path_bc('basis', bc)/f'L{L}_N{N}/{U_}_{J_}'/f'{th_}'
    if bool_mkdir:
        path_basis.mkdir(parents=True, exist_ok=True)

    return path_basis


def get_path_fig_top(
        bc: str,
        L: int,
        N: int,
        J: Union[float, Iterable[float]],
        U: Union[float, Iterable[float]]
    ):
    """
    Create path the top directory of the figure folder

    Args:
        bc (str): specifier of boundary condition
        L (int): number of lattice sites
        N (int): number of atoms
        J (float or Iterable): hopping parameter
        U (float or Iterable): on-site interaction

    Returns:
        path_basis (Path): path where figures are stored
    """
    J_ = get_J_str(J)
    fig_dir = f'L{L}_N{N}_{J_}'
    if U is not None:
        U_ = get_U_str(U)
        fig_dir += f'_{U_}'

    return get_path_bc('fig', bc)/fig_dir


