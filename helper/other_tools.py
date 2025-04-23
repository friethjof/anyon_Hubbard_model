import os
import math
import numpy as np
from datetime import datetime


def time_str(sec):
    hour = sec // 3600
    sec %= 3600
    min = sec // 60
    sec %= 60
    return f'{int(hour)}h:{int(min)}m:{sec:.2f}s'


def appears_in_set(input_set, input_val, threshold):
    """Add input_val to input_set if no item in input_set is close to input_val

    Args:
        input_set (set): set
        input_val (float): val
        threshold (float): limit

    Returns:
        bool : whether input_val has been added or not
    """

    nearest_list = [el for el in input_set if abs(input_val - el) < threshold]
    if len(nearest_list) == 0:
        input_set.add(input_val)
        return True
    else:
        assert len(nearest_list) == 1
        return False


def find_degeneracies(eval_in, threshold=1e-10):
    """Determine the degeneracies of a given input list.

    Returns
    dict : degeneracies as keys and list of indices where they appear as values
    """


    # initialize first value
    dict_energies = {}

    for i, energy in enumerate(eval_in):


        # eval_keys = [eval(el) for el in dict_energies.keys()]
        bool_found = False
        for key in dict_energies.keys():
            # energy is within key
            if np.abs(eval(key) - energy) < threshold:
                bool_found = True
                dict_energies[key].append(i)

            if bool_found:
                break

        # make new entry
        if not bool_found:
            dict_energies[str(energy)] = [i]


        # # check whether energy appears in set
        # bool_exists = appears_in_set(energy_set, energy, threshold)

        # if bool_exists:
        #     # add to list


        # else:
        #     # make new entry
        #     dict_energies[str(energy)] = [i]

        # if bool_add:
        #     add_to_dict(dict_energies)

        #     dict_energies[energy_prev] = state_ind_list

        #     # initialize new
        #     state_ind_list = [i]

        # else:
        #     state_ind_list.append(i)

        # energy_prev = str(energy)

    # dict_energies[energy_prev] = state_ind_list

    return dict_energies

# def get_path_prop(path_data_top, L, N, U, theta, psi_ini):
#     """Create path of data from input pars. J is assumed to be always 1
#
#     Args:
#         path_data_top (Path): top path of data folder
#         L (int): number of lattice sites
#         N (int): number of atoms
#         U (float): on-site interaction
#         theta (float): statistical angle
#         psi_ini (str): initial state
#
#     Returns:
#         path_prop (Path): path where propagation is stored
#     """
#
#     path_basis = get_path_basis(path_data_top, L, N, U, theta)
#     path_prop = path_basis/f'{psi_ini}'
#     path_prop.mkdir(parents=True, exist_ok=True)
#
#     return path_prop


def make_schmidt_decomp(bipartitepurestate_tensor):
    """https://github.com/stephenhky/pyqentangle/blob/master/pyqentangle/
    schmidt.py"""
    state_dims = bipartitepurestate_tensor.shape
    mindim = np.min(state_dims)
    vecs1, diags, vecs2_h = np.linalg.svd(bipartitepurestate_tensor)
    vecs2 = vecs2_h.transpose()
    decomposition = [(diags[k], vecs1[:, k], vecs2[:, k])
        for k in range(mindim)]
    decomposition = sorted(decomposition, key=lambda dec: dec[0], reverse=True)
    return decomposition




def vec_in_subspace(mat, vec):
    """Determine wether a given vector lies within a subsapce, i.e.,
    whether the vector can be written as superposition of basis vectors:
    min||Ax-v||
    """

    x = np.linalg.lstsq(mat, vec, rcond=None)[0]
    vec_min = np.abs(mat.dot(x) - vec)
    length_vec = np.abs(np.vdot(vec_min, vec_min))**2

    return length_vec


def is_file_older_than(path_file, date_list):
    cutoff_date = datetime(*date_list)
    file_time = datetime.fromtimestamp(os.path.getmtime(path_file))
    if file_time < cutoff_date:
        return True
    return False


def orthonormalize_input_vec(vec_list):
    "return a set of orthogonal and normed vectors"

    # Q, R = np.linalg.qr(vec_list)

    # ortho_bool = True

    # vec_out = []
    # for i, vec1 in enumerate(vec_list):

    #     for j, vec2 in enumerate(vec_list):
    #         sp = np.vdot(vec1, vec2)
    #         if i == j and np.abs(sp - 1) > 1e-10:
    #             ortho_bool = False

    #         if i != j and np.abs(sp) > 1e-10:
    #             ortho_bool = False
    # assert ortho_bool

    def gram_schmidt(vectors):
        basis = []
        for v in vectors:
            w = v - np.sum( np.vdot(v,b)*b  for b in basis )
            if (w > 1e-10).any():  
                basis.append(w/np.linalg.norm(w))
        return np.array(basis)



    return gram_schmidt(vec_list)
