from pathlib import Path

import scipy
import numpy as np
import matplotlib.pyplot as plt

from hamiltonian.solve_hamilt import AnyonHubbardHamiltonian
from propagation.scan_propagation import ScanPropagation
from propagation.scan_schmidt_echo import ScanClass_SchmidtEcho

#-------------------------------------------------------------------------------
# ground state analysis
#-------------------------------------------------------------------------------
if False:
    th_list = np.array([0.0, 1.0])*np.pi
    # th_list = np.array([0, 0.2, 0.5, 0.8, 1.0])*np.pi
    # th_list = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])*np.pi
    # th_list = np.array([0.6, 0.7, 0.8, 0.9, 1.0])*np.pi

    L_list = [100]
    # L_list = [60, 110, 120, 130, 140, 150]


    # L_list = [60]

    U = 0
    N = 2

    for L in L_list:
        for th in th_list:
            hamil_class = AnyonHubbardHamiltonian(
                bc='open',
                # bc='twisted_gauge',
                L=L,
                N=N,
                J=1, 
                U=U,
                theta=th,
                bool_save=True
            )

            print(f'L={L}', f'theta={round(th/np.pi,1)}pi', f'basis size={hamil_class.basis.length}')

            #-------------------------------------------------------------------
            # save evals fpr level statistics
            #-------------------------------------------------------------------
            if True:
                if hamil_class.bc == 'twisted_gauge':
                    hamil_class.save_evals_in_same_U_sector()


                elif hamil_class.bc == 'open':

                    path_txt = Path('/afs/physnet.uni-hamburg.de/users/zoq_t/'
                    'ftheel/Schreibtisch/project_ahm2/level_statistics/'
                    'data_obc')

                    if np.abs(th) < 1e-10 or np.abs(th -np.pi) < 1e-10:
                         hamil_class.save_evals_in_same_UI_sector(path_txt)

                    else:
                        hamil_class.evals()

                        path_name = f'L{L}_N{N}_U{round(U, 8)}_thpi_{round(th/np.pi, 8)}'.replace('.', '_')
                        with open(path_txt/path_name, 'w+') as f:
                            for energy in hamil_class.evals():
                                f.write(str(round(energy, 16)) + '\n')


#-------------------------------------------------------------------------------
# out-of-time-ordered correlator (OTOC)
#-------------------------------------------------------------------------------
if False:

    scan_prop = ScanPropagation(
        bc = 'open',
        U = 2,
        J = 1,
        L = 7,
        N = 4,
        theta=np.pi/4,
        Tprop = 5,
        dtprop = 0.1,
    )

    scan_prop.OTOC_vary_site_j(
        beta=1/6,
        fix_site_k=4
    )



#-------------------------------------------------------------------------------
# Loschmidt echo
#-------------------------------------------------------------------------------
if False:
    se_class = ScanClass_SchmidtEcho(
        bc = 'open',
        U = 0.1,
        J = 1,
        theta=0.5*np.pi,
        L = 10,
        N = 2,
        Tprop = 10000,
        dtprop = 1
    )
    time = np.logspace(-1, 5, 100)
    se_class.time = time

    # se_class.psi0_str = 'psi0_n00n'
    se_class.psi0_str = 'psi0_GS'



    se_class.d_U_list = np.array([0.01, 0.1, 1.0])
    se_class.d_th_list = np.array([0.01, 0.1, 1.0])*np.pi


    # se_class.vary_delta_theta()
    se_class.vary_delta_U()


if False:
    N_rand = 100
    t_bin_size = 10
    t_bin_delta = 0.2

    time = np.logspace(-1, 5, 100)


    #---------------------------------------------------------------------------
    # vary theta
    #---------------------------------------------------------------------------
    if True:
        se_class = ScanClass_SchmidtEcho(
            bc = 'open',
            J = 1,
            L = 10,
            N = 2,
            U = 0.0,
            d_th=0.05*np.pi,
            # d_U=1,
        )
        se_class.time = time
        se_class.psi0_str = 'random_psi0'


        se_class.theta_list = np.array([0.0, 0.2, 0.5, 0.8, 1.0])*np.pi
        se_class.vary_theta_rand(
            N_rand=N_rand,
            t_bin_size=t_bin_size,
            t_bin_delta=t_bin_delta
        )


    #---------------------------------------------------------------------------
    # vary U
    #---------------------------------------------------------------------------
    if False:

        se_class = ScanClass_SchmidtEcho(
            bc = 'open',
            J = 1,
            L = 10,
            N = 2,
            theta = 1.0 * np.pi,
            d_U=0.0001,
        )
        se_class.time = time
        se_class.psi0_str = 'random_psi0'


        se_class.U_list = np.array([0.0, 0.4, 0.8, 1.0])
        se_class.vary_U_rand(N_rand=N_rand)




#-------------------------------------------------------------------------------
# collision experiments
#-------------------------------------------------------------------------------

if False:

    scan_prop = ScanPropagation(
        bc = 'twisted',
        U = 0,
        J = 1,
        L = 20,
        N = 2,
        Tprop = 20,
        dtprop = 0.1
    )

    # scan_prop.psi0_str = 'psi0_n00n'
    # scan_prop.psi0_str = 'psi0_nstate_0-0-1-1-0-0'
    scan_prop.psi0_str = 'psi0_nstate_0-0-0-0-0-0-0-0-0-1-1-0-0-0-0-0-0-0-0-0'


    scan_prop.theta_list = np.array([0, 0.2, 0.5, 0.8, 1]) * np.pi
    

    scan_prop.vary_theta_single(obs_name='num_op')
    scan_prop.vary_theta_single(obs_name='nstate_SVN')


    # scan_prop.vary_theta_multi(obs_name='nstate_SVN')

    # scan_prop.vary_theta_multi_rand_ini(
    #     obs_name='nstate_svn_rand_max',
    #     args_dict={
    #         'rand_n':200
    #     }
    # )




#-------------------------------------------------------------------------------
# quench dynamics -> quench theta
#-------------------------------------------------------------------------------

if False:

    scan_prop = ScanPropagation(
        bc = 'open',
        L = 10,
        N = 2,
        U = 1,
        J = 1,
        Tprop = 20,
        dtprop = 0.1
    )

    scan_prop.psi0_str = 'psi0_GS'

    scan_prop.th_ini_list = np.array([0.0, 0.2, 1.0]) * np.pi
    scan_prop.th_fin_list = np.array([0.0, 0.2, 0.5, 0.8, 1.0]) * np.pi
    

    # scan_prop.quench_theta_single(obs_name='num_op')



#-------------------------------------------------------------------------------
# quench dynamics -> quench U
#-------------------------------------------------------------------------------

if False:

    scan_prop = ScanPropagation(
        bc = 'open',
        L = 10,
        N = 2,
        J = 1,
        theta = 0.2*np.pi,
        Tprop = 20,
        dtprop = 0.1
    )

    scan_prop.psi0_str = 'psi0_GS'

    scan_prop.U_ini_list = [0, 1]
    scan_prop.U_fin_list = [0, 1, 10]
    
    # scan_prop.quench_U_single(obs_name='num_op')
