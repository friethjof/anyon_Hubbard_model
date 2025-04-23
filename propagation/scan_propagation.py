import math
import gc
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from hamiltonian.solve_hamilt import AnyonHubbardHamiltonian
from propagation.propagate_static import PropagationStatic


from propagation import calc_otoc
from helper import path_dirs
from helper import other_tools
from helper import plot_helper
from helper import operators


class ScanPropagation():
    """Organize exact diagonalization method, plotting, scaning of paramters.
    """

    def __init__(self, **args_in):
        """Initialize parameters.

        Args:
            bc (str): boundary conditions of lattice, open (hard-wall) | periodic
            L (int): number of lattice sites
            N (int): number of atoms
            J (float): hopping constant
            U (float): on-site interaction
            theta (float): statistical angle

            psi0 (str): specifies initial state for propagation
            Tprop (float): final time of propagation
            dtprop (float): time steps of propagation
        """

        self.bc = args_in.get('bc', None)
        self.L = args_in.get('L', None)
        self.N = args_in.get('N', None)
        self.U = args_in.get('U', None)
        self.J = args_in.get('J', None)
        self.theta = args_in.get('theta', None)

        # self.psi0_class = args_in.get('psi0_class', None)
        self.psi0_str = args_in.get('psi0_str', None)
        self.Tprop = args_in.get('Tprop', None)
        self.dtprop = args_in.get('dtprop', None)

        # vary over these theta values
        self.theta_list = [self.theta]

        # quench theta
        self.th_ini_list = None
        self.th_fin_list = None

        # quench U
        self.U_ini_list = None
        self.U_fin_list = None


    #===========================================================================
    def get_path_fig(self, **args_in):
        path_fig = path_dirs.get_path_fig_top_dyn(
            bc=self.bc,
            L=self.L,
            N=self.N,
            J=args_in.get('J_ini', self.J),
            U=args_in.get('U_ini', self.U),
            psi0_str=args_in.get('psi0_name', self.psi0_str),
            Tprop=self.Tprop,
            dtprop=self.dtprop
        )

        if 'theta_ini' in args_in.keys():
            th_ = f'{round(args_in["theta_ini"]/np.pi, 8)}'.replace('.', '_')
            path_fig = (
                path_fig.parent/
                f'quench_theta_ini_{th_}pi_{path_fig.name}'
            )

        if 'U_ini' in args_in.keys():
            U_ = f'{round(args_in["U_ini"], 8)}'.replace('.', '_')
            path_fig = path_fig.parent/f'quench_U_ini_{U_}_{path_fig.name}'

        return path_fig


    def get_psi0_ini_class(self, **args_in):
        assert isinstance(self.bc, str)
        assert isinstance(self.L, int)
        assert isinstance(self.N, int)

        hamil_class = AnyonHubbardHamiltonian(
            bc=self.bc,
            L=self.L,
            N=self.N,
            J=args_in.get('J', self.J),
            U=args_in.get('U', self.U),
            theta=args_in.get('theta', self.theta)
        )
        return hamil_class 



    def get_prop_class(self, **args_in):

        assert isinstance(self.bc, str)
        assert isinstance(self.L, int)
        assert isinstance(self.N, int)
        assert isinstance(self.psi0_str, str)
        assert isinstance(self.Tprop, (int, float))
        assert isinstance(self.dtprop, (int, float))

        prop_class = PropagationStatic(
            bc=self.bc,
            L=self.L,
            N=self.N,
            J=args_in.get('J', self.J),
            U=args_in.get('U', self.U),
            theta=args_in.get('theta', self.theta),
            psi0_str=self.psi0_str,
            Tprop=self.Tprop,
            dtprop=self.dtprop,
            psi0_ini_class=args_in.get('psi0_ini_class', None),
        )
        return prop_class


    def get_prop_class_quench(self, **args_in):

        psi0_class = self.get_psi0_ini_class(
            J=args_in.get('J_ini', self.J),
            U=args_in.get('U_ini', self.U),
            theta=args_in.get('theta_ini', self.theta),
        )

        prop_class = self.get_prop_class(
            J=args_in.get('J_prop', self.J),
            U=args_in.get('U_prop', self.U),
            theta=args_in.get('theta_prop', self.theta),
            psi0_ini_class=psi0_class
        )

        return prop_class


    #===========================================================================
    # plotting
    #===========================================================================
    def make_single_plot(self, obs_name, prop_class, fig_name, title):

        #-----------------------------------------------------------------------
        # <n_i>(t)
        #-----------------------------------------------------------------------
        if obs_name == 'num_op':
            time, num_op = prop_class.num_op_mat()

            plot_helper.num_op_cplot(
                fig_name=fig_name,
                time=prop_class.time,
                L=self.L,
                num_op_mat=num_op,
                title=title
            )

        #-----------------------------------------------------------------------
        # nstate_SVN
        #-----------------------------------------------------------------------
        if obs_name == 'nstate_SVN':
            svn, svn_max = prop_class.nstate_SVN()
            plot_helper.plot_lines(
                fig_name=fig_name,
                x_lists=[prop_class.time],
                y_lists=[svn],
                fig_args={
                    'xlabel':'time',
                    'ylabel':'SVN',
                    'title':title,
                    'hline':svn_max
                }
            )


    #===========================================================================
    def vary_theta_single(self, obs_name, args_dict=None):


        assert self.bc is not None
        assert self.L is not None
        assert self.N is not None
        assert self.psi0_str is not None
        assert self.Tprop is not None
        assert self.dtprop is not None


        for theta in self.theta_list:

            self.theta = theta

            print(f'theta/pi={theta/np.pi:.3f}')


            #---------------------------------------------------------------
            # get paths


            th_ = f'thpi_{round(theta/np.pi, 8)}'.replace('.', '_')

            fig_name = self.get_path_fig()/f'{obs_name}'/f'{th_}_{obs_name}'
            title = plot_helper.make_title(
                L=self.L,
                N=self.N,
                U=self.U,
                theta=theta,
                psi0=self.psi0_str
            )

            #---------------------------------------------------------------
            # make plot
            self.make_single_plot(
                obs_name=obs_name,
                prop_class=self.get_prop_class(),
                fig_name=fig_name,
                title=title
            )



            #-------------------------------------------------------------------
            gc.collect()


    #===========================================================================
    def vary_theta_multi(self, obs_name, args_dict=None):
        """Compared to vary_theta_single, here the for loop is shifted inside
        the if statement allowing for a multi-plot
        """

        assert self.bc is not None
        assert self.L is not None
        assert self.N is not None
        assert self.Tprop is not None
        assert self.dtprop is not None


        label_list = []
        for theta in self.theta_list:
            label_list.append(f'theta/pi={theta/np.pi:.3f}')



        #-----------------------------------------------------------------------
        # nstate_SVN
        #-----------------------------------------------------------------------
        if obs_name == 'nstate_SVN':

            fig_name = self.get_path_fig()/f'{obs_name}_multi'/f'{obs_name}_multi'
            title = plot_helper.make_title(
                L=self.L,
                N=self.N,
                U=self.U,
                theta=None,
                psi0=self.psi0_str
            )

            svn_list = []
            for theta in self.theta_list:
                self.theta = theta
                print(f'theta/pi={theta/np.pi:.3f}')
                prop_class = self.get_prop_class()
                svn, svn_max = prop_class.nstate_SVN()
                svn_list.append(svn)

            plot_helper.plot_lines(
                fig_name=fig_name,
                x_lists=[prop_class.time],
                y_lists=svn_list,
                label_list=label_list,
                fig_args={
                    'xlabel':'time',
                    'ylabel':'SVN',
                    'title':title,
                    'hline':svn_max
                }
            )



    #===========================================================================
    # random initial states
    #===========================================================================
    def get_rand_prop(self):
        
        prop_classs = self.get_prop_class()
    
        ini_size = prop_classs.basis.length
        rand_vec = np.random.uniform(0, 1, ini_size) + 1.j * np.random.uniform(0, 1, ini_size)
        rand_vec /= np.sqrt(np.vdot(rand_vec, rand_vec))

        prop_classs.psi0 = rand_vec
        prop_classs.bool_save = False

        return prop_classs


    def vary_theta_multi_rand_ini(self, obs_name, args_dict=None):
        """Compared to vary_theta_single, here the for loop is shifted inside
        the if statement allowing for a multi-plot
        """

        assert self.bc is not None
        assert self.L is not None
        assert self.N is not None
        assert self.Tprop is not None
        assert self.dtprop is not None

        rand_n = args_dict['rand_n']

        label_list = []
        for theta in self.theta_list:
            label_list.append(f'theta/pi={theta/np.pi:.3f}')

        fig_name = self.get_path_fig(psi0_name='psi0_rand')/f'{obs_name}_multi'/f'{obs_name}_multi'
        title = plot_helper.make_title(
            L=self.L,
            N=self.N,
            U=self.U,
            theta=None,
            psi0=f'rand_n={rand_n}'
        )

        #-----------------------------------------------------------------------
        # nstate_svn_rand_max
        #-----------------------------------------------------------------------
        if obs_name == 'nstate_svn_rand_max':
            

            fig_name = fig_name.parent/(fig_name.name + f'rand_n_{rand_n}')

            svn_max_list = []
            for theta in self.theta_list:
                self.theta = theta
                print(f'theta/pi={theta/np.pi:.3f}')


                svn_rand_max_list = []
                for i in range(rand_n):
                    prop_class = self.get_rand_prop()
                    svn, svn_upper_bound = prop_class.nstate_SVN()
                    svn_max = np.max(svn)
                    print('rand', i+1, 'svn_max', svn_max)
                    svn_rand_max_list.append(svn_max)

                svn_max_list.append(max(svn_rand_max_list))


            plot_helper.plot_lines(
                fig_name=fig_name,
                x_lists=[self.theta_list],
                y_lists=[svn_max_list],
                label_list=label_list,
                fig_args={
                    'xlabel':r'$\theta$',
                    'ylabel':'SVN',
                    'title':title,
                    'hline':svn_upper_bound
                }
            )


    #===========================================================================
    # quench theta
    #===========================================================================
    def quench_theta_single(self, obs_name):
        "Quench theta from th_ini to th_prop"


        for theta_ini in self.th_ini_list:

            path_fig = self.get_path_fig(theta_ini=theta_ini)

            for theta_prop in self.th_fin_list:

                if theta_ini == theta_prop:
                    continue

                prop_class = self.get_prop_class_quench(
                    theta_ini=theta_ini,
                    theta_prop=theta_prop
                )

                print_str = f'quench: theta: {round(theta_ini/np.pi, 3)}'
                print_str += f' -> {round(theta_prop/np.pi, 3)}'
                print(print_str)


                th_prop_ = f'th_prop_{round(theta_prop/np.pi, 8)}pi'.replace('.', '_')
                fig_name = path_fig/obs_name/f'{th_prop_}_{obs_name}'

                title = plot_helper.make_title(
                    L=self.L,
                    N=self.N,
                    U=self.U,
                    theta_ini=theta_ini,
                    theta_prop=theta_prop,
                    psi0=self.psi0_str
                )

                #---------------------------------------------------------------
                # make plot
                self.make_single_plot(
                    obs_name=obs_name,
                    prop_class=prop_class,
                    fig_name=fig_name,
                    title=title
                )



    #===========================================================================
    # quench U
    #===========================================================================
    def quench_U_single(self, obs_name):
        "Quench theta from th_ini to th_prop"

        th_ = f'th_prop_{round(self.theta/np.pi, 8)}pi'.replace('.', '_')

        for U_ini in self.U_ini_list:

            path_fig = self.get_path_fig(U_ini=U_ini)

            for U_prop in self.U_fin_list:

                if U_ini == U_prop:
                    continue

                prop_class = self.get_prop_class_quench(
                    U_ini=U_ini,
                    U_prop=U_prop
                )

                print(f'quench: U: {round(U_ini, 3)} -> {round(U_prop, 3)}')


                U_prop_ = f'U_prop_{round(U_prop, 8)}'.replace('.', '_')
                fig_name = path_fig/obs_name/f'{th_}_{U_prop_}_{obs_name}'

                title = plot_helper.make_title(
                    L=self.L,
                    N=self.N,
                    theta=self.theta,
                    U_ini=U_ini,
                    U_prop=U_prop,
                    psi0=self.psi0_str
                )


                #---------------------------------------------------------------
                # make plot
                self.make_single_plot(
                    obs_name=obs_name,
                    prop_class=prop_class,
                    fig_name=fig_name,
                    title=title
                )



    #===========================================================================
    # out-of-time-ordered correlator (OTOC)
    #===========================================================================
    def OTOC_vary_site_j(self, beta, fix_site_k):
        'try to reproduce Fig. 3 10.1103/PhysRevLett.121.250404'

        L_list = list(range(1, self.L+1))
        time = np.arange(0, self.Tprop+self.dtprop, self.dtprop)
        hamil_class = self.get_psi0_ini_class(theta=self.theta)

        OTOC_jk = []
        time_75_perc = []
        for site_j in L_list:

            otoc_ij = calc_otoc.get_otoc_ij_beta(
                hamil_class=hamil_class,
                site_i=site_j,
                site_j=fix_site_k,
                beta=beta,
                time=time,
            )
            OTOC_jk.append(otoc_ij)

            val_75perc = otoc_ij[0] * 0.75
            t_ind = np.argmin(np.abs(otoc_ij-val_75perc))
            time_75_perc.append(time[t_ind])



        x, y = np.meshgrid(L_list, time)
        im = plt.pcolormesh(x, y, np.array(OTOC_jk).T, shading='auto', cmap='viridis')
        plt.colorbar(im)
        plt.plot(L_list, time_75_perc, color='red')
        plt.show()