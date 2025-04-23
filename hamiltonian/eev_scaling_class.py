import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

from hamiltonian.solve_hamilt import AnyonHubbardHamiltonian

from helper import path_dirs


#-------------------------------------------------------------------------------
class EEVScalingClass():
    """
    Calculate integrable observable following:
        W. Beugeling, R. Moessner, and Masudul Haque, "Finite-size scaling of
        eigenstate thermalization", Phys. Rev. E 89, 042112 (2014)
        https://link.aps.org/doi/10.1103/PhysRevE.89.042112
    
    EEV : eigenstate expectation values

    1) Calculate for a subset of eigenvectors the expectation values of an
        observable.
    2) Calculate standard deviation
    3) Do this for different Hilbert space sizes.
    
    """

    def __init__(self, **args_in):

        self.bc = args_in.get('bc', None)
        self.L_list = args_in.get('L_list', None)
        self.N_list = args_in.get('N_list', None)
        self.U = args_in.get('U', None)
        self.J = args_in.get('J', None)
        self.theta_list = args_in.get('theta_list', None)
        self.obs_name = args_in.get('obs_name', None)
        self.delta_E = args_in.get('delta_E', None)
        self.central_average_range = args_in.get('central_average_range', None)

        self.temp_dict = {}



    #===========================================================================
    # calculate sigma_A^2 for individual L, N, theta
    #===========================================================================
    def get_deltaE(self, L=None):
        if self.delta_E == '0.025L':
            assert L is not None
            return 0.025*L
        elif self.delta_E == '0.0025L':
            assert L is not None
            return 0.0025*L
        elif self.delta_E == '0.01Lsqrt':
            assert L is not None
            return 0.01*np.sqrt(L)
        elif self.delta_E == '2divL':
            return 2/L
        else:
            raise NotImplementedError


    def get_obs_list(self, hamil_class):
        if self.obs_name == 'hop_2_hc':
            return hamil_class.get_hopping_EEV(site_i=2)
        elif self.obs_name == 'hop_8_hc':
            return hamil_class.get_hopping_EEV(site_i=8)


    def get_ham_class(self, L, N, theta):
        return AnyonHubbardHamiltonian(
            bc=self.bc,
            L=L,
            N=N,
            J=self.J, 
            U=self.U,
            theta=theta,
            bool_save=True
        )
         

    def get_microcanonical_average(self, L, N, theta):
        """Eq. (1) of https://link.aps.org/doi/10.1103/PhysRevE.89.042112

        <A> = 1/N sum_{a in E+dE, E-dE} A_{a,a}

        Do this only for the states in the central 20% region of the energy
        spectrum.
        """

        hamil_class = self.get_ham_class(L, N, theta)
        H_dim = hamil_class.basis.length


        #-----------------------------------------------------------------------
        # storing
        #-----------------------------------------------------------------------
        c_range_ = f"{self.central_average_range}".replace('.', '_')
        obs_av_name = "microcan_av"
        obs_av_name += f"_obs_{self.obs_name}"
        obs_av_name += f"_crange_{c_range_}"
        obs_av_name += f"_dE_{self.delta_E}"
        obs_av_name += '.npz'

        path_microcan_av_npz = hamil_class.path_basis/obs_av_name
        # if path_microcan_av_npz.is_file():
        #     return H_dim, np.load(path_microcan_av_npz)['varA_mce_av']

        #-----------------------------------------------------------------------
        energies = hamil_class.evals()
        # plt.scatter(energies/L, energies)
        # plt.show()
        # exit()

        dE = self.get_deltaE(L)
        obs_A = self.get_obs_list(hamil_class)


        E_mean = np.mean(energies)
        c_range = (
            np.abs(np.min(energies) - np.max(energies))
            * self.central_average_range
        )
        E_min = E_mean - c_range
        E_max = E_mean + c_range


        #-----------------------------------------------------------------------
        # Calculate average of microcanonical ensemble (mce)
        #-----------------------------------------------------------------------
        A_ii_var_list_squared = []  # standard0, 2, 0 deviation of average
        A_ii_var_list = []
        av_set_size = []
        for i, energy in enumerate(energies):
            if E_min <= energy and energy <= E_max:

                # mce : micro canonical ensemble boundaries
                e_min_i = np.argmin(np.abs(energies- (energy - dE)))
                e_max_i = np.argmin(np.abs(energies- (energy + dE)))

                obs_A_slice = obs_A[e_min_i:e_max_i+1]
                
                A_ii = obs_A[i]  # bare expectation value
                A_ii_av = np.mean(obs_A_slice)  # avergaged expecation value
                
                av_set_size.append(len(obs_A_slice))

                A_ii_var_list_squared.append( (A_ii - A_ii_av)**2 )
                A_ii_var_list.append( (A_ii - A_ii_av) )



        N_central = len(A_ii_var_list)
        str_print = f'L={L}, N={N}, theta={np.round(theta/np.pi, 3)}pi:'
        str_print += f' central average over {N_central} states'
        str_print += f' => ratio of {round(N_central/energies.shape[0], 3)}'
        str_print += f', avergaged ensemble size = {round(np.mean(av_set_size), 3)}'
        str_print += f', min={np.min(av_set_size)}, max={np.max(av_set_size)}'
        print(str_print)


        varA_mce_av = np.sqrt(np.mean(A_ii_var_list_squared))  # <D A^2 >_c
        varA_mce_av_2 = np.mean(A_ii_var_list) ** 2            # <D A >_c^2

        # print(varA_mce_av)
        # plt.plot(np.array(A_ii_var_list_squared)**2, label='rhs')
        # plt.plot(A_ii_var_list, label='lhs')
        # plt.legend()
        # plt.show()
        # exit()

        np.savez(
            path_microcan_av_npz,
            varA_mce_av=varA_mce_av,     # <D A^2 >_c
            varA_mce_av_2=varA_mce_av_2  # <D A >_c^2
        )

        return H_dim, varA_mce_av, varA_mce_av_2


    #===========================================================================
    # calculate sigma_A for increasing Hilbert space dimension
    #===========================================================================
    def get_sigma_A_wrt_D(self, theta):

        key_D = f'dim_list_theta_{theta}'
        key_sigm_A = f'sigm_A_list_theta_{theta}'
        if key_sigm_A in self.temp_dict.keys():
            return self.temp_dict[key_D], self.temp_dict[key_sigm_A]

        

        D_list = []  # Hilbert space dimension
        sigm_A_list = []  # standard deviation of sigma A

        for N in self.N_list:
            for L in self.L_list:
                N = int(L/2)

                H_dim, varA_mce_av, _ = self.get_microcanonical_average(L, N, theta)

                D_list.append(H_dim)
                sigm_A_list.append(varA_mce_av)
        
        D_list = np.array(D_list)
        sigm_A_list = np.array(sigm_A_list)

        idx = D_list.argsort()
        D_list = D_list[idx]
        sigm_A_list = sigm_A_list[idx]

        self.temp_dict[key_D] = D_list
        self.temp_dict[key_sigm_A] = sigm_A_list

        return D_list, sigm_A_list



    #===========================================================================
    # Fit the behavior of sigma A with respect to the Hilbert space dimension
    #===========================================================================
    def get_D_fit(self, theta):
        """Calculate the best fit for sigma_{Delta A}(dim)
        ~ c0 D^{-e}
        """

        D_list, sigm_A_list = self.get_sigma_A_wrt_D(theta)

        def sigm_A_func(D, c0, e):
            return c0 * D**(-e)

        popt, pcov = curve_fit(
            f=sigm_A_func,
            xdata=D_list,
            ydata=sigm_A_list,
            method='lm'
        )

        c0_fin, e_fin = popt
        sigm_A_fit = sigm_A_func(D_list, c0_fin, e_fin)

        return e_fin, sigm_A_fit
    


    #===========================================================================
    # Determine bin size
    #===========================================================================
    def plot_bin_size(self, theta):
        

        for N in self.N_list:
            for L in self.L_list:
                hamil_class = self.get_ham_class(L, N, theta)
                print(hamil_class.basis.length)
                energies = hamil_class.evals()
                dE = self.get_deltaE(L)
                bin_sizes = []
                for i, energy in enumerate(energies):
                    # mce : micro canonical ensemble boundaries
                    e_min_i = np.argmin(np.abs(energies- (energy - dE)))
                    e_max_i = np.argmin(np.abs(energies- (energy + dE)))

                    bin_sizes.append(e_max_i-e_min_i+1)

                plt.scatter(energies/L, bin_sizes)
                plt.show()
                exit()


    def check_ineq_Eq4(self):
        "check inequality in Eq. (4)"

        for N in self.N_list:
            for L in self.L_list:
                for theta in self.theta_list:
                    H_dim, varA_mce_av, varA_mce_av_2 = \
                        self.get_microcanonical_average(L, N, theta)
                    print(f'N={N}, L={L}, theta={round(theta/np.pi, 3)}pi')
                    print("<D A^2 >_c", varA_mce_av)
                    print("<D A >_c^2", varA_mce_av_2)
                    print()


    #===========================================================================
    # Quick plots
    #===========================================================================
    def plot_D_fit(self, theta):
        D_list, sigm_A_list = self.get_sigma_A_wrt_D(theta)
        e_fin, sigm_A_fit = self.get_D_fit(theta)

        plt.plot()
        plt.title(f'e={e_fin}')
        plt.plot(D_list, sigm_A_list, label='bare')
        plt.plot(D_list, sigm_A_fit, label='fit')
        # plt.yscale('log')
        # plt.xscale('log')
        plt.legend()
        plt.show()
        plt.close()


    def plot_e_fit(self):
        e_list = []
        for theta in self.theta_list:
            e_val, sigm_A_val = self.get_D_fit(theta)
            e_list.append(e_val)

        plt.plot(self.theta_list/np.pi, e_list)
        plt.show()