from adv_figures import energy_spectrum
from adv_figures import energy_spectrum_zoom
from adv_figures import energy_spectrum_zoom_v2
from adv_figures import energy_spectrum_pbc_single
from adv_figures import energy_spectrum_tgbc_single
from adv_figures import energy_spectrum_thpi_U
from adv_figures import loschmidt_echo
from adv_figures import loschmidt_echo_fft
# from adv_figures import level_statistics_obc
from adv_figures import level_statistics_obc
from adv_figures import level_statistics_tgbc
from adv_figures import fidelity
from adv_figures import fidelity_energy_randstates



# energy_spectrum.make_plot()
# energy_spectrum_zoom.make_plot()
# energy_spectrum_pbc_single.make_plot()
# energy_spectrum_tgbc_single.make_plot()
# energy_spectrum_zoom_v2.make_plot()

# energy_spectrum_thpi_U.make_plot()


# loschmidt_echo.make_plot()
# loschmidt_echo_fft.make_plot()

# level_statistics_obc.make_plot()
# level_statistics_tgbc.make_plot()

# L = 20
# for U in [0, 0.5, 1.0, 2.0]:
#     fidelity.make_plot(L=L, U=U, E_mid=0.0, E_range=0.1)
#     fidelity.make_plot(L=L, U=U, E_mid=0.1, E_range=0.1)
#     fidelity.make_plot(L=L, U=U, E_mid=0.2, E_range=0.1)
#     fidelity.make_plot(L=L, U=U, E_mid=0.3, E_range=0.1)
#     fidelity.make_plot(L=L, U=U, E_mid=0.4, E_range=0.1)
#     fidelity.make_plot(L=L, U=U, E_mid=0.0, E_range=0.2)
#     fidelity.make_plot(L=L, U=U, E_mid=0.2, E_range=0.2)
#     fidelity.make_plot(L=L, U=U, E_mid=0.3, E_range=0.2)


# for U in [0.0, 0.7]:#, 0.1, 0.2, 0.5, 0.7, 0.8, 0.9, 1.0]:
# # for U in [0, -0.1, -0.2]:
# # for U in [ -0.05, -0.01, -0.005]:
#     for L in [10, 15, 20, 40]:
#         # fidelity.make_plot(L=L, U=U, t_bin_delta=0.02)
#         fidelity.make_plot(L=L, U=U, t_bin_delta=0.05, theta_set=1, type='randstates_gauss', E_mid=0.1, E_range=0.2)
#         fidelity.make_plot(L=L, U=U, t_bin_delta=0.05, theta_set=2, type='randstates_gauss', E_mid=0.1, E_range=0.2)
#         # fidelity.make_plot(L=L, U=U, t_bin_delta=0.07)
#         # fidelity.make_plot(L=L, U=U, t_bin_delta=0.10)

# L = 40
# U = 0
# fidelity.make_plot(L=L, U=U, t_bin_delta=0.02)
# fidelity.make_plot(L=L, U=U, t_bin_delta=0.05)
# fidelity.make_plot(L=L, U=U, t_bin_delta=0.07)
# fidelity.make_plot(L=L, U=U, t_bin_delta=0.10)

# fidelity.make_plot()
# fidelity_energy_randstates.make_plot()

for th_set in [1, 2]:
    for U in [0.0, 0.7]:#, 0.1, 0.2, 0.5, 0.7, 0.8, 0.9, 1.0]:
        for L in [40]:
            for t_bin_delta in [0.20]:# [0.05, 0.10, 0.20]:

                for E_par in [
                    [0.0, 0.1],
                    [0.0, 0.2],
                    [0.0, 0.3],
                    [0.1, 0.1],
                    [0.2, 0.1],
                    [0.2, 0.2],
                ]:

                    E_mid, E_range = E_par
                    fidelity.make_plot(
                        L=L,
                        U=U,
                        t_bin_delta=t_bin_delta,
                        theta_set=th_set,
                        type='randstates_gauss',
                        E_mid=E_mid,
                        E_range=E_range
                    )
