import numpy as np
import matplotlib.pyplot as plt


class PlotObservables:
    """
    Helper class for plotting observables.

    Keyword Args:
        parent (class): instance of Hamiltonian class

    """

    def __init__(self, parent):

        self.parent = parent

    def _get_title(self) -> str:
        """
        Returns:
            Create string with system parameters.
        """

        title = f"$L={self.parent.L}$, $N={self.parent.N}$, "
        title += f"$J={self.parent.J}$, "
        title += r"$\theta =" + f"{self.parent.theta/np.pi:.3f}" + r"\pi$, "
        title += f"$U={self.parent.U}$"
        return title

    def one_body_density(self, bool_save: bool = False, with_title: bool = True):
        """
        Plot the one-body density pof the ground state.

        Args:
            bool_save (boolean): determines whether to save plot or just show it

        Return:
            None
        """

        x_grid = np.array(range(1, self.parent.L + 1))
        obd = self.parent.one_site_density()

        plt.plot(x_grid, obd, color="black", marker="o")
        plt.bar(x_grid, obd, alpha=0.3, color="cornflowerblue")

        plt.xlabel(r"lattice site $i$")
        plt.ylabel(r"$\langle \hat{b}_i^\dagger \hat{b}_i \rangle$")

        plt.title(self._get_title())

        if bool_save:
            raise NotImplementedError
        else:
            plt.show()
