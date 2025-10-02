import numpy as np
from typing import List, Generator


class Basis:
    """
    Construct number state basis for a system defined by L and N.

    Generates the basis states |n_1 n_2 ... n_L> where the total number of
    particles is conserved: sum_i n_i = N

    Example:
        For L = N = 4, basis states include
        |4000>, |3100>, ..., |0004>.

    Args:
        L (int): Number of lattice sites.
        N (int): Number of particles.

    Attributes:
        basis_list (np.ndarray): Array of basis states.
        length (int): Number of basis states.
    """

    def __init__(self, L: int, N: int) -> None:
        """
        Initialize the basis with all possible number states.

        Args:
            L (int): Number of lattice sites.
            N (int): Total particle number.
        """
        self.N = N
        self.L = L

        # Generate all basis states and store as numpy array (reversed order)
        self.basis_list = np.array(list(self._generate_numberstates(L, N))[::-1])
        self.length = self.basis_list.shape[0]


    def _generate_numberstates(self, length: int, total_sum: int
                               ) -> Generator[List[int], None, None]:
        """
        Generate all lists of a given `length` that sum to `total_sum`.

        # source https://stackoverflow.com/questions/7748442/generate-all-possible-lists-of-length-n-that-sum-to-s-in-python

        Args:
            length (int): Length of the list.
            total_sum (int): Target sum of elements.

        Yields:
            List[int]: A list of integers summing to total_sum.
        """
        if length == 1:
            yield [total_sum]
        else:
            for value in range(total_sum + 1):
                for permutation in self._generate_numberstates(
                    length - 1, total_sum - value
                ):
                    yield [value] + permutation
