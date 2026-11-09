import numpy as np
from scipy.linalg import solve_banded

from .transform import payoff, far_field, TAU_MAX, XMIN, XMAX


def crank_nicolson(N, M):

    """ Solves the heat equation u_tau = u_xx with Crank-Nicolson.

    Parameters
    -----------
    N : int, number of grid intervals in x.
    M : int, number of time steps.

    Returns
    --------
    x : array, the grid.
    u : array, the solution at TAU_MAX.
    M : int, number of time steps.

    Notes
    -----
    Implicit, so it is stable for any dt. Each step solves one tridiagonal
    system. u = 0 on the left and far_field on the right.
    """

    x = np.linspace(XMIN, XMAX, N + 1)
    dx = x[1] - x[0]
    dt = TAU_MAX / M
    lam = dt / dx**2

    ab = np.zeros((3, N - 1))
    ab[0, 1:] = -lam / 2
    ab[1, :] = 1 + lam
    ab[2, :-1] = -lam / 2

    u = payoff(x)
    for n in range(M):
        rhs = u[1:-1] + lam / 2 * (u[2:] - 2 * u[1:-1] + u[:-2])
        right = far_field(x[-1], (n + 1) * dt)
        rhs[-1] += lam / 2 * right
        u = np.concatenate([[0.0], solve_banded((1, 1), ab, rhs), [right]])

    return x, u, M
