import numpy as np

from .transform import payoff, far_field, TAU_MAX, XMIN, XMAX


def lbm(N, tau_r=1.0):

    """ Solves the heat equation u_tau = u_xx with a D1Q3 lattice Boltzmann
    scheme (BGK collision).

    Parameters
    -----------
    N : int, number of grid intervals in x.
    tau_r : float, starting relaxation time. It is adjusted slightly so the
        number of steps is a whole number and the diffusion coefficient is
        exactly 1.

    Returns
    --------
    x : array, the grid.
    u : array, the solution at TAU_MAX.
    M : int, number of time steps.

    Notes
    -----
    Velocities -1, 0, +1 with weights 1/6, 2/3, 1/6, so c_s^2 = 1/3 and the
    diffusion coefficient is c_s^2 (tau_r - 1/2) dx^2 / dt. The scheme is
    explicit, so dt has to scale with dx^2.
    Boundaries use anti-bounce-back with the wall half a cell outside the
    last node: u = 0 on the left, far_field on the right.
    """

    x = np.linspace(XMIN, XMAX, N + 1)
    dx = x[1] - x[0]
    M = int(np.ceil(TAU_MAX / ((tau_r - 0.5) * dx**2 / 3)))
    dt = TAU_MAX / M
    tau_r = 0.5 + 3 * dt / dx**2

    w = np.array([1 / 6, 2 / 3, 1 / 6])
    f = w[:, None] * payoff(x)[None, :]

    for n in range(M):
        u = f.sum(axis=0)
        fs = f - (f - w[:, None] * u) / tau_r

        f = np.empty_like(fs)
        f[1] = fs[1]
        f[2, 1:] = fs[2, :-1]
        f[0, :-1] = fs[0, 1:]

        t_wall = (n + 0.5) * dt
        f[2, 0] = -fs[0, 0]
        f[0, -1] = -fs[2, -1] + 2 * w[0] * far_field(x[-1] + dx / 2, t_wall)

    return x, f.sum(axis=0), M
