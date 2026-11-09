import numpy as np

from bslbm.transform import (bs_call, payoff, far_field, to_price, K, r, T,
                             XMAX)


class TestTransform:

    def test_atm_price(self):
        """Textbook value for S = K = 100, r = 5%, sigma = 20%, T = 1."""
        assert abs(bs_call(100.0) - 10.4506) < 1e-4

    def test_payoff_maps_back(self):
        """At tau = 0 the transformed payoff turns back into max(S - K, 0)."""
        x = np.linspace(-1, 1, 41)
        S, V = to_price(x, payoff(x), tau=0.0)
        assert np.allclose(V, np.maximum(S - K, 0.0))

    def test_far_field_is_discounted_forward(self):
        """The right boundary is S - K exp(-r T) at the start date."""
        tau = 0.02
        S, V = to_price(np.array([XMAX]), far_field(XMAX, tau), tau=tau)
        assert np.isclose(V[0], S[0] - K * np.exp(-r * T))
