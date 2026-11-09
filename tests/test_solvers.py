from bslbm import lbm, crank_nicolson, rmse


class TestSolvers:

    def test_lbm_matches_closed_form(self):
        x, u, _ = lbm(320)
        assert rmse(x, u) < 1e-3

    def test_cn_matches_closed_form(self):
        x, u, _ = crank_nicolson(320, 40)
        assert rmse(x, u) < 1e-3

    def test_lbm_second_order(self):
        """Halving dx should cut the error by about 4."""
        x1, u1, _ = lbm(160)
        x2, u2, _ = lbm(320)
        assert rmse(x1, u1) / rmse(x2, u2) > 3.5

    def test_lbm_steps_scale_with_dx_squared(self):
        """Explicit scheme: doubling N needs about four times the steps."""
        _, _, m1 = lbm(160)
        _, _, m2 = lbm(320)
        assert 3.5 < m2 / m1 < 4.5
