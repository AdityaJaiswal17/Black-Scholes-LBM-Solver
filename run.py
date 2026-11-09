from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from bslbm import bs_call, lbm, crank_nicolson, to_price, rmse

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

out = Path("results")
out.mkdir(exist_ok=True)

print(f"closed-form ATM call: {float(bs_call(100.0)):.4f}")

rows = []
for N in (80, 160, 320, 640, 1280):
    x, u, m_lbm = lbm(N)
    xc, uc, m_cn = crank_nicolson(N, max(4, N // 8))
    rows.append({"N": N, "lbm_rmse": rmse(x, u), "lbm_steps": m_lbm,
                 "cn_rmse": rmse(xc, uc), "cn_steps": m_cn})
table = pd.DataFrame(rows)
print(table.to_string(index=False, float_format="%.2e"))
table.to_csv(out / "convergence.csv", index=False, float_format="%.4e")

x, u, _ = lbm(640)
S, V = to_price(x, u)
print(f"LBM price at S = 100 (N = 640): {np.interp(100.0, S, V):.4f}")

fig, ax = plt.subplots(figsize=(6, 4.5))
ax.loglog(table["N"], table["lbm_rmse"], "o-", label="lattice Boltzmann")
ax.loglog(table["N"], table["cn_rmse"], "s-", label="Crank-Nicolson")
ref = table["lbm_rmse"].iloc[0] * (table["N"].iloc[0] / table["N"]) ** 2
ax.loglog(table["N"], ref, "k--", lw=0.8, label="slope -2")
ax.set_xlabel("grid intervals N")
ax.set_ylabel("RMSE vs closed form, S in [50, 150]")
ax.legend()
fig.tight_layout()
fig.savefig(out / "convergence.png", dpi=150)
print("saved to", out)
