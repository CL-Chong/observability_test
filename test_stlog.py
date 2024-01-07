import numpy as np

import observability_aware_control as oac

mdl = oac.models.multi_quadrotor.MultiQuadrotor(
    3,
    1.0,
    has_baro=False,
    has_odom=True,
)
stlog = oac.algorithms.STLOG(mdl, 2)
# obs_comps = np.concatenate(
#     (
#         # range(0, 3),
#         # range(4, 10),
#         range(10, 13),
#         # range(14, 20),
#         range(20, 23),
#         # range(24, 30),
#     )
# )
obs_comps = range(0, mdl.nx)

print(f"states = {mdl.nx}, obs = {mdl.ny}")

res = np.load("demo_optimization_results.npz")
idx_sim = 500

x0 = res["states"][idx_sim, :]
u0 = res["inputs"][idx_sim, :]
print(u0)
# u0 = np.array([1] * mdl.nu)

dt_stlog = 0.2
mat = stlog(
    x0,
    u0,
    dt_stlog,
)
# [np.ix_(obs_comps, obs_comps)]

eigenValues, eigenVectors = np.linalg.eig(mat)

idx_sort = eigenValues.argsort()[::-1]
eigenValues = eigenValues[idx_sort]
eigenVectors = np.round(eigenVectors[:, idx_sort], 2)


print(eigenValues[-3:], eigenVectors[-3:])
