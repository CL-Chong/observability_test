import numpy as np
import jax.numpy as jnp
import jax
import observability_aware_control as oac

jax.config.update("jax_enable_x64", True)
mdl = oac.models.multi_quadrotor.MultiQuadrotor(
    3,
    1.0,
    has_baro=False,
    has_odom=True,
)
stlog = oac.algorithms.STLOG(mdl, 2)

obs_comps = [10, 11, 12, 20, 21, 22]


res = np.load("demo_optimization_results.npz")
idx_sim = 500

x0 = res["states"][idx_sim, :]
u0 = res["inputs"][idx_sim, :]
# u0 = np.array([1] * mdl.nu)

dt_stlog = 0.2
mat = stlog(x0, u0, dt_stlog)
# [np.ix_(obs_comps, obs_comps)]

print(jnp.linalg.norm(mat, -2))
print(jnp.linalg.norm(mat[np.ix_(obs_comps, obs_comps)], -2))

jnp.set_printoptions(precision=3, suppress=True, linewidth=999)
print(mat)
