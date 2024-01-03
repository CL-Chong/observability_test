import jax
import jax.numpy as jnp
import jax.numpy.linalg as la
from src.observability_aware_control.models import multi_quadrotor
from src.observability_aware_control.algorithms import (
    STLOG,
    OPCCost,
    CooperativeOPCProblem,
    CooperativeLocalizationOptions,
    forward_dynamics,
)

jax.config.update("jax_enable_x64", True)
mdl = multi_quadrotor.MultiQuadrotor(3, 1.0)
order = 1
window = 20
dt = 0.1
stlog = STLOG(mdl, order)
u_lb = jnp.tile(jnp.r_[0.0, -0.4, -0.4, -2.0], (window, mdl.n_robots))
u_ub = jnp.tile(jnp.r_[11.0, 0.4, 0.4, 2.0], (window, mdl.n_robots))
obs_comps = (10, 11, 20, 21)
opts = CooperativeLocalizationOptions(
    window=window,
    id_leader=0,
    lb=u_lb,
    ub=u_ub,
    obs_comps=obs_comps,
    method="trust-constr",
    optim_options={
        "xtol": 1e-5,
        "gtol": 1e-9,
        "disp": False,
        "verbose": 0,
        "maxiter": 150,
    },
    min_v2v_dist=-1,
    max_v2v_dist=-1,
)

min_problem = CooperativeOPCProblem(stlog, opts)
obs_comps = jnp.array(obs_comps)


res = jnp.load("data/optimization_results_ok2.npz")
x = res["states"]
u = res["inputs"]

x0 = x[0, :]
us = u[0:window, :]
# print(x0.shape, us.shape)
i_stlog = (...,) + jnp.ix_(obs_comps, obs_comps)

# Use the .opc method since the full .objective method contains logic separating
# independent from controlled variables
cost, stlog_v = min_problem.opc(us, x0, dt, return_stlog=True)
print(jnp.diagonal(stlog_v, axis1=1, axis2=2))
print(cost)

soln = min_problem.minimize(x0, us, dt)
cost, stlog_v2 = min_problem.opc(soln.x, x0, dt, return_stlog=True)
print(jnp.diagonal(stlog_v2, axis1=1, axis2=2))
print(cost)
