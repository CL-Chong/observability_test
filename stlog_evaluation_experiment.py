from math import sqrt
import tqdm
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from observability_aware_control.models import multi_quadrotor
from observability_aware_control.algorithms import (
    STLOG,
    CooperativeOPCProblem,
    CooperativeLocalizationOptions,
)
import observability_aware_control.algorithms.misc.trajectory_generation as planning
import observability_aware_control.algorithms.misc.simple_ekf as ekf


def run_state_est(kf, xs, us, dt, cov_op_init):
    cov_num = []
    cov_op = cov_op_init
    for x, u in zip(xs, us):
        x_pred, cov_pred = kf.predict(x, cov_op, u, dt)
        x, cov_op = kf.update(x_pred, cov_pred, kf.hfcn(x_pred))
        cov_num.append(jnp.array(cov_op))
    return jnp.stack(cov_num)


mdl = multi_quadrotor.MultiQuadrotor(3, 1.0)
order = 1
window = 20
dt = 0.1
stlog = STLOG(mdl, order)
u_lb = jnp.tile(jnp.r_[0.0, -0.4, -0.4, -2.0], (window, mdl.n_robots))
u_ub = jnp.tile(jnp.r_[11.0, 0.4, 0.4, 2.0], (window, mdl.n_robots))
obs_comps = (10, 11, 12, 20, 21, 22)
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
        "verbose": 1,
        "maxiter": 500,
    },
    min_v2v_dist=sqrt(0.2),
    max_v2v_dist=sqrt(10),
)

min_problem = CooperativeOPCProblem(stlog, opts)
obs_comps = jnp.array(obs_comps)

speed = 2.5
n_steps = 500
n_splits = 2
dist = jnp.r_[n_steps * speed * dt, 0, 0]
p_refs = [
    jnp.linspace(p0, p0 + dist, n_splits, axis=1)
    for p0 in [
        jnp.r_[2.0, 1e-3, 10.0],
        jnp.r_[-1e-3, 1.0, 10.0],
        jnp.r_[2e-4, -1.0, 10.0],
    ]
]
t_ref = (jnp.linspace(0, n_steps * dt, n_splits),) * 3
t_sample = jnp.r_[0:n_steps] * dt
ms = planning.MinimumSnap(5, [0, 0, 1, 1], planning.MinimumSnapAlgorithm.CONSTRAINED)

states_traj, inputs_traj = ms.generate_trajectories(t_ref, p_refs, t_sample)
x0 = states_traj[:, :, 100].ravel()
us = (
    inputs_traj[:, :, 100 : 100 + window]
    .swapaxes(0, 1)
    .reshape(-1, window, order="F")
    .T
)


i_stlog = (...,) + jnp.ix_(obs_comps, obs_comps)
kf = ekf.SimpleEKF(
    lambda x, u, dt: x + dt * mdl.dynamics(x, u),
    mdl.observation,
    jnp.diag(jnp.tile(jnp.r_[0.1, jnp.full(3, 0.01)], mdl.n_robots)),
    jnp.diag(
        jnp.r_[
            jnp.full(mdl.DIM_LEADER_POS_OBS, 1e-3),
            jnp.full(mdl.DIM_ALT_OBS * (mdl.n_robots - 1), 1e-3),
            jnp.full(mdl.DIM_ATT_OBS * mdl.n_robots, 1e-3),
            jnp.full(mdl.DIM_BRNG_OBS * (mdl.n_robots - 1), 1e-3),
        ]
    ),
)

# Use the .opc method since the full .objective method contains logic separating
# independent from controlled variables
cost = {}
stlog = {}
xs = {}
cov = {}
cost["init"], stlog["init"], xs["init"] = min_problem.opc(
    us, x0, dt, return_stlog=True, return_traj=True
)
cov["init"] = run_state_est(kf, xs["init"], us, dt, 0.2 * jnp.eye(30))[i_stlog]

key = jax.random.PRNGKey(1000)
n_rand = 200
us_rand = jax.random.uniform(
    key,
    shape=(n_rand,) + u_lb.shape,
    minval=jnp.stack([u_lb] * n_rand),
    maxval=jnp.stack([u_ub] * n_rand),
)
for idx in tqdm.trange(n_rand):
    us = us_rand[idx, :, :]
    cost[f"rand{idx}"], stlog[f"rand{idx}"], xs[f"rand{idx}"] = min_problem.opc(
        us, x0, dt, return_stlog=True, return_traj=True
    )
    cov[f"rand{idx}"] = run_state_est(kf, xs[f"rand{idx}"], us, dt, 0.2 * jnp.eye(30))[
        i_stlog
    ]


soln = min_problem.minimize(x0, us, dt)
us = soln.x
cost["opt"], stlog["opt"], xs["opt"] = min_problem.opc(
    us, x0, dt, return_stlog=True, return_traj=True
)
cov["opt"] = run_state_est(kf, xs["opt"], us, dt, 0.2 * jnp.eye(30))[i_stlog]

stddev = {k: 3 * jnp.sqrt(jnp.diagonal(v, axis1=1, axis2=2)) for k, v in cov.items()}

fig, axs = plt.subplots(2, 3)
print(len(axs))
time = jnp.r_[0:window] * dt
for idv in range(2):
    for idx, it in enumerate("xyz"):
        axs[idv, idx].plot(
            time,
            stddev["init"][:, idv * 3 + idx],
            "--",
            color="r",
            label=f"Initial v{idv}, axis {idx}",
            alpha=0.5,
            linewidth=2,
        )
        axs[idv, idx].plot(
            time,
            stddev["opt"][:, idv * 3 + idx],
            color="b",
            label=f"Optimized v{idv}, axis {idx}",
            linewidth=2,
        )

        for idr in range(50):
            axs[idv, idx].plot(
                time,
                stddev[f"rand{idr}"][:, idv * 3 + idx],
                linewidth=0.5,
                alpha=0.5,
            )
        axs[idv, idx].legend()
        axs[idv, idx].set_xlabel("Time (s)")
        axs[idv, idx].set_ylabel("Position Standard Deviation (m)")
        axs[idv, idx].set_title(f"Vehicle {idv}, {it}-axis")
plt.show()
