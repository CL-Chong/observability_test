import jax
import jax.numpy as jnp
import tqdm


def gaussian_noise(key, cov, shape):
    noise = jax.random.multivariate_normal(key, jnp.zeros(cov.shape[0]), cov, shape)
    return jnp.diagonal(noise, axis1=1, axis2=2)


@jax.tree_util.Partial(
    jax.jit,
    static_argnums=[0],
    static_argnames=["simulate_input_noise", "simulate_sensor_noise", "disable_update"],
)
def run_state_est(
    kf,
    x_init,
    us,
    ys,
    ts,
    cov_init,
    key,
    *,
    simulate_input_noise,
    simulate_sensor_noise,
    disable_update,
):
    f_key, h_key = jax.random.split(key)

    def ekf_update(x_tup, u_tup):
        x_op, cov_op = x_tup
        u, y, dt_op = u_tup

        x_pred, cov_pred = kf.predict(x_op, cov_op, u, dt_op)
        if disable_update:
            res = x_pred, cov_pred
        else:
            res = kf.update(x_pred, cov_pred, y)

        return res, res

    xs_tup = (x_init, cov_init)
    if simulate_input_noise:
        us += gaussian_noise(f_key, kf.in_cov, us.shape) / 4
    if simulate_sensor_noise:
        ys += gaussian_noise(h_key, kf.obs_cov, ys.shape)
    us_tup = (us, ys, jnp.gradient(ts))
    _, (x_hist, cov_hist) = jax.lax.scan(ekf_update, xs_tup, us_tup)

    return x_hist, cov_hist


def rms(data):
    return jnp.sqrt((data**2).mean())


def evaluate_state_estimation(
    kf,
    states,
    inputs,
    time,
    init_cov,
    mdl,
    keys,
    *,
    simulate_input_noise=True,
    simulate_sensor_noise=True,
    disable_update=False,
):
    x_errs = []
    cov_hists = []

    for seed in tqdm.tqdm(keys):
        observations = jax.vmap(kf.hfcn)(states)
        x_hist, cov_hist = run_state_est(
            kf,
            states[0, ...],
            inputs,
            observations,
            time,
            init_cov,
            seed,
            simulate_input_noise=simulate_input_noise,
            simulate_sensor_noise=simulate_sensor_noise,
            disable_update=disable_update,
        )

        x_err = x_hist - states
        x_err = x_err.reshape(x_err.shape[0], mdl.n_robots, -1)[:, :, 0:3]
        x_errs.append(x_err)

        cov_hist = 3 * jnp.sqrt(jnp.diagonal(cov_hist, axis1=1, axis2=2))
        cov_hist = cov_hist.reshape(cov_hist.shape[0], mdl.n_robots, -1)[:, :, 0:3]
        cov_hists.append(cov_hist)

    x_err = jnp.array(x_errs).mean(axis=0)
    cov_hist = jnp.array(cov_hists).mean(axis=0)
    return time, cov_hist, x_err
