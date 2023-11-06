import jax.numpy as jnp
import numpy as np
import numpy.testing as npt

import models.autodiff as ad_models
import observability_aware_control.algorithms.autodiff as ad_algorithms
from models import models
from observability_aware_control import algorithms

DT = 0.01
EPS = 1e-2
N_STEPS = 50
RNG = np.random.default_rng(seed=114514)

NUM_TRIALS = 10

params = {
    "num": {
        "sys": models.MultiRobot(n_robots=3),
        "dt": np.ones(N_STEPS, dtype=np.float32) * DT,
    },
    "ad": {
        "sys": ad_models.MultiRobot(n_robots=3),
        "dt": jnp.ones(N_STEPS) * DT,
    },
}


def generate_ic_and_controls():
    return {
        "x0": RNG.uniform(-1, 1, params["num"]["sys"].nx),
        "u": RNG.uniform(-1, 1, (params["num"]["sys"].nu, N_STEPS)),
    }


def test_numsolve():
    for i_trial in range(NUM_TRIALS):
        rand_vals = generate_ic_and_controls()
        params["num"].update(rand_vals)
        params["ad"].update({k: jnp.asarray(v) for k, v in rand_vals.items()})

        num_results = algorithms.numsolve_sigma(**params["num"])
        ad_results = ad_algorithms.numsolve_sigma(**params["ad"])
        for lhs, rhs, key in zip(num_results, ad_results, ("state", "observation")):
            npt.assert_allclose(
                lhs,
                rhs,
                rtol=1e-4,
                atol=1e-3,
                err_msg=f"Failure on {i_trial} : {key} trajectory",
            )


def test_numlog():
    for i_trial in range(NUM_TRIALS):
        if i_trial == 0:
            axes = None
        else:
            nx = params["num"]["sys"].nx
            n_axes = RNG.integers(2, nx - 1)
            axes = RNG.choice(np.arange(1, nx), n_axes, replace=False)
        rand_vals = generate_ic_and_controls()
        params["num"].update(rand_vals)
        params["ad"].update({k: jnp.asarray(v) for k, v in rand_vals.items()})

        num_results = algorithms.numlog(
            **params["num"], eps=np.float32(EPS), perturb_axis=axes
        )
        ad_results = ad_algorithms.numlog(
            **params["ad"], eps=np.float32(EPS), perturb_axis=axes
        )
        npt.assert_allclose(
            num_results,
            ad_results,
            rtol=1e-4,
            atol=1e-3,
            err_msg=f"Failure on {i_trial}",
        )
