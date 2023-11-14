import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
from internal import algorithms, models

import observability_aware_control.algorithms.autodiff as ad_algorithms
import observability_aware_control.models.autodiff as ad_models

DT = 0.01
EPS = 1e-2
N_STEPS = 50
RNG = np.random.default_rng(seed=114514)

NUM_TRIALS = 10

params = {
    "num": {
        "sys": models.LeaderFollowerRobots(n_robots=3),
        "dt": np.ones(N_STEPS, dtype=np.float32) * DT,
    },
    "ad": {
        "sys": ad_models.multi_planar_robot.LeaderFollowerRobots(n_robots=3),
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
        ad_results = ad_algorithms.numsolve_sigma(**params["ad"], axis=1)
        for lhs, rhs, key in zip(num_results, ad_results, ("state", "observation")):
            npt.assert_allclose(
                lhs,
                rhs,
                rtol=1e-4,
                atol=1e-3,
                err_msg=f"Failure on {i_trial} : {key} trajectory",
            )


def test_numsolve_leader_follower_vs_reference_sensing_ad():
    leader_follower = ad_models.multi_planar_robot.LeaderFollowerRobots(n_robots=3)
    leader = ad_models.planar_robot.PlanarRobot()
    followers = ad_models.multi_planar_robot.ReferenceSensingRobots(n_robots=2)

    for _ in range(NUM_TRIALS):
        rand_vals = generate_ic_and_controls()
        params["num"].update(rand_vals)
        params["ad"].update({k: jnp.asarray(v) for k, v in rand_vals.items()})

        combined_x, combined_y = ad_algorithms.numsolve_sigma(
            leader_follower,
            params["ad"]["x0"],
            params["ad"]["u"],
            params["ad"]["dt"],
            axis=1,
        )
        x0_leader, x0_follower = jnp.split(params["ad"]["x0"], (leader.nx,))
        u_leader, u_follower = jnp.split(params["ad"]["u"], (leader.nu,))
        leader_x = ad_algorithms.numsolve_sigma(
            leader,
            x0_leader,
            u_leader,
            params["ad"]["dt"],
            without_observation=True,
            axis=1,
        )

        follower_x, follower_y = ad_algorithms.numsolve_sigma(
            followers,
            x0_follower,
            u_follower,
            params["ad"]["dt"],
            h_args=leader_x[0:2, :],
            axis=1,
        )
        npt.assert_allclose(combined_x, jnp.vstack([leader_x, follower_x]), rtol=1e-5)
        npt.assert_allclose(combined_y[1:, :], follower_y, rtol=1e-5)


def test_numsolve_leader_follower_vs_reference_sensing_conventional():
    leader_follower = models.LeaderFollowerRobots(n_robots=3)
    leader = models.Robot()
    followers = models.ReferenceSensingRobots(n_robots=2)

    for _ in range(NUM_TRIALS):
        rand_vals = generate_ic_and_controls()
        params["num"].update(rand_vals)
        params["ad"].update({k: np.asarray(v) for k, v in rand_vals.items()})

        combined_x, combined_y = algorithms.numsolve_sigma(
            leader_follower, params["ad"]["x0"], params["ad"]["u"], params["ad"]["dt"]
        )
        x0_leader, x0_follower = np.split(params["ad"]["x0"], (leader.nx,))
        u_leader, u_follower = np.split(params["ad"]["u"], (leader.nu,))
        leader_x = algorithms.numsolve_sigma(
            leader,
            x0_leader,
            u_leader,
            params["ad"]["dt"],
            without_observation=True,
        )

        follower_x, follower_y = algorithms.numsolve_sigma(
            followers,
            x0_follower,
            u_follower,
            params["ad"]["dt"],
            h_args=leader_x[0:2, :],
        )
        npt.assert_allclose(combined_x, np.vstack([leader_x, follower_x]), rtol=1e-5)
        npt.assert_allclose(combined_y[1:, :], follower_y, rtol=1e-5)


def test_numlog_leader_follower_vs_reference_sensing_ad():
    leader_follower = ad_models.multi_planar_robot.LeaderFollowerRobots(n_robots=3)
    leader = ad_models.planar_robot.PlanarRobot()
    followers = ad_models.multi_planar_robot.ReferenceSensingRobots(n_robots=2)

    for _ in range(NUM_TRIALS):
        rand_vals = generate_ic_and_controls()
        params["num"].update(rand_vals)
        params["ad"].update({k: jnp.asarray(v) for k, v in rand_vals.items()})

        expected = ad_algorithms.numlog(
            leader_follower,
            params["ad"]["x0"],
            params["ad"]["u"],
            params["ad"]["dt"],
            eps=EPS,
            perturb_axis=[3, 4, 6, 7],
            axis=1,
        )
        x0_leader, x0_follower = jnp.split(params["ad"]["x0"], (leader.nx,))
        u_leader, u_follower = jnp.split(params["ad"]["u"], (leader.nu,))
        leader_x = ad_algorithms.numsolve_sigma(
            leader,
            x0_leader,
            u_leader,
            params["ad"]["dt"],
            without_observation=True,
            axis=1,
        )

        result = ad_algorithms.numlog(
            followers,
            x0_follower,
            u_follower,
            params["ad"]["dt"],
            eps=EPS,
            perturb_axis=[0, 1, 3, 4],  # Note leader is no longer part of the state
            h_args=leader_x[0:2, :],
            axis=1,
        )
        npt.assert_allclose(expected, result, rtol=1e-4, atol=1e-3)


def test_numlog_leader_follower_vs_reference_sensing_conventional():
    leader_follower = models.LeaderFollowerRobots(n_robots=3)
    leader = models.Robot()
    followers = models.ReferenceSensingRobots(n_robots=2)

    for _ in range(NUM_TRIALS):
        rand_vals = generate_ic_and_controls()
        params["num"].update(rand_vals)
        params["ad"].update({k: np.asarray(v) for k, v in rand_vals.items()})

        expected = algorithms.numlog(
            leader_follower,
            params["ad"]["x0"],
            params["ad"]["u"],
            params["ad"]["dt"],
            eps=EPS,
            perturb_axis=[3, 4, 6, 7],
        )
        x0_leader, x0_follower = np.split(params["ad"]["x0"], (leader.nx,))
        u_leader, u_follower = np.split(params["ad"]["u"], (leader.nu,))
        leader_x = algorithms.numsolve_sigma(
            leader,
            x0_leader,
            u_leader,
            params["ad"]["dt"],
            without_observation=True,
        )

        result = algorithms.numlog(
            followers,
            x0_follower,
            u_follower,
            params["ad"]["dt"],
            eps=EPS,
            perturb_axis=[0, 1, 3, 4],  # Note leader is no longer part of the state
            h_args=leader_x[0:2, :],
        )
        npt.assert_allclose(expected, result, rtol=1e-4, atol=1e-3)


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
            **params["ad"], eps=np.float32(EPS), perturb_axis=axes, axis=1
        )
        npt.assert_allclose(
            num_results,
            ad_results,
            rtol=1e-4,
            atol=1e-3,
            err_msg=f"Failure on {i_trial}",
        )
