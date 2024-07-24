import functools

import jax
import jax.numpy as jnp
import sympy as sp

from observability_aware_control.algorithms import common

jax.config.update("jax_enable_x64", True)


def dynamics(x, u):
    return jnp.array(
        [
            jnp.cos(x[2]) * u[0],
            jnp.sin(x[2]) * u[0],
            u[1],
        ]
    )


def sym_dynamics(x, u):
    return sp.Matrix(
        [
            sp.cos(x[2]) * u[0],
            sp.sin(x[2]) * u[0],
            u[1],
        ]
    )


def observation(x, lm):
    rot = jnp.array(
        [
            [jnp.cos(x[2]), jnp.sin(x[2])],
            [-jnp.sin(x[2]), jnp.cos(x[2])],
        ]
    )
    return rot @ (lm - x[0:2])


def sym_observation(x, lm):
    rot = sp.Matrix(
        [
            [sp.cos(x[2]), sp.sin(x[2])],
            [-sp.sin(x[2]), sp.cos(x[2])],
        ]
    )
    return sp.Matrix(rot @ (lm - x[0:2]))


def make_symbolic_lie_derivatives(order):
    sym = {
        "x": sp.MatrixSymbol("x", 3, 1),
        "u": sp.MatrixSymbol("u", 2, 1),
        "lm": sp.MatrixSymbol("lm", 2, 1),
    }

    lfh = sym_observation(sym["x"], sym["lm"])
    expected_lie_derivatives = []
    for _ in range(0, order + 1):
        expected_lie_derivatives.append(sp.lambdify(list(sym.values()), lfh))
        lfh = lfh.jacobian(sym["x"]) @ sym_dynamics(sym["x"], sym["u"])
    return expected_lie_derivatives


ORDER = 5

NUM_TRIALS = 500


def test_lie_derivative():

    expected_lie_derivatives = make_symbolic_lie_derivatives(ORDER)

    result_lie_derivatives = [
        jax.jit(it)
        for it in common.lie_derivative(
            functools.partial(observation, lm=jnp.zeros(2)), dynamics, ORDER
        )
    ]

    key = jax.random.PRNGKey(1000)
    x_key, u_key = jax.random.split(key)
    x_batch = jax.random.uniform(
        x_key,
        (NUM_TRIALS, 3),
        minval=jnp.array([-1.0, -1.0, -jnp.pi]),
        maxval=jnp.array([1.0, 1.0, jnp.pi]),
    )
    u_batch = jax.random.uniform(
        u_key, (NUM_TRIALS, 2), minval=jnp.array([0, -5]), maxval=jnp.array([10, 5])
    )

    assert len(result_lie_derivatives) == len(expected_lie_derivatives) == ORDER + 1

    for result, expected in zip(result_lie_derivatives, expected_lie_derivatives):

        for x, u in zip(x_batch, u_batch):
            result_value = result(x, u)
            expected_value = expected(
                x[..., None], u[..., None], jnp.zeros((2, 1))
            ).ravel()
            assert jnp.allclose(result_value, expected_value)
