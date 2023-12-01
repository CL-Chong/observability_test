import inspect
import time
import timeit

import casadi as cs
import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt

from src.observability_aware_control import algorithms
from src.observability_aware_control.models import autodiff, symbolic

jax.config.update("jax_enable_x64", True)

T = 0.01

smdl = symbolic.multi_planar_robot.LeaderFollowerRobots(3)
amdl = autodiff.multi_planar_robot.LeaderFollowerRobots(3)


def x_op():
    return


def u_op():
    return jax.random.uniform(key, [smdl.nu])


x_sym = cs.MX.sym("x", smdl.nx)
u_sym = cs.MX.sym("u", smdl.nu)

order = 3

sym_builder = algorithms.symbolic.STLOG(
    smdl, order, algorithms.symbolic.STLOGOptions(window=5)
)
v_sym = sym_builder.objective(t_fix=0.2)


num_builder = algorithms.autodiff.STLOG(
    amdl, order, algorithms.autodiff.STLOGOptions(window=5, input_axis=1)
)
v_num = num_builder.objective()

# subkey = jax.random.split(key, 100)
rng = np.random.default_rng(114514)
x = rng.uniform(size=smdl.nx)
u = rng.uniform(size=(smdl.nu, 5))

t1 = time.perf_counter()
_ = jax.grad(v_num)(u, x, 0.2)
t2 = time.perf_counter()
# Run the operations to be profiled
print(f"JIT in {t2 - t1}s")
exit(0)

sig = inspect.signature(np.allclose)
for _ in range(100):
    x = rng.uniform(size=smdl.nx)
    u = rng.uniform(size=(smdl.nu, 5))
    npt.assert_allclose(
        v_num(u, x, t),
        v_sym(u, x),
        rtol=sig.parameters["rtol"].default,
        atol=sig.parameters["atol"].default,
    )

print("JIT done")
timer = timeit.Timer(
    "v_num(rng.uniform(size=(smdl.nu, 5)), rng.uniform(size=smdl.nx))",
    globals=locals(),
)
reps = 5000
timings = np.asarray(timer.repeat(number=reps)) / reps
print(f"Autodiff - mean {timings.mean():.8f} std {timings.std():.8f}")
timer = timeit.Timer(
    "v_sym(rng.uniform(size=(smdl.nu, 5)), rng.uniform(size=smdl.nx))",
    globals=locals(),
)
timings = np.asarray(timer.repeat(number=reps)) / reps
print(f"Symbolic - mean {timings.mean():.8f} std {timings.std():.8f}")
