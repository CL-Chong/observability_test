import observability_aware_control as oac
import numpy as np

mdl = oac.models.multi_quadrotor.MultiQuadrotor(
    3,
    1.0,
    has_baro=False,
    has_odom=True,
)
stlog = oac.algorithms.STLOG(mdl, 2)

res = np.load("demo_optimization_results.npz")

x0 = res["states"][500, :]
u0 = res["inputs"][500, :]
print(
    np.linalg.norm(
        stlog(
            x0,
            u0,
            0.2,
        ),
        ord=-2,
    )
)
