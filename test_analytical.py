"""Quick simulator/controller sanity checks.

This script runs one of:
  - open-loop (u=0)
  - PID feedback
  - finite-horizon discrete LQR

and then produces an animation and time-series plots.
"""
import argparse
import math
import numpy as np
import torch
from env import InvertedPendulum
from controllers import FiniteHorizonLQRController, PIDController
from utils import animate_pendulum, plot_time_series, wrap_angle
torch.set_default_dtype(torch.float64)


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", type=str, required=True, choices=["free", "pid", "lqr"])
    ap.add_argument("--init_theta", type=float, default=math.pi+0.4)
    ap.add_argument("--init_dtheta", type=float, default=0.0)
    ap.add_argument("--total_time", type=float, default=20.0)
    args = ap.parse_args()

    # Select which controller to run.
    controller_type = args.kind

    # Environment and rollout horizon.
    dt = 0.02 
    env = InvertedPendulum(m=1.0, l=1.0, g=9.81, b=0.5, u_max=10.0, dt=dt)
    x0 = [args.init_theta, args.init_dtheta] # initial state: [theta, dtheta]
    T = args.total_time # simulation time
    N = int(T/dt) # number of horizon to simulate
    
    # Instantiate the requested controller.
    if controller_type == "free":
        title = "Open-Loop Inverted Pendulum Model Test (u = 0)" 
    elif controller_type == "pid":
        kp = 25.0 
        ki = 1.0 
        kd = 5.0
        controller = PIDController(dt=env.dt, u_max=env.u_max, kp=kp, ki=ki, kd=kd)
        title = "PID Controller"
    elif controller_type == "lqr":
        Q = torch.diag(torch.tensor([25.0, 1.0]))
        R = torch.tensor([[0.5]])
        Qf = 10.0 * Q
        controller = FiniteHorizonLQRController(env, N=N, Q=Q, R=R, Qf=Qf)
        title = "Finite-Horizon Discrete LQR"
    else:
        raise ValueError("controller_type must be:'\{lqr','pid','free'\}.")

    # Run the simulation.
    if controller_type == "free":
        xs, us = env.simulate_open_loop(x0=x0, N=N)
    else:
        xs, us = env.rollout(controller, x0=x0, N=N)

    # Visualization.
    animate_pendulum(xs, us, dt=env.dt, l=env.l, u_max=env.u_max, title=title)

    # ---- Plot time series ----
    if controller_type == "lqr":
        gains_dict={
            "Q": controller.Q.tolist(),  # or store Q explicitly
            "R": controller.R.tolist(),
            "Qf": controller.Qf.tolist()
        }
    elif controller_type == "pid":
        gains_dict={
            "kp": controller.kp, 
            "ki": controller.ki,
            "kd": controller.kd
        }
    else:
        gains_dict=None

    plot_time_series(
        xs, us, dt=env.dt, u_max=env.u_max,
        x0=x0,
        controller_type=controller_type,
        gains_dict=gains_dict
    )



if __name__ == "__main__":
    main()



