"""Collect imitation-learning data from the existing finite-horizon LQR.

Outputs:
  - data/lqr_dataset.npz
      states:  (num_traj, N, 2)   [theta, theta_dot]
      actions: (num_traj, N)      LQR torque
      x0s:     (num_traj, 2)
      dt: float
      N: int
  - data/images/traj_XX/frame_YYYY.png  (RGB renderings, size=64)

Initial state distribution:
  theta0 ~ Uniform[pi ± 0.5]
  theta_dot0 ~ Uniform[±0.2]

DO NOT need to modify this script.
"""

import os
import math
import json
import numpy as np
import torch

from env import InvertedPendulum
from controllers import FiniteHorizonLQRController
from utils import render_pendulum_frame


def main():
    # ----- Config -----
    seed = 0
    num_traj = 100
    dt = 0.02
    T = 10.0 
    N = int(T / dt)

    theta_range = 0.5
    dtheta_range = 0.2

    img_size = 64
    save_images = True

    out_dir = "data"
    img_dir = os.path.join(out_dir, "images")
    os.makedirs(out_dir, exist_ok=True)
    if save_images:
        os.makedirs(img_dir, exist_ok=True)

    # ----- Reproducibility -----
    np.random.seed(seed)
    torch.manual_seed(seed)

    # ----- Environment + LQR controller -----
    env = InvertedPendulum(m=1.0, l=1.0, g=9.81, b=0.5, u_max=10.0, dt=dt)
    Q = torch.diag(torch.tensor([25.0, 1.0]))
    R = torch.tensor([[0.5]])
    Qf = 10.0 * Q
    lqr = FiniteHorizonLQRController(env, N=N, Q=Q, R=R, Qf=Qf)

    # ----- Storage -----
    states = np.zeros((num_traj, N, 2), dtype=np.float32)
    actions = np.zeros((num_traj, N), dtype=np.float32)
    x0s = np.zeros((num_traj, 2), dtype=np.float32)

    meta = {
        "seed": seed,
        "num_traj": num_traj,
        "dt": dt,
        "T": T,
        "N": N,
        "theta_init_range": [float(math.pi - theta_range), float(math.pi + theta_range)],
        "dtheta_init_range": [-dtheta_range, dtheta_range],
        "img_size": img_size,
        "env": {"m": env.m, "l": env.l, "g": env.g, "b": env.b, "u_max": env.u_max},
        "lqr": {"Q": Q.tolist(), "R": R.tolist(), "Qf": Qf.tolist()},
    }

    print("Generating data...")
    # ----- Rollouts -----
    for i in range(num_traj):
        theta0 = np.random.uniform(math.pi - theta_range, math.pi + theta_range)
        dtheta0 = np.random.uniform(-dtheta_range, dtheta_range)
        x0 = [theta0, dtheta0]
        # x0 = torch.tensor([theta0, theta_dot0], dtype=torch.float64)
        x0s[i] = np.asarray(x0, dtype=np.float32)

        xs, us = env.rollout(lqr, x0=x0, N=N)  # xs: (N+1,2), us: (N,)
        xs = xs[:-1]  # align with actions (N)

        states[i] = xs.detach().cpu().numpy().astype(np.float32)
        actions[i] = us.detach().cpu().numpy().astype(np.float32)

        if save_images:
            traj_dir = os.path.join(img_dir, f"traj_{i:02d}")
            os.makedirs(traj_dir, exist_ok=True)
            for t in range(N):
                th = float(states[i, t, 0])
                u = float(actions[i, t])
                img = render_pendulum_frame(th, u=0.0, l=env.l, u_max=env.u_max, size=img_size) # u must be 0
                # Save as png via matplotlib
                # Using matplotlib avoids extra deps like PIL
                import matplotlib.pyplot as plt
                plt.imsave(os.path.join(traj_dir, f"frame_{t:04d}.png"), img)

    # ----- Save dataset -----
    np.savez_compressed(
        os.path.join(out_dir, "lqr_dataset.npz"),
        states=states,
        actions=actions,
        x0s=x0s,
        dt=np.float32(dt),
        N=np.int32(N),
        meta_json=json.dumps(meta),
    )

    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("\nSaved dataset to data/lqr_dataset.npz")
    if save_images:
        print("Saved images under data/images/")


if __name__ == "__main__":
    main()
"""Collect imitation-learning data from the existing finite-horizon LQR.

Outputs:
  - data/lqr_dataset.npz
      states:  (num_traj, N, 2)   [theta, theta_dot]
      actions: (num_traj, N)      LQR torque
      x0s:     (num_traj, 2)
      dt: float
      N: int
  - data/images/traj_XX/frame_YYYY.png  (RGB renderings, size=64)

Initial state distribution:
  theta0 ~ Uniform[pi ± 0.5]
  theta_dot0 ~ Uniform[±0.2]

DO NOT need to modify this script.
"""

import os
import math
import json
import numpy as np
import torch

from env import InvertedPendulum
from controllers import FiniteHorizonLQRController
from utils import render_pendulum_frame


def main():
    # ----- Config -----
    seed = 0
    num_traj = 100
    dt = 0.02
    T = 10.0 
    N = int(T / dt)

    theta_range = 0.5
    dtheta_range = 0.2

    img_size = 64
    save_images = True

    out_dir = "data"
    img_dir = os.path.join(out_dir, "images")
    os.makedirs(out_dir, exist_ok=True)
    if save_images:
        os.makedirs(img_dir, exist_ok=True)

    # ----- Reproducibility -----
    np.random.seed(seed)
    torch.manual_seed(seed)

    # ----- Environment + LQR controller -----
    env = InvertedPendulum(m=1.0, l=1.0, g=9.81, b=0.5, u_max=10.0, dt=dt)
    Q = torch.diag(torch.tensor([25.0, 1.0]))
    R = torch.tensor([[0.5]])
    Qf = 10.0 * Q
    lqr = FiniteHorizonLQRController(env, N=N, Q=Q, R=R, Qf=Qf)

    # ----- Storage -----
    states = np.zeros((num_traj, N, 2), dtype=np.float32)
    actions = np.zeros((num_traj, N), dtype=np.float32)
    x0s = np.zeros((num_traj, 2), dtype=np.float32)

    meta = {
        "seed": seed,
        "num_traj": num_traj,
        "dt": dt,
        "T": T,
        "N": N,
        "theta_init_range": [float(math.pi - theta_range), float(math.pi + theta_range)],
        "dtheta_init_range": [-dtheta_range, dtheta_range],
        "img_size": img_size,
        "env": {"m": env.m, "l": env.l, "g": env.g, "b": env.b, "u_max": env.u_max},
        "lqr": {"Q": Q.tolist(), "R": R.tolist(), "Qf": Qf.tolist()},
    }

    print("Generating data...")
    # ----- Rollouts -----
    for i in range(num_traj):
        theta0 = np.random.uniform(math.pi - theta_range, math.pi + theta_range)
        dtheta0 = np.random.uniform(-dtheta_range, dtheta_range)
        x0 = [theta0, dtheta0]
        # x0 = torch.tensor([theta0, theta_dot0], dtype=torch.float64)
        x0s[i] = np.asarray(x0, dtype=np.float32)

        xs, us = env.rollout(lqr, x0=x0, N=N)  # xs: (N+1,2), us: (N,)
        xs = xs[:-1]  # align with actions (N)

        states[i] = xs.detach().cpu().numpy().astype(np.float32)
        actions[i] = us.detach().cpu().numpy().astype(np.float32)

        if save_images:
            traj_dir = os.path.join(img_dir, f"traj_{i:02d}")
            os.makedirs(traj_dir, exist_ok=True)
            for t in range(N):
                th = float(states[i, t, 0])
                u = float(actions[i, t])
                img = render_pendulum_frame(th, u=0.0, l=env.l, u_max=env.u_max, size=img_size) # u must be 0
                # Save as png via matplotlib
                # Using matplotlib avoids extra deps like PIL
                import matplotlib.pyplot as plt
                plt.imsave(os.path.join(traj_dir, f"frame_{t:04d}.png"), img)

    # ----- Save dataset -----
    np.savez_compressed(
        os.path.join(out_dir, "lqr_dataset.npz"),
        states=states,
        actions=actions,
        x0s=x0s,
        dt=np.float32(dt),
        N=np.int32(N),
        meta_json=json.dumps(meta),
    )

    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("\nSaved dataset to data/lqr_dataset.npz")
    if save_images:
        print("Saved images under data/images/")


if __name__ == "__main__":
    main()
