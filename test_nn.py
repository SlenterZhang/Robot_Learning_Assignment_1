"""Evaluate a learned policy in two ways:

1) Regression fit (predicted action vs LQR action) on the dataset
2) Closed-loop rollout in the environment, with time-series plots + animation

Usage examples:
  # after running collect_lqr_data.py and training
  python test_nn.py --kind mlp  --ckpt models/mlp.pt
  python test_nn.py --kind cnn  --ckpt models/cnn.pt
  python test_nn.py --kind lstm --ckpt models/lstm.pt
"""

import argparse
import math
import os
import numpy as np
import torch

from env import InvertedPendulum
from controllers import FiniteHorizonLQRController, load_learned_controller
from utils import plot_time_series, animate_pendulum


def load_dataset(npz_path: str):
    d = np.load(npz_path, allow_pickle=True)
    states = d["states"].astype(np.float32)    # (num_traj, N, 2)
    actions = d["actions"].astype(np.float32)  # (num_traj, N)
    dt = float(d["dt"])
    return states, actions, dt


@torch.no_grad()
def predict_actions(kind: str, controller, states: np.ndarray):
    """Return predicted actions shaped (num_traj, N)."""
    num_traj, N, _ = states.shape
    preds = np.zeros((num_traj, N), dtype=np.float32)
    for i in range(num_traj):
        if hasattr(controller, "reset"):
            controller.reset()
        for t in range(N):
            x = torch.from_numpy(states[i, t]).to(torch.float64)
            u = controller(x, t)
            preds[i, t] = float(u)
    return preds


def plot_regression(pred: np.ndarray, gt: np.ndarray, out_path: str):
    import matplotlib.pyplot as plt
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    p = pred.reshape(-1)
    y = gt.reshape(-1)
    mse = float(np.mean((p - y) ** 2))

    plt.figure(figsize=(5, 5))
    plt.scatter(y, p, s=3, alpha=0.4)
    lim = max(np.max(np.abs(y)), np.max(np.abs(p)))
    plt.plot([-lim, lim], [-lim, lim], linestyle="--")
    plt.xlabel("LQR action")
    plt.ylabel("NN predicted action")
    plt.title(f"Action fit (MSE={mse:.4e})")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[Saved regression plot] {out_path}")


def plot_training_curves(ckpt_path: str, out_path: str):
    import matplotlib.pyplot as plt
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    tr = ckpt.get("train_losses", [])
    va = ckpt.get("val_losses", [])
    if len(tr) == 0:
        print("No losses found in checkpoint; skipping loss plot.")
        return

    plt.figure(figsize=(6, 4))
    plt.plot(tr, label="train")
    if len(va) == len(tr):
        plt.plot(va, label="val")
    plt.xlabel("epoch")
    plt.ylabel("MSE")
    plt.title("Training curve")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[Saved loss curve] {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", type=str, required=True, choices=["mlp", "cnn", "lstm"])
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--data", type=str, default="data/lqr_dataset.npz")
    ap.add_argument("--traj", type=int, default=0, help="which trajectory to animate/plot")
    args = ap.parse_args()

    # --- dataset ---
    states, actions, dt = load_dataset(args.data)
    num_traj, N, _ = states.shape
    assert 0 <= args.traj < num_traj

    # --- env + controller ---
    env = InvertedPendulum(m=1.0, l=1.0, g=9.81, b=0.5, u_max=10.0, dt=dt)
    ctrl = load_learned_controller(args.kind, args.ckpt, env)

    # --- regression evaluation ---
    pred = predict_actions(args.kind, ctrl, states)
    os.makedirs("figures", exist_ok=True)
    plot_regression(pred, actions, out_path=f"figures/{args.kind}_regression.png")
    plot_training_curves(args.ckpt, out_path=f"figures/{args.kind}_loss.png")

    # --- rollout in env from the chosen initial state ---
    x0 = states[args.traj, 0].tolist()
    xs, us = env.rollout(ctrl, x0=x0, N=N)

    # figures: time-series and animation
    gains_dict = {"ckpt": args.ckpt}
    plot_time_series(xs, us, dt=env.dt, u_max=env.u_max, x0=x0,
                     controller_type=args.kind, gains_dict=gains_dict)
    animate_pendulum(xs, us, dt=env.dt, l=env.l, u_max=env.u_max,
                     title=f"{args.kind.upper()} rollout")


if __name__ == "__main__":
    main()
