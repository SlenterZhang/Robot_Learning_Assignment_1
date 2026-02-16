"""Utilities
NO need to change this
"""

import math
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

# Offscreen rendering for dataset image generation
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# ----------------------------
# Utils
# ----------------------------
def wrap_angle(theta: float) -> float:
    """Wrap angle to (-pi, pi]."""
    return (theta + math.pi) % (2 * math.pi) - math.pi

def fmt(x):
    if isinstance(x, (list, tuple)):
        return "_".join(fmt(v) for v in x)
    try:
        # numpy scalar, torch scalar, python float/int
        x = float(x)
    except Exception as e:
        raise TypeError(f"Unsupported type for fmt(): {type(x)}") from e

    return f"{x:.2f}".replace(" ", "").replace("[", "").replace("]", "").replace(",", "_").replace(".", "pt")

# ----------------------------
# Animation (single rollout)
# ----------------------------
def animate_pendulum(xs, us=None, dt=0.02, l=1.0, u_max=5.0,
                     title="Inverted Pendulum", save_path=None, fps=50):
    """
    xs: (N+1,2) [theta, theta_dot]
    us: (N,) torque (optional)
    """
    if hasattr(xs, "detach"):
        xs = xs.detach().cpu().numpy()
    if us is not None and hasattr(us, "detach"):
        us = us.detach().cpu().numpy()

    theta = xs[:, 0]
    T = xs.shape[0]
    N = T - 1
    if us is not None:
        assert us.shape[0] == N, "us must have length N when xs has length N+1"

    fig, ax = plt.subplots(figsize=(4.8, 4.8))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-1.2 * l, 1.2 * l)
    ax.set_ylim(-1.2 * l, 1.2 * l)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)

    ax.plot([0], [0], marker="o", markersize=6)

    rod, = ax.plot([], [], linewidth=3)
    bob, = ax.plot([], [], marker="o", markersize=10)

    time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes)
    step_text = ax.text(0.02, 0.88, "", transform=ax.transAxes)
    u_text = ax.text(0.02, 0.81, "", transform=ax.transAxes) if us is not None else None

    # torque bar
    if us is not None:
        bar_ax = fig.add_axes([0.15, 0.02, 0.7, 0.08])
        bar_ax.set_xlim(-u_max, u_max)
        bar_ax.set_ylim(0, 1)
        bar_ax.set_yticks([])
        bar_ax.set_xticks([-u_max, 0, u_max])
        bar_ax.set_title("u_k (torque)", fontsize=10)
        torque_bar = bar_ax.barh([0.5], [0.0], height=0.5)[0]
    else:
        torque_bar = None

    def init():
        rod.set_data([], [])
        bob.set_data([], [])
        time_text.set_text("")
        step_text.set_text("")
        if u_text is not None:
            u_text.set_text("")
        if torque_bar is not None:
            torque_bar.set_width(0.0)
        artists = (rod, bob, time_text, step_text)
        if u_text is not None:
            artists += (u_text,)
        if torque_bar is not None:
            artists += (torque_bar,)
        return artists

    def update(k):
        th = theta[k]
        x = l * np.sin(th)
        y = -l * np.cos(th)

        rod.set_data([0.0, x], [0.0, y])
        bob.set_data([x], [y])

        time_text.set_text(f"t = {k*dt:.2f} s")
        step_text.set_text(f"k = {k}/{N}")

        if us is not None:
            if k < N:
                uk = float(us[k])
                u_text.set_text(f"u[{k}] = {uk:+.3f}")
                if torque_bar is not None:
                    torque_bar.set_width(uk)
            else:
                u_text.set_text("terminal (no u)")
                if torque_bar is not None:
                    torque_bar.set_width(0.0)

        artists = (rod, bob, time_text, step_text)
        if u_text is not None:
            artists += (u_text,)
        if torque_bar is not None:
            artists += (torque_bar,)
        return artists

    ani = animation.FuncAnimation(
        fig, update, frames=T, init_func=init,
        interval=1000.0 / fps, blit=True
    )
    plt.show()

    if save_path is not None:
        if save_path.endswith(".mp4"):
            writer = animation.FFMpegWriter(fps=fps)
            ani.save(save_path, writer=writer)
        elif save_path.endswith(".gif"):
            ani.save(save_path, writer="pillow", fps=fps)
        else:
            raise ValueError("save_path must end with .mp4 or .gif")

    return ani


def plot_time_series(xs, us, dt, u_max, x0, controller_type, gains_dict):
    """
    Plot:
      1) theta vs time
      2) theta_dot vs time
      3) control input u vs time

    xs: (N+1,2) torch or numpy  [theta, theta_dot]
    us: (N,)   torch or numpy  [u]
    gains_dict:
      - LQR: {"Q": Q, "R": R, "Qf": Qf}
      - PID: {"kp": kp, "ki": ki, "kd": kd}
    """
    # ---- convert to numpy ----
    if hasattr(xs, "detach"):
        xs = xs.detach().cpu().numpy()
    if hasattr(us, "detach"):
        us = us.detach().cpu().numpy()

    theta = xs[:, 0]
    theta = np.unwrap(theta)
    theta_dot = xs[:, 1]
    u = us

    T = len(theta)
    t = np.arange(T) * dt
    t_u = np.arange(len(u)) * dt

    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)

    # ---------------- theta ----------------
    axes[0].plot(t, theta, linewidth=2)
    axes[0].set_ylabel(r"$\theta$ (rad)")
    axes[0].grid(True)

    # ---------------- theta_dot ----------------
    axes[1].plot(t, theta_dot, linewidth=2)
    axes[1].set_ylabel(r"$\dot{\theta}$ (rad/s)")
    axes[1].grid(True)

    # ---------------- control ----------------
    axes[2].step(t_u, u, where="post", linewidth=2)
    axes[2].set_ylabel(r"$u$ (torque)")
    axes[2].axhline(-u_max, linestyle="--", color="gray", label="max u")
    axes[2].axhline(u_max, linestyle="--", color="gray")
    axes[2].set_xlabel("time (s)")
    axes[2].legend()
    axes[2].grid(True)

    # ---------------- title ----------------
    if controller_type.lower() == "lqr":
        title = (
            f"Finite-Horizon Discrete LQR, x0 =[{x0[0]:.2f}, {x0[1]:.2f}]\n"
            f"Q = {gains_dict['Q']},  "
            f"R = {gains_dict['R']},  "
            f"Qf = {gains_dict['Qf']}"
        )
    elif controller_type.lower() == "pid":
        title = (
            f"PID Control, x0 =[{x0[0]:.2f}, {x0[1]:.2f}]\n"
            f"$k_p$ = {gains_dict['kp']},  "
            f"$k_i$ = {gains_dict['ki']},  "
            f"$k_d$ = {gains_dict['kd']}"
        )
    elif controller_type.lower() in ["mlp", "cnn", "lstm"]:
        title = (
            f"Learned Policy ({controller_type.upper()}), x0 =[{x0[0]:.2f}, {x0[1]:.2f}]\n"
            f"ckpt = {gains_dict.get('ckpt', 'N/A')}"
        )
    elif controller_type.lower() == "free":
        title = f"Open-Loop Inverted Pendulum Model Test (u = 0), x0 =[{x0[0]:.2f}, {x0[1]:.2f}]"
    else:
        title = "Unknown Controller"

    fig.suptitle(title, fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    

    # ---------------- save figure ----------------
    save_dir = "figures"
    os.makedirs(save_dir, exist_ok=True)

    if controller_type.lower() == "lqr":
        fname = f"pendulum_lqr_init{fmt(x0)}_Q{fmt(gains_dict['Q'])}_R{fmt(gains_dict['R'])}_Qf{fmt(gains_dict['Qf'])}.png"
    elif controller_type.lower() == "pid":
        fname = f"pendulum_pid_init{fmt(x0)}_kp{fmt(gains_dict['kp'])}_ki{fmt(gains_dict['ki'])}_kd{fmt(gains_dict['kd'])}.png"
    elif controller_type.lower() in ["mlp", "cnn", "lstm"]:
        fname = f"pendulum_{controller_type.lower()}_init{fmt(x0)}.png"
    elif controller_type.lower() in ["mlp", "cnn", "lstm"]:
        fname = f"pendulum_{controller_type.lower()}_init{fmt(x0)}.png"
    else:
        fname = f"pendulum_init{fmt(x0)}.png"

    save_path = os.path.join(save_dir, fname)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"[Saved figure to {save_path}]")

    # show figure
    plt.show()


def render_pendulum_frame(theta: float,
                          u: float = 0.0,
                          l: float = 1.0,
                          u_max: float = 10.0,
                          size: int = 64) -> np.ndarray:
    """Render a single pendulum observation as an RGB uint8 image.

    - theta convention: 0 down, pi up
    - returns image of shape (size, size, 3)

    This is intentionally simple and deterministic so CNN training and
    policy rollout use the same rendering.
    """
    # Create a tiny offscreen figure
    fig = plt.Figure(figsize=(1, 1), dpi=size)
    canvas = FigureCanvas(fig)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(-1.2 * l, 1.2 * l)
    ax.set_ylim(-1.2 * l, 1.2 * l)
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')

    # Draw pivot
    ax.plot([0], [0], marker='o', markersize=3)

    # Rod and bob
    x = l * np.sin(theta)
    y = -l * np.cos(theta)
    ax.plot([0, x], [0, y], linewidth=2)
    ax.plot([x], [y], marker='o', markersize=4)

    # Small torque indicator (horizontal bar) at bottom
    # normalized to [-1, 1]
    if u_max > 1e-9:
        un = float(np.clip(u / u_max, -1.0, 1.0))
        ax.plot([-1.0, 1.0], [-1.05 * l, -1.05 * l], linewidth=1)
        ax.plot([0, un], [-1.05 * l, -1.05 * l], linewidth=3)

    canvas.draw()
    buf = np.asarray(canvas.buffer_rgba())
    img = buf[..., :3].copy()
    return img


def image_to_tensor(img: np.ndarray) -> np.ndarray:
    """Convert uint8 HxWx3 to float32 CHW in [0,1]."""
    if img.dtype != np.uint8:
        raise ValueError("Expected uint8 image")
    x = img.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))
    return x