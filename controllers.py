import numpy as np
import torch 
import math 
from utils import wrap_angle
from env import InvertedPendulum


# ----------------------------
# PID Controller
# ----------------------------
class PIDController:
    """
    PID around upright equilibrium (theta=pi).

    Students should implement:
      - __call__(x, t) returning torque u
      - reset() to clear integrator state

    Hint:
      e = wrap(theta - pi)
      de can be approximated by -theta_dot
      u = kp*e + ki*int_e + kd*de
    """
    def __init__(self, dt, u_max, kp=0.0, ki=0.0, kd=0.0):
        self.dt = float(dt)
        self.u_max = float(u_max)
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)

        self.int_e = 0.0 # integrated error

    def reset(self):
        self.int_e = 0.0

    def __call__(self, x: torch.Tensor, t: int) -> torch.Tensor:
        theta = float(x[0])
        theta_dot = float(x[1])

        ###################################
        #  TODO2.1: write your code here  #
        ################################### 
        e = wrap_angle(theta - math.pi)    
        de = -theta_dot

        u_pre = -(self.kp * e + self.ki * self.int_e) + (self.kd * de)
        if abs(u_pre) < self.u_max:
            self.int_e += e * self.dt

        int_limit = 5.0
        self.int_e = float(np.clip(self.int_e, -int_limit, int_limit))

        u = -(self.kp * e + self.ki * self.int_e) + (self.kd * de)
        ###################################

        u = float(np.clip(u, -self.u_max, self.u_max))
        return torch.tensor(u, dtype=torch.float64)



# ----------------------------
# Finite Horizon LQR
# ----------------------------
class FiniteHorizonLQRController:
    """
    Finite-horizon discrete LQR around upright equilibrium theta=pi.
    Uses nonlinear simulator for rollout, but control computed from linearized discrete model.
    """
    def __init__(self, env: InvertedPendulum, N=400, Q=None, R=None, Qf=None):
        self.env = env
        self.N = N

        self.x_eq = torch.tensor([math.pi, 0.0])
        self.u_eq = torch.tensor(0.0)

        # Linearized dynamics
        self.Ad, self.Bd = env.discretize_euler_linear()
        dtype = self.Ad.dtype
        device = self.Ad.device

        if Q is None:
            Q = torch.diag(torch.tensor([25.0, 1.0]))
        if R is None:
            R = torch.tensor([[0.5]])
        if Qf is None:
            self.Qf = 10.0 * self.Q

        # if user passed tensors, still force dtype/device
        Q  = Q.to(dtype=dtype, device=device)
        R  = R.to(dtype=dtype, device=device)
        Qf = Qf.to(dtype=dtype, device=device)

        self.Q = Q 
        self.R = R 
        self.Qf = Qf 

        self.K_seq, self.P_seq = self.finite_horizon_lqr(self.Ad, self.Bd, self.Q, self.R, self.Qf, N)

    def reset(self):
        pass  # nothing stateful

    # run time to use LQR
    def __call__(self, x: torch.Tensor, t: int) -> torch.Tensor:    
        t = min(max(int(t), 0), self.N - 1)

        ###################################
        #  TODO3.3: write your code here  #
        ###################################  
        # NOTE: dx = x - x_equilibrium
        x_eq = self.x_eq.to(dtype=x.dtype, device=x.device)
        dx = x - x_eq
        ###################################

        # wrap angle error
        dx0 = wrap_angle(dx[0])
        dx = dx.clone()
        dx[0] = dx0

        ###################################
        #  TODO3.3: write your code here  #
        ###################################  
        # NOTE: u = u_equilibrium + du
        # NOTE: the dimension of the tensor is important
        K = self.K_seq[t].to(dtype=x.dtype, device=x.device)   # (1,2)
        dx_col = dx.view(-1, 1)                                # (2,1)
        du = -(K @ dx_col).view(())                            # scalar tensor (0-dim)
        u_eq = self.u_eq.to(dtype=x.dtype, device=x.device)    # scalar tensor
        u = u_eq + du
        ###################################        

        u = torch.clamp(u, -self.env.u_max, self.env.u_max)
        return u

    # obtain LQR
    def finite_horizon_lqr(self, Ad, Bd, Q, R, Qf, N):
        """
        Dynamics: x_{t+1} = A x_t + B u_t
        Cost: sum x^T Q x + u^T R u + terminal x^T Qf x
        Returns:
        K_seq: list length N, each K[t] is (m,n)
        P_seq: list length N+1
        """
        n = Ad.shape[0]
        m = Bd.shape[1]
        assert Q.shape == (n, n)
        assert Qf.shape == (n, n)
        assert R.shape == (m, m)

        ###################################
        #  TODO3.3: write your code here  #
        ###################################  
        # NOTE: u = -Kx
        # NOTE: J^star = x^T P x

        P_seq = [None] * (N + 1)
        K_seq = [None] * N

        # Termination condation
        P_seq[N] = Qf.clone()

        # backward Riccati recursion
        for t in range(N - 1, -1, -1):
            Pn1 = P_seq[t + 1]

            # S = R + B^T P B   (m,m)
            S = R + Bd.T @ Pn1 @ Bd

            # K = S^{-1} (B^T P A)   (m,n)
            # use solve for stability: S K = (B^T P A)
            K = torch.linalg.solve(S, Bd.T @ Pn1 @ Ad)

            # P = Q + A^T P A - A^T P B K
            P = Q + Ad.T @ Pn1 @ Ad - Ad.T @ Pn1 @ Bd @ K

            # (optional) enforce symmetry numerically
            P = 0.5 * (P + P.T)
            K_seq[t] = K
            P_seq[t] = P
        return K_seq, P_seq


# ----------------------------
# Learned (imitation) controllers
# ----------------------------
import os
import torch.nn as nn
from utils import render_pendulum_frame, image_to_tensor


class LearnedMLPController:
    """State -> action controller loaded from models/mlp.pt"""

    def __init__(self, ckpt_path: str, u_max: float = 10.0):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        assert ckpt.get("type") == "mlp", "Checkpoint type mismatch"
        self.mean = ckpt["mean"].float()
        self.std = ckpt["std"].float()
        self.u_max = float(u_max)

        hidden = ckpt.get("config", {}).get("hidden", 128)

        from policies import MLPPolicy
        self.model = MLPPolicy(hidden=hidden)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()

    def reset(self):
        pass

    @torch.no_grad()
    def __call__(self, x: torch.Tensor, t: int) -> torch.Tensor:
        z = (x.float() - self.mean) / self.std
        # Inference
        ###################################
        #  TODO4.2: write your code here  #
        ###################################             
        u = self.model(z.unsqueeze(0)).squeeze(0).squeeze(-1)
        ###################################

        u = torch.clamp(u, -self.u_max, self.u_max)
        return u.to(torch.float64)


class LearnedCNNController:
    """Image(obs(theta,u=0)) -> action controller loaded from models/cnn.pt.

    At test time, we render the current theta into the same 64x64 RGB image
    used for training.
    """

    def __init__(self, ckpt_path: str, l: float = 1.0, u_max: float = 10.0, img_size: int = 64):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        assert ckpt.get("type") == "cnn", "Checkpoint type mismatch"
        self.u_max = float(u_max)
        self.l = float(l)
        self.img_size = int(img_size)

        from policies import CNNPolicy
        self.model = CNNPolicy()
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()

    def reset(self):
        pass

    @torch.no_grad()
    def __call__(self, x: torch.Tensor, t: int) -> torch.Tensor:
        theta = float(x[0])
        img = render_pendulum_frame(theta, u=0.0, l=self.l, u_max=self.u_max, size=self.img_size) # u must be zero
        obs = image_to_tensor(img)
        obs = torch.from_numpy(obs).unsqueeze(0)  # (1,3,H,W)
        # Inference
        ###################################
        #  TODO4.3: write your code here  #
        ###################################             
        u = self.model(obs).squeeze(0).squeeze(-1)
        ###################################

        u = torch.clamp(u, -self.u_max, self.u_max)
        return u.to(torch.float64)


class LearnedLSTMController:
    """Past H states -> action controller loaded from models/lstm.pt."""

    def __init__(self, ckpt_path: str, u_max: float = 10.0):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        assert ckpt.get("type") == "lstm", "Checkpoint type mismatch"
        self.mean = ckpt["mean"].float()
        self.std = ckpt["std"].float()
        self.H = int(ckpt["horizon"])
        self.u_max = float(u_max)

        hidden = ckpt.get("config", {}).get("hidden", 64)
        layers = ckpt.get("config", {}).get("layers", 1)

        from policies import LSTMPolicy
        self.model = LSTMPolicy(hidden=hidden, layers=layers)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()

        self.buffer = []  # list of last H states

    def reset(self):
        self.buffer = []

    @torch.no_grad()
    def __call__(self, x: torch.Tensor, t: int) -> torch.Tensor:
        self.buffer.append(x.float())
        if len(self.buffer) > self.H:
            self.buffer.pop(0)
        # pad with the first state if not enough history
        while len(self.buffer) < self.H:
            self.buffer.insert(0, self.buffer[0].clone())

        seq = torch.stack(self.buffer, dim=0)  # (H,2)
        seq = (seq - self.mean) / self.std
        seq = seq.unsqueeze(0)  # (1,H,2)
        # Inference
        ###################################
        #  TODO4.4: write your code here  #
        ###################################               
        u = self.model(seq).squeeze(0).squeeze(-1)
        ###################################

        u = torch.clamp(u, -self.u_max, self.u_max)
        return u.to(torch.float64)


def load_learned_controller(kind: str, ckpt_path: str, env: InvertedPendulum):
    kind = kind.lower()
    if kind == "mlp":
        return LearnedMLPController(ckpt_path, u_max=env.u_max)
    if kind == "cnn":
        return LearnedCNNController(ckpt_path, l=env.l, u_max=env.u_max, img_size=64)
    if kind == "lstm":
        return LearnedLSTMController(ckpt_path, u_max=env.u_max)
    raise ValueError("kind must be one of: mlp, cnn, lstm")
