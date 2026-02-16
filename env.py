import numpy as np
import torch 
from utils import wrap_angle

# ----------------------------
# Inverted Pendulum (nonlinear discrete simulator)
# State: x = [theta, theta_dot]
# theta=0 down, theta=pi up
# ----------------------------
class InvertedPendulum:
    def __init__(self, m=1.0, l=1.0, g=9.81, b=0.5, u_max=10.0, dt=0.02):
        self.m = m
        self.l = l
        self.g = g
        self.b = b
        self.u_max = u_max
        self.dt = dt
        self.I = m * l * l  # point mass at end

    def f(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        # Dynamics function 
        th, thd = x
        u = torch.clamp(u, -self.u_max, self.u_max)
        
        ###################################
        #  TODO1.2: write your code here  #
        ###################################
        thdd = (u - self.b * thd - self.m * self.g * self.l * torch.sin(th)) / self.I
        
        return torch.stack([thd, thdd])

    def discretize_euler_linear(self):
        # Around equilibirulum pi
        # Linearize continuous dynamics (A, B), then discretize (Euler) to get (Ad,Bd)

        ###################################
        #  TODO3.3: write your code here  #
        ###################################  

        return Ad, Bd

    def step(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        # RK4 method for better approximation accuracy in discrete time
        k1 = self.f(x, u)
        k2 = self.f(x + 0.5 * self.dt * k1, u)
        k3 = self.f(x + 0.5 * self.dt * k2, u)
        k4 = self.f(x + self.dt * k3, u)

        x_next = x + (self.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        return x_next

    def rollout(self, controller, x0=None, N=0):
        """
        Returns:
        xs: (N+1,2) torch
        us: (N,)   torch
        """
        # if x0 is None:
        #     x0 = torch.tensor([math.pi + 0.4, 0.0], dtype=torch.float64)
        # else:
        #     x0 = torch.tensor(x0, dtype=torch.float64)
        x0 = torch.tensor(x0, dtype=torch.float64)
        x0[0] = torch.tensor(wrap_angle(float(x0[0])))
        x = x0.clone()

        if hasattr(controller, "reset"):
            controller.reset()

        xs = []
        us = []
        for t in range(N):
            u = controller(x, t)
            xs.append(x.clone())
            us.append(u.clone())
            x = self.step(x, u)
        xs.append(x.clone())
        return torch.stack(xs), torch.stack(us).view(-1)

    def simulate_open_loop(self, x0, N):
        """
        Simulate pendulum with u(t)=0
        """
        x0 = torch.tensor(x0)
        x0[0] = torch.tensor(wrap_angle(float(x0[0])))
        x = x0.clone()

        xs = [x.clone()]
        us = []

        for _ in range(N):
            u = torch.tensor(0.0)
            us.append(u)
            x = self.step(x, u)
            xs.append(x.clone())

        return torch.stack(xs), torch.stack(us)
