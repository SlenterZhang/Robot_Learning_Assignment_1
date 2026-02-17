"""Train an MLP policy: state -> LQR action.

This is a standard behavior-cloning setup:
  - Input: (theta, theta_dot)
  - Target: LQR torque

The checkpoint stores the model weights and the normalization statistics used
for states.

Usage:
  python train_mlp.py --data data/lqr_dataset.npz --out models/mlp.pt
"""

import argparse
import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from policies import MLPPolicy


class StateActionDataset(Dataset):
    def __init__(self, npz_path: str, split: str = "train", train_frac: float = 0.9, seed: int = 0):
        d = np.load(npz_path, allow_pickle=True)
        states = d["states"]  # (Tj, N, 2)
        actions = d["actions"]  # (Tj, N)

        X = states.reshape(-1, 2)
        y = actions.reshape(-1, 1)

        rng = np.random.default_rng(seed)
        idx = np.arange(X.shape[0])
        rng.shuffle(idx)
        n_train = int(train_frac * len(idx))
        if split == "train":
            sel = idx[:n_train]
        else:
            sel = idx[n_train:]

        self.X = torch.from_numpy(X[sel]).float()
        self.y = torch.from_numpy(y[sel]).float()

        # Normalization (fit on train split, reused for val)
        if split == "train":
            self.mean = self.X.mean(dim=0)
            self.std = self.X.std(dim=0).clamp_min(1e-6)
        else:
            # placeholders; caller should inject train stats for val
            self.mean = None
            self.std = None

    def set_normalization(self, mean: torch.Tensor, std: torch.Tensor):
        self.mean = mean
        self.std = std

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        x = (self.X[i] - self.mean) / self.std
        return x, self.y[i]


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    ds_train = StateActionDataset(args.data, split="train", train_frac=args.train_frac, seed=args.seed)
    ds_val = StateActionDataset(args.data, split="val", train_frac=args.train_frac, seed=args.seed)
    ds_val.set_normalization(ds_train.mean, ds_train.std)

    train_loader = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = MLPPolicy(hidden=args.hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    train_losses, val_losses = [], []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for x, y in train_loader:
            # x - data, y - label
            x, y = x.to(device), y.to(device)
            
            # Backpropagation
            ###################################
            #  TODO4.2: write your code here  #
            ###################################  
            # HINT: first forward pass, compute loss, zero_grad, and backward and step the optimizer
            pred = model(x)
            loss = loss_fn(pred, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            total += loss.item() * x.shape[0]
        train_loss = total / len(ds_train)

        model.eval()
        total = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                loss = loss_fn(pred, y)
                total += loss.item() * x.shape[0]
        val_loss = total / len(ds_val)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        if epoch % args.print_every == 0 or epoch == 1 or epoch == args.epochs:
            print(f"Epoch {epoch:04d} | train {train_loss:.6f} | val {val_loss:.6f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    ckpt = {
        "type": "mlp",
        "model_state": model.state_dict(),
        "mean": ds_train.mean.cpu(),
        "std": ds_train.std.cpu(),
        "config": vars(args),
        "train_losses": train_losses,
        "val_losses": val_losses,
    }
    torch.save(ckpt, args.out)
    print(f"Saved: {args.out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="data/lqr_dataset.npz")
    p.add_argument("--out", type=str, default="models/mlp.pt")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--train_frac", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--print_every", type=int, default=5)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
