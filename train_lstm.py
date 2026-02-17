"""Train an LSTM policy: past state window -> current LQR action.

Input is a fixed-length window of states [theta, theta_dot] with length H.
The target is the LQR torque corresponding to the last state in the window.

The checkpoint stores model weights, the state normalization statistics, and
the chosen window length.

Usage:
  python train_lstm.py --data data/lqr_dataset.npz --out models/lstm.pt --horizon 10
"""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from policies import LSTMPolicy
from torch.utils.data import Dataset, DataLoader


class StateSeqDataset(Dataset):
    def __init__(self, npz_path: str, horizon: int, split: str = "train", train_frac: float = 0.9, seed: int = 0):
        d = np.load(npz_path, allow_pickle=True)
        states = d["states"]  # (num_traj, N, 2)
        actions = d["actions"]  # (num_traj, N)
        num_traj, N, _ = states.shape

        items = []  # (traj_idx, t_end)
        for i in range(num_traj):
            for t_end in range(horizon - 1, N):
                items.append((i, t_end))

        rng = np.random.default_rng(seed)
        idx = np.arange(len(items))
        rng.shuffle(idx)
        n_train = int(train_frac * len(idx))
        sel = idx[:n_train] if split == "train" else idx[n_train:]
        self.items = [items[j] for j in sel]

        self.states = torch.from_numpy(states).float()
        self.actions = torch.from_numpy(actions).float().unsqueeze(-1)
        self.horizon = horizon

        if split == "train":
            # normalization fitted on all states in train trajectories/windows
            X_all = self.states.reshape(-1, 2)
            self.mean = X_all.mean(dim=0)
            self.std = X_all.std(dim=0).clamp_min(1e-6)
        else:
            self.mean, self.std = None, None

    def set_normalization(self, mean: torch.Tensor, std: torch.Tensor):
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        traj, t_end = self.items[i]
        t0 = t_end - (self.horizon - 1)
        x = self.states[traj, t0:t_end + 1, :]  # (H,2)
        x = (x - self.mean) / self.std
        y = self.actions[traj, t_end, :]  # (1,)
        return x, y


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    ds_train = StateSeqDataset(args.data, args.horizon, split="train", train_frac=args.train_frac, seed=args.seed)
    ds_val = StateSeqDataset(args.data, args.horizon, split="val", train_frac=args.train_frac, seed=args.seed)
    ds_val.set_normalization(ds_train.mean, ds_train.std)

    train_loader = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = LSTMPolicy(hidden=args.hidden, layers=args.layers).to(device)
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
            #  TODO4.4: write your code here  #
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
        "type": "lstm",
        "model_state": model.state_dict(),
        "mean": ds_train.mean.cpu(),
        "std": ds_train.std.cpu(),
        "horizon": args.horizon,
        "config": vars(args),
        "train_losses": train_losses,
        "val_losses": val_losses,
    }
    torch.save(ckpt, args.out)
    print(f"Saved: {args.out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="data/lqr_dataset.npz")
    p.add_argument("--out", type=str, default="models/lstm.pt")
    p.add_argument("--horizon", type=int, default=10)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--layers", type=int, default=1)
    p.add_argument("--train_frac", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--print_every", type=int, default=1)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
