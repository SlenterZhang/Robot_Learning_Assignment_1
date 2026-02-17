import torch
import torch.nn as nn

class MLPPolicy(nn.Module):
    def __init__(self, in_dim=2, hidden=128):
        super().__init__()
        
        
        # Define NN Architecture
        ###################################
        #  TODO4.2: write your code here  #
        ###################################
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )
        ###################################
        

    def forward(self, x):
        return self.net(x)


class CNNPolicy(nn.Module):
    def __init__(self, in_ch=3):
        super().__init__()

        # Define NN Architecture
        ###################################
        #  TODO4.3: write your code here  #
        ###################################          
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 16, kernel_size=5, stride=2, padding=2), # -> (B,16,32,32)
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2), # -> (B, 32, 16, 16)
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # -> (B, 64, 8, 8)
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)), # -> (B,64,4,4)
            nn.Flatten(),                   # -> (B, 64*4*4)
        )
        self.head = nn.Sequential(
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )
        ###################################

    def forward(self, x):
        z = self.conv(x)
        return self.head(z)


class LSTMPolicy(nn.Module):
    def __init__(self, in_dim=2, hidden=64, layers=1):
        super().__init__()

        # Define NN Architecture
        ###################################
        #  TODO4.4: write your code here  #
        ###################################             
        self.lstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )
        ###################################


    def forward(self, x):
        # x: (B,H,2)
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(last)
