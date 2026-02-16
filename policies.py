import torch
import torch.nn as nn

class MLPPolicy(nn.Module):
    def __init__(self, in_dim=2, hidden=128):
        super().__init__()
        
        
        # Define NN Architecture
        ###################################
        #  TODO4.2: write your code here  #
        ###################################
        self.net = TO_IMPLEMENT  
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
        self.conv = TO_IMPLEMENT
        self.head = TO_IMPLEMENT # MLP head
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
        self.lstm = TO_IMPLEMENT
        self.head = TO_IMPLEMENT # MLP head
        ###################################


    def forward(self, x):
        # x: (B,H,2)
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(last)
