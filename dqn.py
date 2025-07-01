import torch
from torch import nn
import torch.nn.functional as F 

class DQN(nn.Module): 

    #This function defines the layers. 
    def __init__(self, state_dim, action_dim, hidden_dim=256): 
        super(DQN, self).__init__() 

        #Input layer (fc0) is implicit in PyTorch 
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    #This function does the calculations, transforms
    def forward(self, x):
        x = F.relu(self.fc1(x)) #x is state, in this case 12 values in an array that represent the current env state
        return self.fc2(x) 
    
if __name__ == '__main__':
    state_dim = 12 
    action_dim = 2 
    net = DQN(state_dim, action_dim)
    state = torch.randn(10, state_dim)
    output = net(state)
    print(output) # This will return Q-Values or estimated future rewards. Not probablities 