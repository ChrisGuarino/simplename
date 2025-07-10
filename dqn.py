import torch
from torch import nn
import torch.nn.functional as F 

class DQN(nn.Module): 

    #This function defines the layers. 
    def __init__(self, state_dim, action_dim, hidden_dim=256, enable_dueling_dqn = True): 
        super(DQN, self).__init__() 

        self.enable_dueling_dqn = enable_dueling_dqn

        #Input layer (fc0) is implicit in PyTorch 
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        
        # Dueling DQN implementation
        if self.enable_dueling_dqn: 
            # Value Stream Layers
            self.fc_value = nn.Linear(hidden_dim, 256)
            self.value = nn.Linear(256, 1)

            # Advantages Stream Layers 
            self.fc_advantages = nn.Linear(hidden_dim, 256)
            self.advantages = nn.Linear(256, action_dim)
        else:
            self.fc2 = nn.Linear(hidden_dim, action_dim) # Non-Dueling DQN output layer

    #This function does the calculations, transforms
    def forward(self, x):
        # Neuron Activation Funtions 
        x = F.relu(self.fc1(x)) #x is state, in this case 12 values in an array that represent the current env state

        # Check for Dueling DQN
        if self.enable_dueling_dqn: 
            # Value Calc 
            v = F.relu(self.fc_value(x))
            V = self.value(v)

            # Advantages Calc
            a = F.relu(self.fc_advantages(x))
            A = self.advantages(a)

            # Calculate Q
            Q = V + A - torch.mean(A, dim=1, keepdim=True)
        else: 
            Q = self.fc2(x) 
        return Q
    
if __name__ == '__main__':
    state_dim = 12 
    action_dim = 2 
    net = DQN(state_dim, action_dim)
    state = torch.randn(10, state_dim)
    output = net(state)
    print(output) # This will return Q-Values or estimated future rewards. Not probablities 