import gymnasium as gym
import numpy as np 

import matplotlib
import matplotlib.pyplot as plt

import random
import torch
from torch import nn
import yaml

from experience_replay import ReplayMemory
from dqn import DQN

from datetime import datetime, timedelta
import argparse 
import itertools

import flappy_bird_gymnasium
import os

# For printing date and time 
DATE_FORMAT = "%m-%d %H:%M:%S"

# Directory for saving run info 
RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Agent: 
    def __init__(self,hyperparameter_set):
        # Open hyperparameter config
        with open('hyperparameters.yml','r') as file: 
            all_hyperparameter_sets = yaml.safe_load(file)
            hyperparameters = all_hyperparameter_sets[hyperparameter_set] #Gets hyperparater array from yaml 

        #Hyperparameters
        self.replay_memory_size = hyperparameters['replay_memory_size'] # Size of replay memory
        self.mini_batch_size = hyperparameters['mini_batch_size'] # size of training data set sampled from the replay memory
        
        # These parameters are for Epsilon-Greedy Algo
        self.epsilon_init = hyperparameters['epsilon_init'] # 1 = 100% random actions 
        self.epsilon_decay = hyperparameters['epsilon_decay'] # random decay rate
        self.epsilon_min = hyperparameters['epsilon_min'] # minimum random actions as a percentage. Policy vs random
        
        # Optimizer Variables
        self.loss_fn = nn.MSELoss() # NN Loss Function. MSE can be swapped to a different optimizer algo
        self.learning_rate_a = hyperparameters['learning_rate_a']
        self.discount_factor_g = hyperparameters['discount_factor_g']
        self.network_sync_rate = hyperparameters['network_sync_rate'] 
        
    def run(self, is_training=True, render=False ):
        # env = gymnasium.make("FlappyBird-v0", render_mode="human" if render else None, use_lidar=False)
        env = gymnasium.make("CartPole-v1", render_mode="human" if render else None)

        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.n

        # Epsilon Value and Rewards Logs
        rewards_per_episode = []
        epsilon_history = []

        policy_dqn = DQN(num_states, num_actions).to(device)

        if is_training: 
            memory = ReplayMemory(self.replay_memory_size)
            
            epsilon = self.epsilon_init

            #Set up the Target Network - Declares the Target Network and then  loads the current weights from the policy network.
            target_dqn = DQN(num_states, num_actions).to(device)
            target_dqn.load_state_dict(policy_dqn.state_dict())

            #Track Number of steps taken. Used for syncing policy with target
            step_count = 0

            # Policy Network Optimizer. Using Adam here but can be swapped with something else
            self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_a)

        for episode in itertools.count(): 
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float, device=device)

            terminated = False
            episode_reward = 0.0

            while not terminated: # Checking if the player is still alive - Episode Loop
                
                #Epsilon-Greedy Implementation Loop - Random Action or Policy Action
                if is_training and random.random() < epsilon: 
                    action = env.action_space.sample() # If the random generated (0-1) is less than the continually degrading epsilon value, take a random action. 
                    action = torch.tensor(action, dtype=torch.int64, device=device)
                else: 
                    with torch.no_grad(): #Dont need a gradient decent calc here so we are turning it off to save comp.
                        # Need to add dimensionality for pytorch. 1d -> 2d to process with policy and then can reduce to 1d before getting index of action to be taken.
                        action = policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax() # If epsilon is lesser than the random number execute on policy. This will constatly be tuned with every episode. We get the index of the action that has the largest Q-Value (future reward)

                # Processing/Get results from decided Action:
                new_state, reward, terminated, _, info = env.step(action.item())
                
                # Accumulate Reward 
                episode_reward += reward

                # Convert to Torch Tensors
                new_state = torch.tensor(new_state, dtype=torch.float, device=device)
                reward = torch.tensor(reward, dtype=torch.float, device=device)

                if is_training:
                    #Save Experience to memory
                    memory.append((state, action, new_state, reward, terminated))

                    # Incriment the step counter
                    step_count +=1
                
                #Move to new state
                state = new_state
            
            # Update episode reward log
            rewards_per_episode.append(episode_reward)

            #Update Epsilon-Greedy and History
            epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
            epsilon_history.append(epsilon)

            # Check if enough experience has been collected 
            if len(memory)>self.mini_batch_size: 
                #Sample from memory and store in variable
                mini_batch = memory.sample(self.mini_batch_size) 
                
                self.optimize(mini_batch, policy_dqn, target_dqn)

                #Copy policy network to target network after a certain number of steps
                if step_count>self.network_sync_rate: 
                    target_dqn.load_state_dict(policy_dqn.state_dict()) #This will copy all of the weights from policy_dqn to target_dqn. 
                    step_count=0 # Reset the step count

    def optimize(self, mini_batch, policy_dqn, target_dqn): 
        #Transpose the list of experiences and separate each element 
        states, actions, new_states, rewards, terminations = zip(*mini_batch)

        #Stack tensors to create batch tensors 
        states = torch.stack(states)
        actions = torch.stack(actions)
        new_states = torch.stack(new_states)
        rewards = torch.stack(rewards)

        terminations = torch.tensor(terminations).float().to(device)

        with torch.no_grad(): 
            #Calculate the target Q-values (expected returns)
            target_q = rewards + (1-terminations) * self.discount_factor_g * target_dqn(new_states).max(dim=1)[0]

        #Calculate Q-values from current policy
        current_q = policy_dqn(states).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze() 
        
        # Compute loss for the entire minibatch
        loss = self.loss_fn(current_q, target_q)

        # Optimize the model 
        self.optimizer.zero_grad() #Clear gradients 
        loss.backward() #Compute gradients (backpropagation)
        self.optimizer.step() #Update network parameters i.e. weights and biases

if __name__ == '__main__': 
    agent = Agent('CartPole1')
    agent.run(is_training=True, render=True)