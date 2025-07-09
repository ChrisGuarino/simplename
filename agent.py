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

import os
import flappy_bird_gymnasium

# For printing date and time 
DATE_FORMAT = "%m-%d %H:%M:%S"

# Directory for saving run info 
RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)

# Used to generate plots and save them as images vs. just rendering to the screen.
matplotlib.use('Agg')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu' # To force CPU

class Agent: 
    def __init__(self,hyperparameter_set):
        # Open hyperparameter config
        with open('hyperparameters.yml','r') as file: 
            all_hyperparameter_sets = yaml.safe_load(file)
            hyperparameters = all_hyperparameter_sets[hyperparameter_set] # Gets hyperparater array from yaml 

        #Hyperparameters - Adjustable
        self.env_id = hyperparameters['env_id']
        self.learning_rate_a = hyperparameters['learning_rate_a'] # Learning Rate (alpha)
        self.discount_factor_g = hyperparameters['discount_factor_g'] # Discound rate (gamma)
        self.network_sync_rate = hyperparameters['network_sync_rate'] # Number of steps that the agent takes before syncing the policy network with the target network
        self.replay_memory_size = hyperparameters['replay_memory_size'] # Size of replay memory
        self.mini_batch_size = hyperparameters['mini_batch_size'] # size of training data set sampled from the replay memory
        self.epsilon_init = hyperparameters['epsilon_init'] # 1 = 100% random actions 
        self.epsilon_decay = hyperparameters['epsilon_decay'] # random decay rate
        self.epsilon_min = hyperparameters['epsilon_min'] # minimum random actions as a percentage. Policy vs random
        self.stop_on_reward = hyperparameters['stop_on_reward'] # stop training after reaching this number of rewards
        self.fc1_nodes = hyperparameters['fc1_nodes'] # Number of 1st Hidden layer nodes
        self.env_make_params = hyperparameters.get('env_make_nodes',{}) # Get optional environment-specific paramters
        self.enable_double_dqn = hyperparameters['enable_double_dqn'] # Double DQN on/off
        
        # NN Variables
        self.loss_fn = nn.MSELoss() # NN Loss Function. MSE can be swapped to a different optimizer algo
        self.optimizer = None # NN optimizer, will initize later on

        # Path to Run Info
        self.LOG_FILE = os.path.join(RUNS_DIR, f'{hyperparameter_set}.log')
        self.MODEL_FILE = os.path.join(RUNS_DIR, f'{hyperparameter_set}.pt')
        self.GRAPH_FILE = os.path.join(RUNS_DIR, f'{hyperparameter_set}.png')
        
    def run(self, is_training=True, render=False ):

        #Logging setup
        if is_training: 
            start_time = datetime.now()
            last_graph_update_time = start_time

            log_message = f'{start_time.strftime(DATE_FORMAT)}: Training starting on {device}...'
            print(log_message)
            with open(self.LOG_FILE, 'w') as file: 
                file.write(log_message+'\n')

        #Create an instance of the environment
        print(f'Environment: {self.env_id}')
        env = gym.make(self.env_id, render_mode="human" if render else None, **self.env_make_params) # **self.env_make_params is in the case you need to pass in environment specific paramters

        # Number of possible actions
        num_actions = env.action_space.n
        
        # Get obeservation space size
        num_states = env.observation_space.shape[0]

        # Epsilon Value and Rewards Logs
        rewards_per_episode = []

        # Create policy and target networks
        policy_dqn = DQN(num_states, num_actions, self.fc1_nodes).to(device)

        if is_training: 

            #Initialize epsilon            
            epsilon = self.epsilon_init

            #Initialize replay memory
            memory = ReplayMemory(self.replay_memory_size)

            #Set up the Target Network - Declares the Target Network and then  loads the current weights from the policy network.
            target_dqn = DQN(num_states, num_actions, self.fc1_nodes).to(device)
            target_dqn.load_state_dict(policy_dqn.state_dict())
            
            # Policy Network Optimizer. Using Adam here but can be swapped with something else
            self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_a)
            
            # List to keep track of epsilon decay
            epsilon_history = []

            #Track Number of steps taken. Used for syncing policy with target
            step_count = 0

            # Track best reward 
            best_reward = -9999999
        
        else: 
            # Load learned policy 
            policy_dqn.load_state_dict(torch.load(self.MODEL_FILE))

            # Switch to evaluation mode 
            policy_dqn.eval()

        # Training Loop, runs indefinately until manual stop
        for episode in itertools.count(): 
            state, _ = env.reset() # Initialize the environment. Rest returns (state,info)
            state = torch.tensor(state, dtype=torch.float, device=device) # Convert state to tensor directly on the device

            terminated = False # True when the agent reaches it's goal or fails
            episode_reward = 0.0 # Used to accumulate rewards per episode

            while(not terminated and episode_reward < self.stop_on_reward): # Checking if the player is still alive - Episode Loop
                
                #Epsilon-Greedy Implementation Loop - Random Action or Policy Action
                if is_training and random.random() < epsilon: 
                    # Select a random action
                    action = env.action_space.sample() # If the random generated (0-1) is less than the continually degrading epsilon value, take a random action. 
                    action = torch.tensor(action, dtype=torch.int64, device=device)
                else: 
                    # Select the best action
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

            # Save model when new best reward is achieved
            if is_training: 
                if episode_reward > best_reward: 
                    log_message = f'{datetime.now().strftime(DATE_FORMAT)}: New Best Reward {episode_reward:0.1f}(+{episode_reward-best_reward}) at episode {episode}'
                    print(log_message)
                    with open(self.LOG_FILE, 'a') as file: 
                        file.write(log_message + '\n')

                    torch.save(policy_dqn.state_dict(), self.MODEL_FILE)
                    best_reward = episode_reward

                # Update graph every x seconds
                current_time = datetime.now() 
                if current_time - last_graph_update_time > timedelta(seconds=10): 
                    self.save_graph(rewards_per_episode, epsilon_history)
                    last_graph_update_time = current_time

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

    def save_graph(self, rewards_per_episode, epsilon_history):
        #Save plots 
        fig = plt.figure(1)

        # Plot average rewards (Y-axis) vs episodes (X-axis)
        mean_rewards = np.zeros(len(rewards_per_episode))
        for x in range(len(mean_rewards)):
            mean_rewards[x] = np.mean(rewards_per_episode[max(0,x-99):(x+1)])
        plt.subplot(121) #plot on 1 row x 2 col grid, at cell 1
        # plt.xlabel('Episodes')
        plt.ylabel('Mean Rewards')
        plt.plot(mean_rewards)

        # Plot epsilon decay (Y-axis) vs episodes (X-axis)
        plt.subplot(122) #plot on 1 row x 2 col grid, at cell 2
        # plt.xlabel('Time Steps')
        plt.ylabel('Epsilon Decay')
        plt.plot(epsilon_history)

        plt.subplots_adjust(wspace=1.0, hspace=1.0)

        #Save Plots 
        fig.savefig(self.GRAPH_FILE)
        plt.close(fig)

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
            # Checks for Double DQN
            if self.enable_double_dqn: 
                best_actions_from_policy = policy_dqn(new_states).argmax(dim=1)
                target_q = rewards + (1-terminations) * self.discount_factor_g * target_dqn(new_states).gather(dim=1, index=best_actions_from_policy.unsqueeze(dim=1)).squeeze()

            else: 
                #Calculate the target Q-values (expected returns) - Regular DQN
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
    # Parse command line inputs 
    parser = argparse.ArgumentParser(description='Train or test model.')
    parser.add_argument('hyperparameters', help='')
    parser.add_argument('--train', help='Training mode', action='store_true')
    args = parser.parse_args()

    dql = Agent(hyperparameter_set=args.hyperparameters)

    if args.train:
        dql.run(is_training=True)
    else: 
        dql.run(is_training=False, render=True)