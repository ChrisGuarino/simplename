import flappy_bird_gymnasium
import gymnasium
from dqn import DQN
import torch
from experience_replay import ReplayMemory
import itertools
import yaml
import random

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Agent: 
    def __init__(self,hyperparameter_set):
        with open('hyperparameters.yml','r') as file: 
            all_hyperparameter_sets = yaml.safe_load(file)
            hyperparameters = all_hyperparameter_sets[hyperparameter_set] #Gets hyperparater array from yaml 

        self.replay_memory_size = hyperparameters['replay_memory_size'] # Size of replay memory
        self.mini_batch_size = hyperparameters['mini_batch_size'] # size of training data set sampled from the replay memory
        # These parameters are for Epsilon-Greedy Algo
        self.epsilon_init = hyperparameters['epsilon_init'] # 1 = 100% random actions 
        self.epsilon_decay = hyperparameters['epsilon_decay'] # random decay rate
        self.epsilon_min = hyperparameters['epsilon_min'] # minimum random actions as a percentage. Policy vs random


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

        for episode in itertools.count(): 
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float, device=device)

            terminated = False
            episode_reward = 0.0

            while not terminated: # Checking if the player is still alive
                
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
                    memory.append((state, action, new_state, reward, terminated))
                
                #Move to new state
                state = new_state
            
            # Update episode reward log
            rewards_per_episode.append(episode_reward)

            #Update Epsilon-Greedy and History
            epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
            epsilon_history.append(epsilon)

if __name__ == '__main__': 
    agent = Agent('CartPole1')
    agent.run(is_training=True, render=True)