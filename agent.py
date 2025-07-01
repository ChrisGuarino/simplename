import flappy_bird_gymnasium
import gymnasium
from dqn import DQN
import torch
from experience_replay import ReplayMemory

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Agent: 
    def run(self, is_training=True, render=False ):
        # env = gymnasium.make("FlappyBird-v0", render_mode="human" if render else None, use_lidar=False)
        env = gymnasium.make("CartPole-v1", render_mode="human" if render else None)

        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.n
        policy_dqn = DQN(num_states, num_actions)

        if is_training: 
            memory = ReplayMemory()

        
        state, _ = env.reset()
        while True:
            # Next action:
            # (feed the observation to your agent here)
            action = env.action_space.sample()

            # Processing:
            new_state, reward, terminated, _, info = env.step(action)
            
            if is_training:
                memory.append((state, action, new_state, reward, terminated))
            
            #Move to new state
            state = new_state
            
            # Checking if the player is still alive
            if terminated:
                break

        env.close()