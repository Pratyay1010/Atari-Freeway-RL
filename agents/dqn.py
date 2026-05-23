import gymnasium as gym
import torch
import numpy as np
import ale_py
import cv2
from torch.utils.tensorboard import SummaryWriter
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
from config import CONFIG
from collections import deque
import random


class ReplayBuffer:
    def __init__(self, buffer_size=100000):
        self.buffer = deque(maxlen=buffer_size)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


def preprocess_frame(frame, obs_type=CONFIG["OBS_TYPE"]):
    """
    Preprocess frame based on observation type
    """
    if obs_type == "rgb":
        # Convert RGB to grayscale and resize
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (84, 84))
    elif obs_type == "grayscale":
        # Already grayscale, just resize
        frame = cv2.resize(frame, (84, 84))
    else:
        raise ValueError(f"Unsupported observation type: {obs_type}")
    
    # Normalize to [0, 1]
    frame = frame.astype(np.float32) / 255.0
    return frame


class FrameStack:
    """Stack multiple frames together for temporal information"""
    def __init__(self, stack_size=4):
        self.stack_size = stack_size
        self.frames = deque(maxlen=stack_size)
    
    def reset(self, frame):
        """Reset with initial frame, duplicated to fill stack"""
        self.frames.clear()
        for _ in range(self.stack_size):
            self.frames.append(frame)
        return np.stack(self.frames, axis=0)  # Shape: (stack_size, 84, 84)
    
    def append(self, frame):
        """Add new frame to stack"""
        self.frames.append(frame)
        return np.stack(self.frames, axis=0)


class CNNPolicy(torch.nn.Module):
    def __init__(self, input_channels: int, num_actions: int):
        super(CNNPolicy, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=4, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1),
            torch.nn.ReLU()
        )
        
        # Calculate the output size of convolutional layers
        conv_out_size = self._get_conv_out((input_channels, 84, 84))
        
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(conv_out_size, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, num_actions)
        )
        
    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))
        
    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)


class FreewayAgent:
    def __init__(self, env):
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Frame stacking
        self.stack_size = 4
        self.frame_stack = FrameStack(stack_size=self.stack_size)
        
        # Model parameters
        self.action_space_dim = env.action_space.n
        self.model = CNNPolicy(self.stack_size, self.action_space_dim).to(self.device)
        self.target_model = CNNPolicy(self.stack_size, self.action_space_dim).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())

        # Optimizer and loss
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.loss_fn = torch.nn.MSELoss()

        # Hyperparameters
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995

        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size=100000)
        self.batch_size = 64
        self.tau = 0.005  # soft update factor

    def get_action(self, obs):
        # Preprocess the frame
        processed_frame = preprocess_frame(obs, CONFIG["OBS_TYPE"])
        
        # Get current state (stacked frames)
        if not hasattr(self, 'current_state'):
            # First frame, initialize stack
            state = self.frame_stack.reset(processed_frame)
            self.current_state = state
        else:
            # Update stack with new frame
            state = self.frame_stack.append(processed_frame)
        
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_space_dim)
        else:
            with torch.no_grad():
                # Convert to tensor and add batch dimension
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
                q_values = self.model(state_tensor)
            return q_values.argmax().item()
        
    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        batch = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to tensors
        states = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)

        # Current Q values
        q_values = self.model(states).gather(1, actions)

        # Target Q values
        with torch.no_grad():
            next_q_values = self.target_model(next_states).max(1)[0].unsqueeze(1)
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # Compute loss
        loss = self.loss_fn(q_values, target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        # Soft update target network
        for target_param, local_param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def train(self, num_episodes: int, save_path: str = CONFIG["ENTRY_NUMBER"]):
        save_path += ".pth"
        print("Starting DQN training with CNN...")
        writer = SummaryWriter(CONFIG["ENTRY_NUMBER"])

        for episode in range(num_episodes):
            obs, _ = self.env.reset()
            
            # Reset frame stack for new episode
            processed_frame = preprocess_frame(obs, CONFIG["OBS_TYPE"])
            current_state = self.frame_stack.reset(processed_frame)
            
            done = False
            total_reward = 0
            action_counts = torch.zeros(self.action_space_dim)

            while not done:
                action = self.get_action(obs)
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                
                # Preprocess next frame
                next_processed_frame = preprocess_frame(next_obs, CONFIG["OBS_TYPE"])
                next_state = self.frame_stack.append(next_processed_frame)
                
                done = terminated or truncated
                
                # Store transition in replay buffer
                self.replay_buffer.push(current_state, action, reward, next_state, done)
                
                # Update current state
                current_state = next_state
                
                # Train the network
                self.update()
                
                total_reward += reward
                action_counts[action] += 1
                obs = next_obs  # Keep raw obs for get_action

            # TensorBoard logging
            writer.add_scalar('Total Reward per Episode', total_reward, episode)
            if episode % 10 == 0:
                # Convert the last frame to tensor for logging
                last_frame = preprocess_frame(obs, CONFIG["OBS_TYPE"])
                obs_tensor = torch.tensor(last_frame, dtype=torch.float32).unsqueeze(0)
                writer.add_image('Last Frame', obs_tensor, episode)

            writer.add_scalar("Epsilon", self.epsilon, episode)
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
            
            writer.add_histogram("Action Distribution", action_counts, episode)
            print(f"Episode {episode+1}/{num_episodes}, Reward={total_reward}")

        print(f"Training finished. Saving model to '{save_path}'...")
        writer.close()
        torch.save(self.model.state_dict(), save_path)

    def load_model(self, path: str):
        try:
            self.model.load_state_dict(torch.load(path))
            self.model.eval()
            print(f"Successfully loaded model from '{path}'")
            return True
        except FileNotFoundError:
            print(f"Error: Model file '{path}' not found.")
            return False


if __name__ == "__main__":
    gym.register_envs(ale_py)

    run_dir = "runs"

    env = gym.make("ALE/Freeway-v5", obs_type=CONFIG["OBS_TYPE"], render_mode="rgb_array")
    
    env = RecordVideo(
        env,
        video_folder=run_dir + "/videos_dqn",
        name_prefix="train",
        episode_trigger=lambda ep: ep % 10 == 0  # record every 10th episode
    )

    env = RecordEpisodeStatistics(env, buffer_length=100)

    agent = FreewayAgent(env)
    agent.train(num_episodes=1000)  # Increased from 100 to 1000 episodes

    env.close()