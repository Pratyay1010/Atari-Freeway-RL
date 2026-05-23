import gymnasium as gym
import torch
import numpy as np
import ale_py
from torch.utils.tensorboard import SummaryWriter
from configs.default import CONFIG
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
from gymnasium.wrappers import FrameStackObservation
from gymnasium.wrappers import GrayscaleObservation, ResizeObservation
import os

class Policy(torch.nn.Module):
    """
    A simple policy network for the Freeway-v5 environment.
    """

    def __init__(self, action_space_dim: int):
        """
        Initializes the model's layers.
        """
        super(Policy, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(4, 16, 8, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 32, 4, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, 3, stride=1),
            torch.nn.ReLU(),
        )
        self.flatten = torch.nn.Flatten()
        
        with torch.no_grad():
            dummy = torch.zeros(1, 4, 84, 84)
            conv_out = self.conv(dummy)
            conv_out_size = conv_out.view(1, -1).size(1)
        
        self.fc1 = torch.nn.Linear(conv_out_size, 512)
        self.fc2 = torch.nn.Linear(512, 256)
        
        # Heads
        self.actor = torch.nn.Linear(256, action_space_dim)
        self.critic = torch.nn.Linear(256, 1)
        
    def forward(self, x):
        # x = [B,C,H,W] normalized image
        x = x / 255.0
        x = self.conv(x)
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))         
        logits = self.actor(x)
        value = self.critic(x)
        return logits, value


class FreewayAgent:
    def __init__(self, env, lr=3e-4, gamma=0.99, entropy_coef=0.01, value_coef=0.5, 
                 rollout_len=5, clip_grad_norm=0.5):
        
        self.env = env
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.rollout_len = rollout_len
        self.clip_grad_norm = clip_grad_norm
        
        self.action_space_dim = env.action_space.n
        self.model = Policy(self.action_space_dim)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, eps=1e-5)    
    
    def get_action(self, obs):
        """
        Process observation and return action, log probability, and value.
        """        
        # Convert numpy array to tensor and add batch dimension
        # obs shape is (4, 84, 84) which is [C, H, W]
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)  # [1, 4, 84, 84]
        
        logits, value = self.model(obs_tensor)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action.item(), dist.log_prob(action), value
    
    def compute_returns_advantages(self, rewards, values, next_value, done):
        R = next_value if not done else torch.zeros(1)
        returns, advantages = [], []
        for step in reversed(range(len(rewards))):
            R = rewards[step] + self.gamma * R
            advantage = R - values[step]
            returns.insert(0, R)
            advantages.insert(0, advantage)
        return torch.cat(returns), torch.cat(advantages)


    def train(self, num_episodes, save_interval=5, rollout_len=5, save_path: str = CONFIG["RUN_NAME"]):
        save_path += ".pth"
        writer = SummaryWriter(CONFIG["RUN_NAME"])

        best_reward = -float('inf')
        best_model_path = "best_model.pth"
        for episode in range(num_episodes):
            obs, _ = self.env.reset()
            done, total_reward = False, 0

            log_probs, values, rewards = [], [], []

            while not done:
                action, log_prob, value = self.get_action(obs)
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                log_probs.append(log_prob)
                values.append(value)
                rewards.append(torch.tensor([reward], dtype=torch.float32))

                total_reward += reward
                obs = next_obs

                if done or len(rewards) >= rollout_len:
                    # Process the final observation for bootstrapping
                    obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)  # [1, 4, 84, 84]
                    
                    _, next_value = self.model(obs_tensor)
                    returns, advantages = self.compute_returns_advantages(rewards, values, next_value, done)

                    # Losses
                    policy_loss = -(torch.cat(log_probs) * advantages.detach()).mean()
                    value_loss = advantages.pow(2).mean()
                    entropy = -(torch.cat(log_probs).exp() * torch.cat(log_probs)).mean()

                    loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

                    # Optimize
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    log_probs, values, rewards = [], [], []
                
                if total_reward > best_reward:
                    best_reward = total_reward
                    torch.save(self.model.state_dict(), best_model_path)
                    print(f"New best model saved with reward: {total_reward:.2f}")

                if episode % save_interval == 0:
                    checkpoint_path = f"checkpoint_episode_{episode}.pth"
                    torch.save(self.model.state_dict(), checkpoint_path)

                    writer.add_scalar("Reward", total_reward, episode)
            print(f"Episode {episode+1}/{num_episodes}: Reward={total_reward}")

        torch.save(self.model.state_dict(), save_path)
        writer.close()


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
    
    run_dir = "./runs/freeway_reinforce"
    os.makedirs(run_dir, exist_ok=True)

    env = gym.make("ALE/Freeway-v5", render_mode="rgb_array")

    env = ResizeObservation(env, (84, 84))
    env = GrayscaleObservation(env, keep_dim=False)
    env = FrameStackObservation(env, stack_size=4)

    env = RecordEpisodeStatistics(env, buffer_length=1000)
    env = RecordVideo(
        env,
        video_folder=run_dir + "/videos_a2c",   
        name_prefix="train",                     
        episode_trigger=lambda ep: ep % 10 == 0   
    )


    agent = FreewayAgent(
        env,
        lr=1e-4,           # Try: 1e-4, 3e-4, 1e-3
        gamma=0.99,         # Keep at 0.99
        entropy_coef=0.1,  # Try: 0.001, 0.01, 0.1
        value_coef=0.25,     # Try: 0.25, 0.5, 1.0
        rollout_len=20,      # Try: 5, 10, 20
        clip_grad_norm=1.0  # Try: 0.5, 1.0
    )    
    agent.train(num_episodes=100, save_interval=5)
    env.close()