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
            torch.nn.Conv2d(4, 32, 8, stride=4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, 4, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, 3, stride=1),
            torch.nn.ReLU(),
        )
        self.flatten = torch.nn.Flatten()
        
        with torch.no_grad():
            dummy = torch.zeros(1, 4, 210, 160)
            conv_out = self.conv(dummy)
            conv_out_size = conv_out.view(1, -1).size(1)
        
        self.fc = torch.nn.Linear(conv_out_size, 512)

        # Heads
        self.actor = torch.nn.Linear(512, action_space_dim)
        self.critic = torch.nn.Linear(512, 1)
        
    def forward(self, x):
        # x = [B,C,H,W] normalized image
        x = x / 255.0
        x = self.conv(x)
        x = self.flatten(x)
        x = torch.relu(self.fc(x))
        logits = self.actor(x)
        value = self.critic(x)
        return logits, value


class FreewayAgent:
    """
    An agent for the Freeway-v5 environment that contains the model and training logic.
    """

    def __init__(self, env, lr=2.5e-4, gamma=0.99):
        """
        Initializes the agent with a model, optimizer, and hyperparameters.
        """
        self.env = env
        self.gamma = gamma
        self.action_space_dim = env.action_space.n
        self.observation_space_dim = np.prod(env.observation_space.shape)
        
        self.model = Policy(self.action_space_dim)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
    
    def get_action(self, obs):
        """
        Should expect raw observations from the environment and return the action expected in the from by the env.
        This current implementation is a placeholder. A real agent would process the observation
        and use its policy network to decide on an action.
        """

        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)  # [1,H,W,C*stack_size]
        obs_tensor = obs_tensor.permute(0, 3, 1, 2)  # [1, C*stack_size, H, W]

        logits, value = self.model(obs_tensor)
        dist = torch.distributions.Categorical(logits=logits)  # pass logits directly
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


    def train(self, num_episodes, rollout_len=5, save_path: str = CONFIG["ENTRY_NUMBER"]):
        save_path += ".pth"
        writer = SummaryWriter(CONFIG["ENTRY_NUMBER"])

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
                    # Bootstrap with last value if not done
                    _, next_value = self.model(
                        torch.tensor(obs, dtype=torch.float32).permute(2,0,1).unsqueeze(0)
                    )
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
    env = GrayscaleObservation(env, keep_dim=True)
    env = FrameStackObservation(env, stack_size=4)

    env = RecordEpisodeStatistics(env, buffer_length=1000)
    env = RecordVideo(
        env,
        video_folder=run_dir + "/videos_a2c",   
        name_prefix="train",                     
        episode_trigger=lambda ep: ep % 1 == 0   
    )


    agent = FreewayAgent(env)
    agent.train(num_episodes=5)
    
    env.close()
