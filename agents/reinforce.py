import gymnasium as gym
import torch
import numpy as np
import ale_py
from torch.utils.tensorboard import SummaryWriter
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
from config import CONFIG


class Policy(torch.nn.Module):
    """
    A simple policy network for the Freeway-v5 environment.
    """

    def __init__(self, observation_space_dim: int, action_space_dim: int):
        """
        Initializes the model's layers.
        """
        super(Policy, self).__init__()
        # Define your policy network here using torch.
        # This is a placeholder; you'll need to define your network layers.
        # Example for a simple feedforward network:
        # self.layer1 = torch.nn.Linear(observation_space_dim, 128)
        # self.layer2 = torch.nn.Linear(128, action_space_dim)
        self.flatten = torch.nn.Flatten()
        self.fc1 = torch.nn.Linear(observation_space_dim, 128)
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3 = torch.nn.Linear(64, 32)
        self.fc4 = torch.nn.Linear(32, action_space_dim)


    def forward(self, state):
        """
        Performs the forward pass of the network.
        """
        # Define your forward function.
        # The current implementation returns a placeholder value.
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)


class FreewayAgent:
    """
    An agent for the Freeway-v5 environment that contains the model and training logic.
    """

    def __init__(self, env):
        """
        Initializes the agent with a model, optimizer, and hyperparameters.
        """
        self.env = env
        self.observation_space_dim = np.prod(env.observation_space.shape)
        self.action_space_dim = env.action_space.n

        # Policy network
        self.model = Policy(self.observation_space_dim, self.action_space_dim)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.gamma = 0.99
        self.epsilon = 1.0         # start fully exploratory
        self.epsilon_min = 0.05    # don't go below this
        self.epsilon_decay = 0.995 # decay per episode
    
    def get_action(self, obs):
        """
        Chooses an action using epsilon-greedy on top of the policy distribution.
        """
        # Preprocess observation
        obs = obs.astype(np.float32) / 255.0
        processed_obs = torch.tensor(obs.flatten(), dtype=torch.float32)

        # Policy forward pass
        logits = self.model.forward(processed_obs)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)

        if np.random.rand() < self.epsilon:
            # Random action (exploration)
            action = np.random.randint(self.action_space_dim)
            # Log-prob under the policy (not the epsilon-mixture); add eps to avoid log(0)
            log_prob = torch.log(probs[action] + 1e-8)
        else:
            # Policy action (exploitation)
            action_tensor = dist.sample()
            action = action_tensor.item()
            log_prob = dist.log_prob(action_tensor)

        return action, log_prob


    def train(self, num_episodes: int, save_path: str = CONFIG["ENTRY_NUMBER"]):
        save_path += ".pth"
        print("Starting REINFORCE training...")
        writer = SummaryWriter(CONFIG["ENTRY_NUMBER"])

        for episode in range(num_episodes):
            obs, _ = self.env.reset()
            done = False
            total_reward = 0

            log_probs, rewards = [], []
            action_counts = torch.zeros(self.action_space_dim)

            while not done:
                action, log_prob = self.get_action(obs)
                next_obs, reward, terminated, truncated, info = self.env.step(action)

                log_probs.append(log_prob)
                rewards.append(reward)
                action_counts[action] += 1

                total_reward += reward
                done = terminated or truncated
                obs = next_obs

            # --- REINFORCE update (once per episode) ---
            returns = []
            G = 0
            for r in reversed(rewards):
                G = r + self.gamma * G
                returns.insert(0, G)
            returns = torch.tensor(returns)
            # Normalize to reduce variance
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
            
            # compute discounted returns
            policy_loss = []
            for log_prob, G in zip(log_probs, returns):
                policy_loss.append(-log_prob * G)
            
            policy_loss = torch.stack(policy_loss).sum()
            
            # Update policy
            self.optimizer.zero_grad()
            policy_loss.backward()
            # Add gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            # -------------------------------------------

            # --- TensorBoard logging ---
            writer.add_scalar('Total Reward per Episode', total_reward, episode)
            if episode % 10 == 0:
                obs_tensor = torch.tensor(obs, dtype=torch.uint8).permute(2, 0, 1)
                writer.add_image('Last Frame', obs_tensor, episode)

            writer.add_scalar("Epsilon", self.epsilon, episode)
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
            
            writer.add_histogram("Action Distribution", action_counts, episode)
            print("Episode {}/{} | Reward={} | Loss={:.4f}".format(episode + 1, num_episodes, total_reward, policy_loss.item()))

        print("Training finished. Saving model to '{}'...".format(save_path))
        writer.close()
        torch.save(self.model.state_dict(), save_path)

    def load_model(self, path: str):
        try:
            self.model.load_state_dict(torch.load(path))
            self.model.eval()
            print("Successfully loaded model from '{}'".format(path))
            return True
        except FileNotFoundError:
            print("Error: Model file '{}' not found.".format(path))
            return False


if __name__ == "__main__":
    # Ensure you have a 'config.py' file with a 'CONFIG' dictionary and 'MODEL_NAME' and 'OBS_TYPE' keys.
    # Example config.py:
    # CONFIG = {
    #   "MODEL_NAME": "2024JRB1234.pth",
    #   "OBS_TYPE": "rgb"
    # }

    gym.register_envs(ale_py)

    run_dir = "runs"

    env = gym.make("ALE/Freeway-v5", obs_type=CONFIG["OBS_TYPE"], render_mode="rgb_array")
    
    env = RecordVideo(
        env,
        video_folder=run_dir + "/videos",
        name_prefix="train",
        episode_trigger=lambda ep: ep % 1 == 0  # record every episode
    )

    env = RecordEpisodeStatistics(env, buffer_length=100)

    agent = FreewayAgent(env)
    agent.train(num_episodes=100)

    env.close()