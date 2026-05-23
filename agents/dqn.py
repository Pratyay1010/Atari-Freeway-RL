import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import ale_py
import random
import matplotlib.pyplot as plt
from collections import deque
from torch.utils.tensorboard import SummaryWriter
from configs.default import CONFIG


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class Policy(nn.Module):
    def __init__(self, observation_space_shape, action_space_dim):
        super().__init__()

        h, w, c = observation_space_shape

        self.conv = nn.Sequential(
            nn.Conv2d(c, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )

        with torch.no_grad():
            dummy_input = torch.zeros(1, c, h, w)
            conv_out_size = self.conv(dummy_input).view(1, -1).size(1)

        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 256),
            nn.ReLU(),
            nn.Linear(256, action_space_dim)
        )

    def forward(self, state):
        state = state.float() / 255.0
        x = self.conv(state)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class FreewayAgent:
    def __init__(self, env):
        self.env = env

        self.observation_space_shape = env.observation_space.shape
        self.action_space_dim = env.action_space.n

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        print(f"Using device: {self.device}")

        self.gamma = CONFIG["GAMMA"]
        self.learning_rate = CONFIG["LEARNING_RATE"]

        self.epsilon = CONFIG["EPSILON"]
        self.epsilon_min = CONFIG["EPSILON_MIN"]
        self.epsilon_decay = CONFIG["EPSILON_DECAY"]

        self.batch_size = CONFIG["BATCH_SIZE"]
        self.target_update_freq = CONFIG["TARGET_UPDATE_FREQ"]
        self.tau = CONFIG["TAU"]

        self.q_network = Policy(
            self.observation_space_shape,
            self.action_space_dim
        ).to(self.device)

        self.target_network = Policy(
            self.observation_space_shape,
            self.action_space_dim
        ).to(self.device)

        self.target_network.load_state_dict(
            self.q_network.state_dict()
        )

        self.target_network.eval()

        self.optimizer = optim.Adam(
            self.q_network.parameters(),
            lr=self.learning_rate
        )

        self.criterion = nn.MSELoss()

        self.replay_buffer = ReplayBuffer(
            CONFIG["BUFFER_SIZE"]
        )

        self.steps_done = 0
        self.episode_rewards = []

    def preprocess_obs(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float32)

        obs = obs.permute(2, 0, 1)

        return obs.to(self.device)

    def get_action(self, obs, training=False):
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_space_dim)

        with torch.no_grad():
            obs = obs.unsqueeze(0)

            q_values = self.q_network(obs)

            return q_values.argmax().item()

    def update_target_network(self):
        target_state_dict = self.target_network.state_dict()
        q_state_dict = self.q_network.state_dict()

        for key in q_state_dict:
            target_state_dict[key] = (
                self.tau * q_state_dict[key]
                + (1 - self.tau) * target_state_dict[key]
            )

        self.target_network.load_state_dict(
            target_state_dict
        )

    def learn(self):
        if len(self.replay_buffer) < self.batch_size:
            return 0

        batch = self.replay_buffer.sample(
            self.batch_size
        )

        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.stack(states).to(self.device)
        next_states = torch.stack(next_states).to(self.device)

        actions = torch.tensor(
            actions,
            dtype=torch.long
        ).unsqueeze(1).to(self.device)

        rewards = torch.tensor(
            rewards,
            dtype=torch.float32
        ).unsqueeze(1).to(self.device)

        dones = torch.tensor(
            dones,
            dtype=torch.float32
        ).unsqueeze(1).to(self.device)

        current_q_values = self.q_network(states).gather(1, actions)

        with torch.no_grad():
            next_q_values = self.target_network(
                next_states
            ).max(1)[0].unsqueeze(1)

            target_q_values = rewards + (
                (1 - dones) * self.gamma * next_q_values
            )

        loss = self.criterion(
            current_q_values,
            target_q_values
        )

        self.optimizer.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.q_network.parameters(),
            1.0
        )

        self.optimizer.step()

        return loss.item()

    def train(self, num_episodes, save_path=CONFIG["RUN_NAME"]):
        save_path += ".pth"

        print("Starting DQN training...")

        writer = SummaryWriter(CONFIG["RUN_NAME"])

        best_reward = float("-inf")

        recent_rewards = deque(maxlen=100)

        for episode in range(num_episodes):
            obs, _ = self.env.reset()

            obs = self.preprocess_obs(obs)

            done = False
            total_reward = 0
            episode_loss = 0
            steps_in_episode = 0

            while not done:
                action = self.get_action(
                    obs,
                    training=True
                )

                next_obs, reward, terminated, truncated, _ = self.env.step(action)

                done = terminated or truncated

                next_obs = self.preprocess_obs(next_obs)

                self.replay_buffer.push(
                    obs,
                    action,
                    reward,
                    next_obs,
                    done
                )

                if len(self.replay_buffer) >= self.batch_size:
                    loss = self.learn()

                    episode_loss += loss

                    self.update_target_network()

                total_reward += reward
                steps_in_episode += 1
                self.steps_done += 1

                obs = next_obs

            self.episode_rewards.append(total_reward)

            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

            recent_rewards.append(total_reward)

            avg_recent_reward = (
                np.mean(recent_rewards)
                if recent_rewards else 0
            )

            if total_reward > best_reward:
                best_reward = total_reward

                torch.save(
                    self.q_network.state_dict(),
                    save_path
                )

                print(
                    f"New best model saved "
                    f"with reward: {best_reward}"
                )

            if episode_loss > 0:
                avg_loss = episode_loss / steps_in_episode

                writer.add_scalar(
                    "Loss/Episode",
                    avg_loss,
                    episode
                )

            writer.add_scalar(
                "Reward/Episode",
                total_reward,
                episode
            )

            writer.add_scalar(
                "Epsilon",
                self.epsilon,
                episode
            )

            writer.add_scalar(
                "Reward/Recent_Average",
                avg_recent_reward,
                episode
            )

            if episode % 50 == 0:
                writer.add_image(
                    "Last_Frame",
                    obs.detach().cpu(),
                    episode
                )

            print(
                f"Episode {episode + 1}/{num_episodes} | "
                f"Reward: {total_reward} | "
                f"Avg Reward: {avg_recent_reward:.2f} | "
                f"Epsilon: {self.epsilon:.3f}"
            )

        self.plot_rewards()

        print(
            f"Training finished. "
            f"Best model saved to '{save_path}'"
        )

        writer.close()

    def plot_rewards(self):
        plt.figure(figsize=(10, 5))

        plt.plot(
            self.episode_rewards,
            label="Episode Reward"
        )

        window_size = 50

        if len(self.episode_rewards) >= window_size:
            moving_avg = np.convolve(
                self.episode_rewards,
                np.ones(window_size) / window_size,
                mode="valid"
            )

            plt.plot(
                range(window_size - 1, len(self.episode_rewards)),
                moving_avg,
                label=f"Moving Average ({window_size})",
                color="red"
            )

        plt.title("DQN Training Rewards")
        plt.xlabel("Episode")
        plt.ylabel("Reward")

        plt.grid(True)
        plt.legend()

        plt.savefig(
            f'{CONFIG["RUN_NAME"]}_training_rewards.png'
        )

        plt.close()

    def load_model(self, path):
        try:
            self.q_network.load_state_dict(
                torch.load(path, map_location=self.device)
            )

            self.q_network.eval()

            self.epsilon = 0.0

            print(
                f"Successfully loaded model from '{path}'"
            )

            return True

        except FileNotFoundError:
            print(
                f"Error: Model file '{path}' not found."
            )

            return False

        except Exception as e:
            print(f"Error loading model: {e}")

            return False

    @property
    def model(self):
        return self.q_network


if __name__ == "__main__":
    gym.register_envs(ale_py)

    env = gym.make(
        "ALE/Freeway-v5",
        obs_type=CONFIG["OBS_TYPE"]
    )

    agent = FreewayAgent(env)

    agent.train(
        num_episodes=CONFIG["TOTAL_EPS"]
    )