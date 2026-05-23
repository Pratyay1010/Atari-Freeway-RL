"""
    Don't Make any changes here, test your train.py code if it works well with this
"""
import gymnasium as gym
import torch
import numpy as np
import pandas as pd
from train import FreewayAgent
from config import CONFIG


# --- Setup ---
MODEL_PATH = CONFIG["ENTRY_NUMBER"]+".pth"
NUM_TEST_EPISODES = 5
REWARDS_CSV_PATH = CONFIG["ENTRY_NUMBER"] + ".csv"

print("Setting up the testing environment...")
print(f"testing {MODEL_PATH}")
env = gym.make("ALE/Freeway-v5", obs_type=CONFIG["OBS_TYPE"])

# Create an instance of the FreewayAgent
agent = FreewayAgent(env)

# Load the trained model weights
if not agent.load_model(MODEL_PATH):
    exit()

# --- Testing Loop ---
print(f"Starting evaluation over {NUM_TEST_EPISODES} episodes...")
all_rewards = []
for episode in range(NUM_TEST_EPISODES):
    obs, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        try:
            action = agent.get_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
        except Exception as e:
            print("Crashed during Step with the following error")
            print(e)
            exit(-1)

        done = terminated or truncated
        total_reward += reward
        obs = next_obs

    all_rewards.append(total_reward)
    print(f"Episode {episode + 1}/{NUM_TEST_EPISODES}, Total Reward: {total_reward}")

# --- Save Rewards to CSV ---
df = pd.DataFrame({
    'episode': range(1, NUM_TEST_EPISODES + 1),
    'reward': all_rewards
})
df.to_csv(REWARDS_CSV_PATH, index=False)

print(f"\nSaved episode rewards to {REWARDS_CSV_PATH}")

# --- Results ---

mean_reward = np.mean(all_rewards)
std_dev_reward = np.std(all_rewards)

print("\n--- Final Results ---")
print(f"Mean Reward over {NUM_TEST_EPISODES} episodes: {mean_reward:.2f}")
print(f"Standard Deviation of Reward: {std_dev_reward:.2f}")