import argparse
import subprocess
import sys

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", type=str, default="dqn")

    args = parser.parse_args()

    if args.agent == "dqn":
        subprocess.run([sys.executable, "-m", "agents.dqn"])

    elif args.agent == "a2c":
        subprocess.run([sys.executable, "-m", "agents.a2c"])

    elif args.agent == "reinforce":
        subprocess.run([sys.executable, "-m", "agents.reinforce"])

    else:
        raise ValueError("Unsupported agent")