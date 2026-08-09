import matplotlib
matplotlib.use('Agg')  # Headless rendering for offline environments / background processes
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

def plot_progress(log_path="checkpoints/training_logs_v4.csv", output_img="checkpoints/training_progress_v4.png"):
    if not os.path.exists(log_path):
        print(f"Error: Log file '{log_path}' not found yet. Make sure your training loop is running and saving logs.")
        return

    # Load logs
    df = pd.read_csv(log_path)
    if len(df) < 5:
        print("Not enough data points yet. Keep training!")
        return

    # Smooth values using running average for cleaner trends
    window = max(1, len(df) // 50)
    df['critic_smooth'] = df['critic_loss'].rolling(window=window, min_periods=1).mean()
    df['actor_smooth'] = df['actor_loss'].rolling(window=window, min_periods=1).mean()
    df['reward_smooth'] = df['step_reward'].rolling(window=window, min_periods=1).mean()

    # Create figure
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    fig.suptitle("Soft Actor-Critic (SAC) Training Diagnostics\nEnd-to-End Visual Driving Policy Progression", fontsize=14, fontweight='bold', y=0.98)

    # 1. Critic TD Error (MSE Loss)
    axes[0].plot(df['step'], df['critic_loss'], color='blue', alpha=0.15, label='Raw Critic Loss')
    axes[0].plot(df['step'], df['critic_smooth'], color='blue', linewidth=2, label='Smoothed (EMA)')
    axes[0].set_ylabel("Critic Loss (MSE)", fontsize=10, fontweight='bold')
    axes[0].set_title("Critic Loss (Temporal-Difference Error Convergence)", fontsize=11, fontweight='bold', loc='left')
    axes[0].grid(True, linestyle='--', alpha=0.5)
    axes[0].legend(loc='upper right')

    # 2. Actor Policy Loss (Policy Search)
    axes[1].plot(df['step'], df['actor_loss'], color='orange', alpha=0.15, label='Raw Actor Loss')
    axes[1].plot(df['step'], df['actor_smooth'], color='orange', linewidth=2, label='Smoothed (EMA)')
    axes[1].set_ylabel("Actor Loss", fontsize=10, fontweight='bold')
    axes[1].set_title("Actor Loss (Policy Search Progress)", fontsize=11, fontweight='bold', loc='left')
    axes[1].grid(True, linestyle='--', alpha=0.5)
    axes[1].legend(loc='upper right')

    # 3. Step Reward Trend
    axes[2].plot(df['step'], df['step_reward'], color='green', alpha=0.15, label='Raw Reward')
    axes[2].plot(df['step'], df['reward_smooth'], color='green', linewidth=2, label='Smoothed (EMA)')
    axes[2].set_ylabel("Step Reward", fontsize=10, fontweight='bold')
    axes[2].set_xlabel("Global Step", fontsize=10, fontweight='bold')
    axes[2].set_title("Step Reward Trend (Performance & Comfort Growth)", fontsize=11, fontweight='bold', loc='left')
    axes[2].grid(True, linestyle='--', alpha=0.5)
    axes[2].legend(loc='lower right')

    # Cleanup and layout
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_img), exist_ok=True)
    plt.savefig(output_img, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Successfully generated training progress visualization at: {output_img}")

if __name__ == "__main__":
    plot_progress()
