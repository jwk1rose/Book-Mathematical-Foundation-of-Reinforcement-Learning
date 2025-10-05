# examples/example_run_policies.py
import sys
sys.path.append("..")
import numpy as np
import random
import matplotlib.pyplot as plt

from src.grid_world import GridWorld
from examples.arguments import args   # uses the updated arguments file
from src.utils import build_deterministic_policy, build_stochastic_policy, simulate_policy, plot_policy_table, print_policy_table

# Fix random seed for reproducibility
SEED = 123
random.seed(SEED)
np.random.seed(SEED)

def main():
    # Create two separate envs so their canvases/plots don't overwrite each other
    env_det = GridWorld()   # uses defaults from args
    env_sto = GridWorld()

    gamma = args.discount_rate

    # Build policies
    policy_det = build_deterministic_policy(env_det)
    policy_sto = build_stochastic_policy(env_sto)

    # Print policy arrays/tables
    print_policy_table(env_det, policy_det, title="Deterministic Policy (数组表示)")
    print_policy_table(env_sto, policy_sto, title="Stochastic Policy (数组表示)")

    plot_policy_table(policy_det, env_det, title="Deterministic Policy (action vs state)")
    plot_policy_table(policy_sto, env_sto, title="Stochastic Policy (action vs state)")

    # Choose arbitrary starting state (you can change it)
    start_state = tuple(args.start_state)
    print("\nChosen start state:", start_state)

    # Simulate deterministic policy
    print("\n--- Simulating Deterministic Policy for 50 steps ---")
    traj_det, rewards_det, G_det = simulate_policy(env_det, policy_det, start_state=start_state, steps=50, gamma=gamma, figure_title="Deterministic Policy")
    print("Deterministic trajectory (first 10 states):", traj_det[:10])
    print("Deterministic discounted return G =", G_det)

    # Simulate stochastic policy (re-seed for reproducibility if desired)
    np.random.seed(SEED + 1)
    random.seed(SEED + 1)
    print("\n--- Simulating Stochastic Policy for 50 steps ---")
    traj_sto, rewards_sto, G_sto = simulate_policy(env_sto, policy_sto, start_state=start_state, steps=50, gamma=gamma, figure_title="Stochastic Policy")
    print("Stochastic trajectory (first 10 states):", traj_sto[:10])
    print("Stochastic discounted return G =", G_sto)

    # For convenience, save the final figures to files
    try:
        plt.figure(env_det.canvas.number)
        plt.suptitle("Deterministic Policy + Trajectory")
        plt.savefig("deterministic_policy_trajectory.png", dpi=200)
        plt.figure(env_sto.canvas.number)
        plt.suptitle("Stochastic Policy + Trajectory")
        plt.savefig("stochastic_policy_trajectory.png", dpi=200)
        print("\nSaved figures: deterministic_policy_trajectory.png, stochastic_policy_trajectory.png")
    except Exception as e:
        print("Warning: failed to save figures automatically:", e)

    # Also print summary
    print("\n\nSummary:")
    print(f"Discount rate (gamma): {gamma}")
    print(f"Deterministic discounted return (50-step trajectory): {G_det}")
    print(f"Stochastic discounted return (50-step trajectory): {G_sto}")

if __name__ == "__main__":
    main()
