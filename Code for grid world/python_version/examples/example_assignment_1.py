# examples/example_run_policies.py
import sys
sys.path.append("..")
import numpy as np
import random
import matplotlib.pyplot as plt

from src.grid_world import GridWorld
from examples.arguments import args   # uses the updated arguments file

# Fix random seed for reproducibility
SEED = 123
random.seed(SEED)
np.random.seed(SEED)

from collections import deque

def build_deterministic_policy(env):
    """
    迷宫法确定性策略：
    - 使用 BFS 从每个可达状态找到到 target_state 的最短路径。
    - 策略选择路径上第一步的动作。
    - 对 forbidden 或 target 状态，策略为 stay。
    返回 shape = (num_states, num_actions) 的矩阵。
    """
    num_states = env.num_states
    num_actions = len(env.action_space)
    policy = np.zeros((num_states, num_actions))

    stay_idx = env.action_space.index((0,0))

    # 预计算 BFS 最短路径 parent
    target = tuple(env.target_state)
    queue = deque([target])
    parent = {target: None}

    while queue:
        current = queue.popleft()
        for action in env.action_space:
            if action == (0,0):  # skip stay
                continue
            prev = (current[0] - action[0], current[1] - action[1])
            if (0 <= prev[0] < env.env_size[0] and
                0 <= prev[1] < env.env_size[1] and
                prev not in env.forbidden_states and
                prev not in parent):
                parent[prev] = current
                queue.append(prev)

    # 构建策略
    for s in range(num_states):
        st = env.idx_to_state[s]

        if st in env.forbidden_states or st == target or st not in parent:
            # 禁区/终点/不可达 -> stay
            policy[s, stay_idx] = 1.0
            continue

        # 找到下一步
        next_state = parent[st]
        dx = next_state[0] - st[0]
        dy = next_state[1] - st[1]
        action = (dx, dy)
        if action in env.action_space:
            a_idx = env.action_space.index(action)
            policy[s, a_idx] = 1.0
        else:
            policy[s, stay_idx] = 1.0

    return policy

def build_stochastic_policy(env):
    """
    随机策略示例（偏好右和下）：
    - 对于非终端非禁区： Right 0.5, Down 0.3, Up 0.1, Left 0.05, Stay 0.05
    - 对于禁区或终点： stay 概率为 1
    返回 (num_states, num_actions) 的矩阵
    """
    num_states = env.num_states
    num_actions = len(env.action_space)
    policy = np.zeros((num_states, num_actions))

    # default distribution (right, down, up, left, stay)
    # ensure ordering matches args.action_space: [(0,1),(1,0),(0,-1),(-1,0),(0,0)]
    # That ordering is: down, right, up, left, stay
    # We'll map desired weights accordingly:
    # desired: right 0.5, down 0.3, up 0.1, left 0.05, stay 0.05
    action_order = env.action_space
    weight_map = {}
    weight_map[(1,0)] = 0.5   # right
    weight_map[(0,1)] = 0.3   # down
    weight_map[(0,-1)] = 0.1  # up
    weight_map[(-1,0)] = 0.05 # left
    weight_map[(0,0)] = 0.05  # stay

    for s in range(num_states):
        st = env.idx_to_state[s]
        if st == tuple(env.target_state) or st in env.forbidden_states:
            stay_idx = env.action_space.index((0,0))
            policy[s, stay_idx] = 1.0
        else:
            probs = np.array([weight_map.get(a, 0.0) for a in env.action_space], dtype=float)
            probs = probs / probs.sum()
            policy[s, :] = probs
    return policy

def simulate_policy(env, policy_matrix, start_state=None, steps=50, gamma=0.95, render_final=True, figure_title="Policy run"):
    """
    在环境上按照 policy_matrix 模拟 steps 步（对于随机策略会按概率采样动作）。
    返回：(trajectory_states_list, rewards_list, discounted_return)
    轨迹中的元素使用 env.traj 内部记录的那种方式（包含小扰动的可视化点），但同时保存纯离散状态序列。
    """
    # reset environment to given start
    if start_state is not None:
        env.start_state = start_state
    s, _ = env.reset()  # reset returns (state, {})
    s = tuple(s)

    num_states, num_actions = policy_matrix.shape
    action_space = env.action_space

    traj_states = [s]
    rewards = []

    # For rendering, initialize canvas and draw policy arrows
    env.render()
    env.add_policy(policy_matrix)
    env.render()


    current_state = s
    for t in range(steps):
        idx = env.state_to_idx[current_state]
        action_probs = policy_matrix[idx]
        # choose action: deterministic if one-hot, otherwise sample
        if np.isclose(action_probs.max(), 1.0) and (action_probs.argmax() == action_probs.nonzero()[0][0]):
            action_idx = int(np.argmax(action_probs))
        else:
            action_idx = int(np.random.choice(len(action_probs), p=action_probs))
        action = action_space[action_idx]

        next_state, reward, done, _ = env.step(action)
        next_state = tuple(next_state)
        traj_states.append(next_state)
        rewards.append(reward)

        env.render()  # update viz

        current_state = next_state
        if done:
            # If reached terminal, for the remainder steps we'll simulate 'stay' actions
            # but still record rewards (should be reward_target on first arrival, then reward_step or reward_target depending on env)
            # break   # we DO NOT break so that trajectory length is exactly steps (按作业要求长度固定)
            pass

    # compute discounted return
    G = 0.0
    for t, r in enumerate(rewards):
        G += (gamma ** t) * r

    return traj_states, rewards, G


import matplotlib.pyplot as plt
import numpy as np


def plot_policy_table(policy_matrix, env, title="Policy Table (action vs state)"):
    """
    用表格的形式可视化策略:
    - 行: action
    - 列: state
    - 单元格内容: 概率 (保留两位小数)
    """
    num_states, num_actions = policy_matrix.shape
    data = policy_matrix.T  # 转置: 行=action, 列=state

    # 把数据转成字符串（保留2位小数）
    cell_text = []
    for a_idx, a in enumerate(env.action_space):
        row = []
        for s in range(num_states):
            st = env.idx_to_state[s]
            if st in env.forbidden_states:
                row.append("X")
            else:
                row.append("{:.2f}".format(policy_matrix[s, a_idx]))
        cell_text.append(row)

    # 创建表格
    fig, ax = plt.subplots(figsize=(num_states * 0.6, num_actions * 0.6))
    ax.set_axis_off()

    table = ax.table(
        cellText=cell_text,
        rowLabels=[str(a) for a in env.action_space],
        colLabels=[f"S{j}" for j in range(num_states)],
        loc="center",
        cellLoc="center"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.2, 1.2)

    plt.title(title)
    plt.show()


def print_policy_table(env, policy_matrix, title="Policy"):
    print("\n" + "="*40)
    print(title)
    print("="*40)
    header = ["state(x,y)"] + [str(a) for a in env.action_space]
    print("\t".join(header))
    for s in range(env.num_states):
        state = env.idx_to_state[s]
        if state in env.forbidden_states:
            continue
        probs = ["{:.2f}".format(p) for p in policy_matrix[s]]
        print(f"{state}\t" + "\t".join(probs))


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
