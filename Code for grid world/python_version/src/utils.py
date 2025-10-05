import numpy as np
import matplotlib.pyplot as plt
import os
from collections import deque
from typing import Optional, Tuple


# =========================
# Policy builders
# =========================

def create_optimal_path_policy(world) -> np.ndarray:
    """
    Build a deterministic policy by back-propagating shortest steps
    from the goal using a reverse BFS over the grid.
    """
    nS = world.num_states
    acts = world.action_space
    pi = np.zeros((nS, len(acts)), dtype=float)

    stay_idx = acts.index((0, 0))
    goal = tuple(world.target_state)

    # Reverse BFS: record a "next-hop" for each state toward the goal
    q = deque([goal])
    next_hop = {goal: None}

    H, W = world.env_size  # (rows, cols)

    while q:
        cur = q.popleft()
        # consider predecessors that could move into `cur`
        for dx, dy in acts:
            if dx == 0 and dy == 0:
                continue
            prev = (cur[0] - dx, cur[1] - dy)
            if 0 <= prev[0] < H and 0 <= prev[1] < W:
                if prev not in world.forbidden_states and prev not in next_hop:
                    next_hop[prev] = cur
                    q.append(prev)

    # Turn the shortest-next-hop map into a deterministic policy
    for s_idx in range(nS):
        s_xy = world.idx_to_state[s_idx]
        if s_xy == goal or s_xy in world.forbidden_states or s_xy not in next_hop:
            pi[s_idx, stay_idx] = 1.0
            continue

        nxt = next_hop[s_xy]
        move = (nxt[0] - s_xy[0], nxt[1] - s_xy[1])
        if move in acts:
            a_idx = acts.index(move)
            pi[s_idx, a_idx] = 1.0
        else:
            pi[s_idx, stay_idx] = 1.0

    return pi


def create_weighted_random_policy(world) -> np.ndarray:
    """
    Build a stationary stochastic policy with fixed action probabilities
    (except terminal/forbidden states that enforce 'stay').
    """
    nS = world.num_states
    acts = world.action_space
    pi = np.zeros((nS, len(acts)), dtype=float)

    # Weight mapping for stochastic policy
    _STOCH_WEIGHTS = {
        (1, 0): 0.7,    # right
        (0, 1): 0.1,    # down
        (0, -1): 0.1,   # up
        (-1, 0): 0.05,  # left
        (0, 0): 0.05    # stay
    }

    base = np.array([_STOCH_WEIGHTS.get(a, 0.0) for a in acts], dtype=float)
    base /= base.sum() if base.sum() > 0 else 1.0

    stay_idx = acts.index((0, 0))
    goal = tuple(world.target_state)

    for s_idx in range(nS):
        s_xy = world.idx_to_state[s_idx]
        if s_xy == goal or s_xy in world.forbidden_states:
            pi[s_idx, stay_idx] = 1.0
        else:
            pi[s_idx, :] = base

    return pi


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


# =========================
# Visualization helpers
# =========================

def plot_policy_table(policy_matrix, env, save_path: Optional[str] = None):
    """
    Visualize policy as a table:
      rows = actions
      cols = states
      cell = probability (2 decimals) or 'X' for forbidden
    No title (per your requirement).
    """
    num_states, num_actions = policy_matrix.shape
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

    fig, ax = plt.subplots(figsize=(max(8, num_states * 0.6), max(3, num_actions * 0.6)))
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

    if save_path:
        fig.canvas.draw()
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)


def print_policy_table(env, policy_matrix, title: str = "Policy"):
    print("\n" + "=" * 40)
    print(title)
    print("=" * 40)
    header = ["state(x,y)"] + [str(a) for a in env.action_space]
    print("\t".join(header))
    for s in range(env.num_states):
        state = env.idx_to_state[s]
        if state in env.forbidden_states:
            continue
        probs = ["{:.2f}".format(p) for p in policy_matrix[s]]
        print(f"{state}\t" + "\t".join(probs))


def _extract_fig_from_env(env):
    """
    Try common places where the GridWorld might keep its figure:
      - env.fig
      - env.canvas.figure
      - fallback: current figure (not recommended, but prevents errors)
    """
    fig = getattr(env, "fig", None)
    if fig is not None:
        return fig
    canvas = getattr(env, "canvas", None)
    if canvas is not None:
        fig2 = getattr(canvas, "figure", None)
        if fig2 is not None:
            return fig2
    return plt.gcf()


def _save_env_figure(env, path: str, dpi: int = 200):
    fig = _extract_fig_from_env(env)
    # Ensure pixels are rendered
    try:
        fig.canvas.draw()
    except Exception:
        pass
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"Saved figure: {path}")


# =========================
# Utilities
# =========================

def _print_ndarray(label: str, arr: np.ndarray, precision: int = 2) -> None:
    with np.printoptions(precision=precision, suppress=True):
        print(f"\n--- {label} ---\n{arr}")


def _save_or_show(path: Optional[str] = None) -> None:
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved figure: {path}")
    else:
        plt.show()
    plt.close()


def add_state_numbers_to_env(env):
    """
    Add state numbers to the grid world visualization.
    This function calls the add_state_numbers method of the GridWorld class.
    """
    env.add_state_numbers()


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