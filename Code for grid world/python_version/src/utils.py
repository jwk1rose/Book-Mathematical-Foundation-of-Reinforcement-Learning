import numpy as np
import matplotlib.pyplot as plt
import os
from collections import deque
from typing import Optional, Tuple, List, Dict

# 导入GridWorld类
from src.grid_world import GridWorld

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


def compute_absorbing_optimal_values(env) -> Dict[Tuple[int, int], float]:
    """
    计算在“到达目标后立即终止且不再累积奖励”的设定下的参考最优状态值。

    设定：
    - 每步移动成本为 `env.reward_step`（此处通常为0）；
    - 到达目标的当步获得 `env.reward_target`；之后 episode 终止；
    - 禁止区域不可达（其值不计算）。

    在该设定且转移确定的情况下，若某状态可达目标，则其最优回报为一次性目标奖励 `env.reward_target`。
    因此返回：非禁止且非目标的状态值为 `env.reward_target`，目标状态值为 0。
    """
    V_opt: Dict[Tuple[int, int], float] = {}
    for s in env.valid_states:
        if s == tuple(env.target_state):
            V_opt[s] = 0.0
        else:
            V_opt[s] = float(env.reward_target)
    return V_opt


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
        try:
            plt.savefig(path, dpi=300, bbox_inches="tight")
            print(f"Saved figure: {path}")
        except Exception as e:
            print(f"Error saving figure: {e}")
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


# =========================
# Value Iteration Algorithm
# =========================

def value_iteration(states, actions, p_r, p_s, gamma, epsilon, max_iterations=1000):
    """
    执行价值迭代算法。
    
    算法步骤:
    1. 初始化值函数 v0
    2. 当值函数未收敛时，进行第k次迭代:
       对于每个状态 s ∈ S:
         对于每个动作 a ∈ A(s):
           计算 q-value: qk(s, a) = Σs' p(s'|s,a) [r(s,a,s') + gamma * v(s')]
         选择最优动作: a*k(s) = arg maxa qk(s, a)
         策略更新: πk+1(a|s) = 1 if a = a*k, else 0
         值函数更新: vk+1(s) = maxa qk(s, a)

    参数:
    - states (list): 所有状态的列表。
    - actions (list): 所有动作的列表。
    - p_r (function): 返回给定状态和动作的奖励的函数。
                      p_r(s, a, s_prime) 应该返回从状态s通过动作a转移到状态s_prime时的奖励。
    - p_s (function): 返回给定状态和动作的转移概率的函数。
                      p_s(s, a, s_prime) 应该返回从状态s通过动作a转移到状态s_prime的概率。
    - gamma (float): 折扣因子。
    - epsilon (float): 收敛阈值(当值更新小于epsilon时,算法停止)。
    - max_iterations (int): 最大迭代次数。

    返回:
    - v (dict): 每个状态的最优状态值。
    - policy (dict): 每个状态的最优策略。
    - iteration_count (int): 执行的迭代次数。
    - v_history (list): 用于可视化的值函数历史记录。
    - policy_history (list): 用于可视化的策略历史记录。
    """
    # 初始化值函数 v0
    v = {s: 0.0 for s in states}
    v_history = [v.copy()]  # 记录初始值函数
    
    # 初始化策略历史记录
    policy_history = []

    # 初始化策略（可以是任何初始动作）
    policy = {s: actions[0] if actions else None for s in states}

    iteration_count = 0
    for iteration in range(max_iterations):
        iteration_count = iteration + 1
        delta = 0
        # 复制当前值用于值更新计算
        v_new = v.copy()
        
        # 用于记录当前迭代的策略
        policy_new = {}

        for s in states:
            # 对于每个状态，计算每个动作的q值
            q_values = {}
            for a in actions:
                # 计算q值: qk(s, a) = Σs' p(s'|s,a) [r(s,a,s') + gamma * v(s')]
                q_value = 0
                for s_prime in states:
                    transition_prob = p_s(s, a, s_prime)
                    if transition_prob > 0:
                        reward = p_r(s, a, s_prime)
                        q_value += transition_prob * (reward + gamma * v[s_prime])
                q_values[a] = q_value

            # 更新值函数
            if q_values:
                v_new[s] = max(q_values.values())
                # 更新状态s的策略
                policy_new[s] = max(q_values, key=q_values.get)
            else:
                v_new[s] = 0
                policy_new[s] = actions[0] if actions else None

            # 跟踪值函数的最大变化以检查收敛性
            delta = max(delta, abs(v_new[s] - v[s]))

        # 更新下一次迭代的值函数
        v = v_new
        v_history.append(v.copy())
        
        # 记录当前迭代的策略历史（转换为矩阵格式）
        policy_matrix = np.zeros((len(states), len(actions)))
        for i, state in enumerate(states):
            best_action = policy_new[state]
            if best_action in actions:
                action_idx = actions.index(best_action)
                policy_matrix[i, action_idx] = 1.0
            # 确保每个状态至少有一个动作（即使概率为0）
            elif len(actions) > 0:
                policy_matrix[i, 0] = 1.0
        policy_history.append(policy_matrix.copy())

        # 如果值的变化小于epsilon，则停止迭代
        if delta < epsilon:
            print(f"价值迭代在 {iteration_count} 次迭代后收敛。")
            # 确保最后一次迭代的策略也被记录
            if len(policy_history) < len(v_history):
                policy_history.append(policy_matrix.copy())
            break
    else:
        print(f"价值迭代在达到最大迭代次数 ({max_iterations}) 后停止。")
        # 确保最后一次迭代的策略也被记录
        if len(policy_history) < len(v_history):
            policy_history.append(policy_matrix.copy())
    
    return v, policy, iteration_count, v_history, policy_history


# =========================
# Visualization Functions
# =========================

def visualize_optimal_policy(env, policy, save_path=None):
    """
    可视化最优策略。
    
    参数:
        env: GridWorld 环境实例
        policy: 最优策略矩阵
        save_path: 保存路径（可选）
    """
    # 创建新的环境实例用于可视化
    vis_env = GridWorld(
        env_size=env.env_size,
        start_state=env.start_state,
        target_state=env.target_state,
        forbidden_states=env.forbidden_states
    )
    vis_env.reset()
    vis_env.render()
    vis_env.add_policy(policy)
    # 策略可视化需要显示状态编号以便理解
    add_state_numbers_to_env(vis_env)
    vis_env.render()
    
    # 确保画布被正确渲染
    try:
        if hasattr(vis_env, 'canvas') and vis_env.canvas is not None:
            if hasattr(vis_env.canvas, 'figure') and vis_env.canvas.figure is not None:
                vis_env.canvas.figure.canvas.draw()
    except Exception as e:
        print(f"Warning: Could not draw canvas: {e}")
        
    _save_or_show(save_path)


def visualize_state_values(env, values, precision=2, save_path=None):
    """
    可视化状态值函数。
    
    参数:
        env: GridWorld 环境实例
        values: 状态值数组
        precision: 值显示精度
        save_path: 保存路径（可选）
    """
    # 创建新的环境实例用于可视化
    vis_env = GridWorld(
        env_size=env.env_size,
        start_state=env.start_state,
        target_state=env.target_state,
        forbidden_states=env.forbidden_states
    )
    vis_env.reset()
    vis_env.render()
    vis_env.add_state_values(values, precision=precision)
    vis_env.render()
    
    # 确保画布被正确渲染
    try:
        if hasattr(vis_env, 'canvas') and vis_env.canvas is not None:
            if hasattr(vis_env.canvas, 'figure') and vis_env.canvas.figure is not None:
                vis_env.canvas.figure.canvas.draw()
    except Exception as e:
        print(f"Warning: Could not draw canvas: {e}")
        
    _save_or_show(save_path)


def visualize_state_values_and_policy(env, values, policy, precision=2, save_path=None):
    """
    可视化状态值函数和最优策略。
    
    参数:
        env: GridWorld 环境实例
        values: 状态值数组
        policy: 策略矩阵
        precision: 值显示精度
        save_path: 保存路径（可选）
    """
    # 创建新的环境实例用于可视化
    vis_env = GridWorld(
        env_size=env.env_size,
        start_state=env.start_state,
        target_state=env.target_state,
        forbidden_states=env.forbidden_states
    )
    vis_env.reset()
    vis_env.render()
    vis_env.add_state_values(values, precision=precision)
    vis_env.add_policy(policy)
    vis_env.render()
    
    # 确保画布被正确渲染
    try:
        if hasattr(vis_env, 'canvas') and vis_env.canvas is not None:
            if hasattr(vis_env.canvas, 'figure') and vis_env.canvas.figure is not None:
                vis_env.canvas.figure.canvas.draw()
    except Exception as e:
        print(f"Warning: Could not draw canvas: {e}")
        
    _save_or_show(save_path)


def visualize_value_evolution(env, V_history, policy_history, gamma, max_figures=10, save_path_prefix=None):
    """
    可视化值函数的演变过程。
    
    参数:
        env: GridWorld 环境实例
        V_history: 值函数历史记录
        policy_history: 策略历史记录
        gamma: 折扣因子
        max_figures: 最大显示图像数
        save_path_prefix: 保存路径前缀（可选）
    """
    total_iterations = len(V_history)
    
    # 确定要显示的迭代索引
    if total_iterations <= max_figures:
        indices = list(range(total_iterations))
    else:
        # 显示前5个和后5个
        first_indices = list(range(5))
        last_indices = list(range(total_iterations - 5, total_iterations))
        indices = first_indices + last_indices
    
    # 为每个选定的迭代创建可视化
    for i, idx in enumerate(indices):
        # 将字典格式的值函数转换为数组格式
        V_dict = V_history[idx]
        values = np.array([V_dict[env.idx_to_state[j]] for j in range(len(env.idx_to_state))])
        
        title = f"Value Function at Iteration {idx}"
        
        if save_path_prefix:
            save_path = f"{save_path_prefix}_iter_{idx}.png"
        else:
            save_path = None
        
        # 获取对应的策略（如果存在）
        if idx < len(policy_history):
            policy = policy_history[idx]
        else:
            policy = None
        
        # 创建新的环境实例用于可视化
        vis_env = GridWorld(
            env_size=env.env_size,
            start_state=env.start_state,
            target_state=env.target_state,
            forbidden_states=env.forbidden_states
        )
        vis_env.reset()
        vis_env.render()
        vis_env.add_state_values(values, precision=2)
        if policy is not None:
            vis_env.add_policy(policy)
        vis_env.render()
        
        # 确保画布被正确渲染
        try:
            if hasattr(vis_env, 'canvas') and vis_env.canvas is not None:
                if hasattr(vis_env.canvas, 'figure') and vis_env.canvas.figure is not None:
                    vis_env.canvas.figure.canvas.draw()
        except Exception as e:
            print(f"Warning: Could not draw canvas: {e}")
        
        _save_or_show(save_path)
        
        # 打印当前迭代的值函数
        print(f"\n--- {title} ---")
        for s in range(len(values)):
            state_xy = env.idx_to_state[s]
            print(f"State {state_xy}: {values[s]:.2f}")


# =========================
# Probability and Reward Functions
# =========================

def create_probability_and_reward_functions_from_env(env):
    """
    从网格世界环境创建概率转移函数和奖励函数。
    
    参数:
        env: GridWorld 环境实例
    
    返回:
        p_r: 奖励函数 p_r(s, a, s_prime) -> reward
        p_s: 概率转移函数 p_s(s, a, s_prime) -> probability
    """
    def p_r(s, a, s_prime):
        """
        奖励函数。
        在网格世界中，奖励是基于当前状态和动作确定的。
        """
        # 获取执行动作后的下一个状态和奖励
        next_state, reward = env._get_next_state_and_reward(s, a)
        # 如果s_prime是实际的下一个状态，则返回奖励，否则返回0
        if next_state == s_prime:
            return reward
        else:
            return 0
    
    def p_s(s, a, s_prime):
        """
        概率转移函数。
        在网格世界中，转移是确定性的。
        """
        # 获取执行动作后的下一个状态
        next_state, _ = env._get_next_state_and_reward(s, a)
        # 如果s_prime是实际的下一个状态，则概率为1.0，否则为0.0
        if next_state == s_prime:
            return 1.0
        else:
            return 0.0
    
    return p_r, p_s


def sample_probability_and_reward_functions(env, num_samples=1000):
    """
    通过采样从环境创建概率转移函数和奖励函数的近似版本。
    
    参数:
        env: GridWorld 环境实例
        num_samples: 采样次数
    
    返回:
        p_r: 采样的奖励函数
        p_s: 采样的概率转移函数
    """
    # 初始化计数器
    transition_counts = {}  # (s, a, s_prime) -> count
    reward_sums = {}        # (s, a, s_prime) -> sum of rewards
    state_action_counts = {}  # (s, a) -> count
    
    # 获取所有状态和动作
    states = list(env.idx_to_state.values())
    actions = env.action_space
    
    # 保存环境的原始起始状态
    original_start_state = env.start_state
    
    # 进行采样
    for _ in range(num_samples):
        # 随机选择一个状态作为起始状态
        state = states[np.random.randint(len(states))]
        
        # 随机选择一个动作
        action = actions[np.random.randint(len(actions))]
        
        # 计算执行动作后的下一个状态和奖励（不修改环境状态）
        next_state, reward = env._get_next_state_and_reward(state, action)
        
        # 更新计数器
        state_action_key = (state, action)
        transition_key = (state, action, next_state)
        
        state_action_counts[state_action_key] = state_action_counts.get(state_action_key, 0) + 1
        transition_counts[transition_key] = transition_counts.get(transition_key, 0) + 1
        reward_sums[transition_key] = reward_sums.get(transition_key, 0) + reward
    
    def p_r(s, a, s_prime):
        """
        采样的奖励函数。
        """
        transition_key = (s, a, s_prime)
        if transition_key in reward_sums and transition_key in transition_counts:
            return reward_sums[transition_key] / transition_counts[transition_key]
        else:
            return 0
    
    def p_s(s, a, s_prime):
        """
        采样的概率转移函数。
        """
        transition_key = (s, a, s_prime)
        state_action_key = (s, a)
        
        if state_action_key in state_action_counts and transition_key in transition_counts:
            return transition_counts[transition_key] / state_action_counts[state_action_key]
        else:
            return 0
    
    # 恢复环境的原始起始状态
    env.start_state = original_start_state
    
    return p_r, p_s


# =========================
# Q-learning Algorithm (Off-policy)
# =========================

def q_learning(
    env,
    behavior_policy_matrix,
    num_episodes=1000,
    alpha=0.1,
    gamma=0.9,
    max_steps=100,
    record_freq=50
):
    """
    Q-learning算法（离策略版本）
    
    算法步骤:
    1. 初始化Q(s,a)和目标策略π_T
    2. 对于每个episode:
       a) 使用行为策略π_b生成episode {s0, a0, r1, s1, a1, r2, ...}
       b) 对于episode中的每一步t:
          - 更新q值: q_{t+1}(s_t, a_t) = q_t(s_t, a_t) - α_t(s_t, a_t)[q_t(s_t, a_t) - (r_{t+1} + γ max_a q_t(s_{t+1}, a))]
          - 更新目标策略: π_{T,t+1}(a|s_t) = 1 if a = argmax_a q_{t+1}(s_t, a), else 0
    
    参数:
        env: GridWorld环境实例
        behavior_policy_matrix: 行为策略矩阵 (num_states, num_actions)
        num_episodes: episode数量
        alpha: 学习率
        gamma: 折扣因子
        max_steps: 每个episode的最大步数
        record_freq: 记录频率
    
    返回:
        Q: 动作值函数字典 {(state, action): value}
        target_policy_matrix: 最终目标策略矩阵
        policy_history: 策略历史记录（用于可视化）
        value_history: 值函数历史记录（用于可视化）
        episode_data: episode数据列表 [(states, actions, rewards), ...]
        state_value_errors: 状态值误差历史
    """
    num_states = env.num_states
    num_actions = len(env.action_space)
    
    # 初始化Q值
    Q = {}
    for s_idx in range(num_states):
        state = env.idx_to_state[s_idx]
        for action in env.action_space:
            Q[(state, action)] = 0.0
    
    # 初始化目标策略为确定性策略（选择Q值最大的动作）
    target_policy_matrix = np.zeros((num_states, num_actions))
    for s_idx in range(num_states):
        target_policy_matrix[s_idx, 0] = 1.0  # 初始化为第一个动作
    
    # 历史记录
    policy_history = []
    value_history = []
    episode_data = []  # 记录所有episode的数据
    state_value_errors = []  # 记录状态值误差
    actual_trajectory = {'states': [], 'actions': [], 'rewards': []}  # 记录实际训练轨迹
    
    # 保存原始起始状态
    original_start_state = env.start_state
    
    # 全局步数计数器
    global_step = 0
    
    for episode_idx in range(num_episodes):
        # 重置环境
        state, _ = env.reset()
        state = tuple(state)
        
        # 记录当前episode的轨迹
        episode_states = [state]
        episode_actions = []
        episode_rewards = []
        
        # 生成一个episode并同时进行Q值更新
        for step in range(max_steps):
            global_step += 1
            
            # 使用行为策略选择动作
            state_idx = env.state_to_idx[state]
            action_probs = behavior_policy_matrix[state_idx]
            action_idx = np.random.choice(num_actions, p=action_probs)
            action = env.action_space[action_idx]
            
            # 执行动作
            next_state, reward, done, _ = env.step(action)
            next_state = tuple(next_state)
            
            # 记录轨迹
            episode_actions.append(action)
            episode_rewards.append(reward)
            episode_states.append(next_state)
            
            # 记录实际训练轨迹（用于展示）
            actual_trajectory['states'].append(state)
            actual_trajectory['actions'].append(action)
            actual_trajectory['rewards'].append(reward)
            
            # Q-learning更新（使用下一个状态的最大Q值）
            next_state_idx = env.state_to_idx[next_state]
            
            # 如果已经结束，下一个状态的值为0
            if done:
                max_next_q = 0.0
            else:
                max_next_q = max([Q[(next_state, a)] for a in env.action_space])
            
            # TD更新公式: Q(s,a) = Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
            td_target = reward + gamma * max_next_q
            td_error = td_target - Q[(state, action)]
            Q[(state, action)] = Q[(state, action)] + alpha * td_error
            
            # 更新目标策略为贪心策略
            q_values = [Q[(state, a)] for a in env.action_space]
            best_action_idx = np.argmax(q_values)
            target_policy_matrix[state_idx, :] = 0.0
            target_policy_matrix[state_idx, best_action_idx] = 1.0
            
            # 定期记录策略和值函数（按步数记录）
            if global_step % record_freq == 0:
                # 记录当前策略
                policy_history.append(target_policy_matrix.copy())
                
                # 计算状态值函数 V(s) = max_a Q(s,a)
                V = {}
                for s_idx in range(num_states):
                    s = env.idx_to_state[s_idx]
                    V[s] = max([Q[(s, a)] for a in env.action_space])
                value_history.append(V.copy())
            
            state = next_state
            
            if done:
                # 到达终止状态，重新开始（对于单episode模式，这意味着从起点继续）
                if num_episodes == 1:  # 单episode模式
                    state, _ = env.reset()
                    state = tuple(state)
                else:
                    break
        
        # 记录episode数据
        episode_data.append({
            'states': episode_states,
            'actions': episode_actions,
            'rewards': episode_rewards
        })
    
    # 最后记录一次策略和值函数
    if len(policy_history) == 0 or global_step % record_freq != 0:
        policy_history.append(target_policy_matrix.copy())
        V = {}
        for s_idx in range(num_states):
            s = env.idx_to_state[s_idx]
            V[s] = max([Q[(s, a)] for a in env.action_space])
        value_history.append(V.copy())
    
    # 恢复环境的原始起始状态
    env.start_state = original_start_state
    
    return Q, target_policy_matrix, policy_history, value_history, episode_data, state_value_errors


# =========================
# Q-learning Algorithm (Off-policy) — 数组版
# =========================

def q_learning_off_policy(
    env: GridWorld,
    behavior_policy_matrix: np.ndarray,
    num_episodes: int = 1000,
    alpha: float = 0.1,
    gamma: float = 0.9,
    max_steps: int = 100,
    record_freq: int = 50
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], List[Dict[Tuple[int, int], float]], List[Dict[str, List]], Dict[str, List]]:
    """
    使用离策略 Q-learning 学习最优策略（数组实现版本）。

    - 行为策略（behavior）由 `behavior_policy_matrix` 给出，用于采样动作；
    - 目标策略（target）始终对当前 Q 值取贪心；
    - Q 值以二维数组形式维护，shape = (num_states, num_actions)。

    参数:
        env: GridWorld 环境实例
        behavior_policy_matrix: 行为策略矩阵，shape = (num_states, num_actions)
        num_episodes: 训练的 episode 数量
        alpha: 学习率
        gamma: 折扣因子
        max_steps: 每个 episode 的最大步数
        record_freq: 按步数记录策略和值函数的频率

    返回:
        qvalue: Q 值数组，shape = (num_states, num_actions)
        policy: 最终目标策略矩阵，shape = (num_states, num_actions)
        policy_history: 目标策略矩阵的历史快照列表（用于可视化/分析）
        value_history: 状态值函数字典的历史列表，每个元素为 {state_xy: V(state)}
        episode_data: 每个 episode 的轨迹数据字典列表，元素格式为
                      {'states': [...], 'actions': [...], 'rewards': [...]}。
        actual_trajectory: 全局训练过程的实际轨迹字典，包含 'states'/'actions'/'rewards'。

    备注:
        本实现与现有 `q_learning(...)` 等价于离策略更新，但以数组存储 Q，
        更贴近经典教科书的实现形式，且便于与策略矩阵对齐。
        特别说明：到达目标状态后不终止，继续使用行为策略采样（例如均匀策略）。
    """
    num_states = env.num_states
    num_actions = len(env.action_space)

    # 初始化 Q 与目标策略（先将第0个动作置为1.0）
    qvalue = np.zeros((num_states, num_actions), dtype=float)
    policy = np.zeros((num_states, num_actions), dtype=float)
    policy[:, 0] = 1.0

    policy_history: List[np.ndarray] = []
    value_history: List[Dict[Tuple[int, int], float]] = []
    episode_data: List[Dict[str, List]] = []
    actual_trajectory: Dict[str, List] = {'states': [], 'actions': [], 'rewards': []}

    # 保存原始起始状态，便于训练结束后还原
    original_start_state = env.start_state

    global_step = 0
    for episode in range(num_episodes):
        state, _ = env.reset()
        state = tuple(state)

        states_in_episode: List[Tuple[int, int]] = [state]
        actions_in_episode: List[Tuple[int, int]] = []
        rewards_in_episode: List[float] = []

        for step in range(max_steps):
            global_step += 1

            state_idx = env.state_to_idx[state]
            action_probs = behavior_policy_matrix[state_idx]
            action_idx = np.random.choice(num_actions, p=action_probs)
            action = env.action_space[action_idx]

            next_state, reward, done, _ = env.step(action)
            next_state = tuple(next_state)

            # 记录 episode 内轨迹
            actions_in_episode.append(action)
            rewards_in_episode.append(reward)
            states_in_episode.append(next_state)

            # 记录全局训练轨迹
            actual_trajectory['states'].append(state)
            actual_trajectory['actions'].append(action)
            actual_trajectory['rewards'].append(reward)

            next_state_idx = env.state_to_idx[next_state]
            # 即使到达目标也不视为终止，继续采样与更新
            max_next_q = float(np.max(qvalue[next_state_idx]))

            td_target = reward + gamma * max_next_q
            td_error = td_target - qvalue[state_idx, action_idx]
            qvalue[state_idx, action_idx] += alpha * td_error

            # 贪心更新目标策略
            best_action_idx = int(np.argmax(qvalue[state_idx]))
            policy[state_idx, :] = 0.0
            policy[state_idx, best_action_idx] = 1.0

            # 记录策略和值函数（按步数）
            if global_step % record_freq == 0:
                policy_history.append(policy.copy())
                V_dict: Dict[Tuple[int, int], float] = {}
                for s_idx in range(num_states):
                    s_xy = env.idx_to_state[s_idx]
                    V_dict[s_xy] = float(np.max(qvalue[s_idx]))
                value_history.append(V_dict)

            state = next_state

        episode_data.append({
            'states': states_in_episode,
            'actions': actions_in_episode,
            'rewards': rewards_in_episode
        })

    # 末尾补一次记录，确保有最终策略和值函数
    if len(policy_history) == 0 or global_step % record_freq != 0:
        policy_history.append(policy.copy())
        V_dict: Dict[Tuple[int, int], float] = {}
        for s_idx in range(num_states):
            s_xy = env.idx_to_state[s_idx]
            V_dict[s_xy] = float(np.max(qvalue[s_idx]))
        value_history.append(V_dict)

    # 还原环境起始状态
    env.start_state = original_start_state

    return qvalue, policy, policy_history, value_history, episode_data, actual_trajectory


def create_epsilon_greedy_policy(num_states, num_actions, epsilon=0.3):
    """
    创建epsilon-greedy行为策略。
    
    参数:
        num_states: 状态数量
        num_actions: 动作数量
        epsilon: 探索率
    
    返回:
        policy_matrix: 策略矩阵 (num_states, num_actions)
    """
    policy_matrix = np.ones((num_states, num_actions)) * (epsilon / num_actions)
    # 为每个状态选择一个主要动作（贪心动作）
    for s in range(num_states):
        greedy_action = np.random.randint(num_actions)  # 随机选择初始贪心动作
        policy_matrix[s, greedy_action] += (1.0 - epsilon)
    return policy_matrix


def create_soft_uniform_policy(num_states, num_actions):
    """
    创建均匀随机行为策略。
    
    参数:
        num_states: 状态数量
        num_actions: 动作数量
    
    返回:
        policy_matrix: 策略矩阵 (num_states, num_actions)
    """
    return np.ones((num_states, num_actions)) / num_actions


def visualize_episode_trajectory(
    env: GridWorld,
    episode_states: List[Tuple[int, int]],
    episode_actions: List[Tuple[int, int]],
    episode_rewards: List[float],
    behavior_policy_matrix: Optional[np.ndarray] = None,
    save_path: Optional[str] = None
):
    """
    可视化 episode 轨迹，坐标系以网格左上角为原点，向右为 x 增长，向下为 y 增长。

    参数:
        env: GridWorld 环境实例
        episode_states: 状态序列，元素为 (x, y)
        episode_actions: 动作序列，元素为 (dx, dy)
        episode_rewards: 奖励序列
        behavior_policy_matrix: 若提供，将以浅色箭头在背景绘制行为策略（概率阈值 0.05）
        save_path: 若提供路径，保存图像；否则直接展示
    """
    # 使用matplotlib直接绘图，不依赖GridWorld的render
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # 设置坐标轴与网格（纵向网格整体右移半格）
    ax.set_xlim(-0.5, env.env_size[0] - 0.5)
    ax.set_ylim(-0.5, env.env_size[1] - 0.5)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.set_xticks(np.arange(env.env_size[0]) + 0.5)
    ax.set_yticks(np.arange(env.env_size[1]) + 0.5)
    ax.grid(True, linestyle="-", color="gray", linewidth="1")
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    
    # 绘制目标状态
    target_rect = patches.Rectangle(
        (env.target_state[0]-0.5, env.target_state[1]-0.5), 
        1, 1, linewidth=1, 
        edgecolor=(0.3010, 0.7450, 0.9330), 
        facecolor=(0.3010, 0.7450, 0.9330),
        label='Target'
    )
    ax.add_patch(target_rect)
    
    # 绘制禁止状态
    for i, forbidden_state in enumerate(env.forbidden_states):
        rect = patches.Rectangle(
            (forbidden_state[0]-0.5, forbidden_state[1]-0.5), 
            1, 1, linewidth=1,
            edgecolor=(0.9290, 0.6940, 0.125), 
            facecolor=(0.9290, 0.6940, 0.125),
            label='Forbidden' if i == 0 else None
        )
        ax.add_patch(rect)
    
    # 如果提供了行为策略，绘制策略箭头（作为背景）
    if behavior_policy_matrix is not None:
        for state_idx in range(env.num_states):
            state = env.idx_to_state[state_idx]
            if state in env.forbidden_states:
                continue
            x, y = state
            for action_idx, prob in enumerate(behavior_policy_matrix[state_idx]):
                if prob > 0.05:  # 只显示概率较大的动作
                    dx, dy = env.action_space[action_idx]
                    if (dx, dy) != (0, 0):
                        ax.add_patch(
                            patches.FancyArrow(
                                x, y,
                                dx=(0.1 + prob / 3) * dx,
                                dy=(0.1 + prob / 3) * dy,
                                color=(0.4660, 0.6740, 0.1880),
                                width=0.001,
                                head_width=0.05,
                                alpha=0.3
                            )
                        )
    
    # 绘制轨迹 - 使用抽取的工具函数，两个位置中心连线，端点随机偏移
    if len(episode_states) > 1:
        for i in range(len(episode_states) - 1):
            draw_random_line(ax, episode_states[i], episode_states[i + 1])
    
    # 标记起点和终点
    start = episode_states[0]
    end = episode_states[-1]
    ax.plot(start[0], start[1], 'o', color='darkgreen', 
           markersize=15, label='Start', zorder=10, markeredgecolor='black', markeredgewidth=2)
    ax.plot(end[0], end[1], '*', color='red', 
           markersize=20, label='End', zorder=10, markeredgecolor='black', markeredgewidth=1)
    
    # 添加图例
    ax.legend(loc='upper right', fontsize=10)
    ax.set_title(f'Episode Trajectory ({len(episode_actions)} steps, Total Reward: {sum(episode_rewards):.2f})', 
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure: {save_path}")
    else:
        plt.show()
    
    plt.close(fig)


def draw_random_line(
    ax,
    pos1: Tuple[float, float],
    pos2: Tuple[float, float],
    offset_scale: float = 0.15,
    color: str = 'green',
    linewidth: float = 2.0,
    alpha: float = 0.7
) -> None:
    """
    在 `pos1` 与 `pos2` 之间绘制一条分段折线：两端点随机偏移，中心点也随机偏移。

    - 折线由两段组成：起点→随机中心点、随机中心点→终点；
    - 起点/终点在各自格子中心附近做小范围随机偏移；
    - 中心点在几何中心附近再加入随机偏移，使每段线具有独特形状。

    参数:
        ax: Matplotlib 的轴对象
        pos1: 起点位置 (x, y)，采用网格坐标（左上为原点，已设置 `invert_yaxis`）
        pos2: 终点位置 (x, y)
        offset_scale: 端点随机偏移的最大幅度
        color: 线条颜色
        linewidth: 线宽
        alpha: 透明度
    """
    x1, y1 = float(pos1[0]), float(pos1[1])
    x2, y2 = float(pos2[0]), float(pos2[1])

    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    # 中心点加入随机偏移（幅度略小于端点）
    center_x += np.random.uniform(-offset_scale * 0.6, offset_scale * 0.6)
    center_y += np.random.uniform(-offset_scale * 0.6, offset_scale * 0.6)

    start_x = x1 + np.random.uniform(-offset_scale, offset_scale)
    start_y = y1 + np.random.uniform(-offset_scale, offset_scale)
    end_x = x2 + np.random.uniform(-offset_scale, offset_scale)
    end_y = y2 + np.random.uniform(-offset_scale, offset_scale)

    ax.plot([start_x, center_x], [start_y, center_y], color=color, linewidth=linewidth, alpha=alpha, zorder=5)
    ax.plot([center_x, end_x], [center_y, end_y], color=color, linewidth=linewidth, alpha=alpha, zorder=5)
    ax.plot(center_x, center_y, 'o', color=color, markersize=3, alpha=0.5, zorder=6)


def visualize_episode_on_env(
    env: GridWorld,
    policy_matrix: np.ndarray,
    episode_length: int,
    save_path: Optional[str] = None,
    title: Optional[str] = None
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]], List[float]]:
    """
    在环境的仿真坐标系上按给定策略矩阵采样并绘制轨迹。

    参数:
        env: GridWorld 环境实例
        policy_matrix: 策略矩阵，shape=(num_states, num_actions)
        episode_length: 采样的步数长度
        save_path: 保存路径（可选）
        title: 图标题（可选）

    返回:
        states, actions, rewards：本次采样得到的轨迹数据
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    fig, ax = plt.subplots(figsize=(10, 7))

    # 绑定到 env 以复用其绘制函数
    env.canvas = fig
    env.ax = ax

    # 坐标系与栅格
    ax.set_xlim(-0.5, env.env_size[0] - 0.5)
    ax.set_ylim(-0.5, env.env_size[1] - 0.5)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.set_xticks(np.arange(env.env_size[0]) + 0.5)
    ax.set_yticks(np.arange(env.env_size[1]) + 0.5)
    ax.grid(True, linestyle="-", color="gray", linewidth=1)
    ax.xaxis.set_ticks_position('top')
    ax.tick_params(
        bottom=False, left=False, right=False, top=False,
        labelbottom=False, labelleft=False, labeltop=False
    )

    # 目标与禁区
    ax.add_patch(
        patches.Rectangle(
            (env.target_state[0]-0.5, env.target_state[1]-0.5), 1, 1,
            linewidth=1, edgecolor=env.color_target, facecolor=env.color_target
        )
    )
    for forbidden_state in env.forbidden_states:
        ax.add_patch(
            patches.Rectangle(
                (forbidden_state[0]-0.5, forbidden_state[1]-0.5), 1, 1,
                linewidth=1, edgecolor=env.color_forbid, facecolor=env.color_forbid
            )
        )

    # 采样并记录轨迹
    states: List[Tuple[int, int]] = []
    actions: List[Tuple[int, int]] = []
    rewards: List[float] = []

    s, _ = env.reset()
    s = tuple(s)
    states.append(s)
    for t in range(episode_length):
        s_idx = env.state_to_idx[s]
        a_idx = int(np.random.choice(len(env.action_space), p=policy_matrix[s_idx]))
        a = env.action_space[a_idx]
        ns, r, done, _ = env.step(a)
        ns = tuple(ns)
        actions.append(a)
        rewards.append(r)
        states.append(ns)
        s = ns
        # 按你的需求：到达目标后继续采样，不提前终止

    # 绘制轨迹：
    # - 默认使用中心点连线并在端点加入随机偏移（更易区分）
    # - 若前后状态相同且动作不是原地不动，则沿动作方向随机伸出一小段线
    def _jitter(p: Tuple[int, int], scale: float = 0.01) -> Tuple[float, float]:
        return (
            float(p[0]) + float(np.random.uniform(-scale, scale)),
            float(p[1]) + float(np.random.uniform(-scale, scale))
        )

    if len(states) > 1:
        for i in range(len(states) - 1):
            s_cur = states[i]
            s_nxt = states[i + 1]
            a_dx, a_dy = actions[i]

            if s_cur == s_nxt and (a_dx, a_dy) != (0, 0):
                sx, sy = _jitter(s_cur)
                seg_len = 1.0 / 3.0
                tx = sx + a_dx * seg_len
                ty = sy + a_dy * seg_len
                ax.plot([sx, tx], [sy, ty], color=env.color_trajectory, linewidth=1.2)
            else:
                # 使用已有工具函数，两端带偏移并通过中点折线连接
                if 'draw_random_line' in globals():
                    draw_random_line(ax, s_cur, s_nxt, color=env.color_trajectory, linewidth=1.2, alpha=0.8)
                else:
                    sx, sy = _jitter(s_cur)
                    tx, ty = _jitter(s_nxt)
                    ax.plot([sx, tx], [sy, ty], color=env.color_trajectory, linewidth=1.2)

    # 起止标记
    ax.plot(states[0][0], states[0][1], 'o', color='darkgreen', markersize=12)
    ax.plot(states[-1][0], states[-1][1], '*', color='red', markersize=14)

    # 标题
    ttl = title or f"Episode Trajectory ({len(actions)} steps, Total Reward: {sum(rewards):.2f})"
    ax.set_title(ttl, fontsize=14, fontweight='bold')

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure: {save_path}")
    else:
        plt.show()
    plt.close(fig)

    return states, actions, rewards


def visualize_path_on_env(
    env: GridWorld,
    state_sequence: List[Tuple[int, int]],
    save_path: Optional[str] = None,
    title: Optional[str] = None,
    overlay_policy: Optional[np.ndarray] = None,
    action_sequence: Optional[List[Tuple[int, int]]] = None
) -> None:
    """
    在环境仿真坐标系上绘制已给定的状态序列轨迹；可选叠加策略箭头。

    绘制规则：
    - 默认连接相邻状态的中心，并在端点加入随机偏移以便区分多条线。
    - 若提供 `action_sequence`，当相邻状态相同且相应动作不是原地不动时，
      从该状态中心沿动作方向随机伸出一小段线，表示尝试移动但被阻挡。

    参数:
        env: GridWorld 环境实例
        state_sequence: 状态序列（元素为 (x,y)）
        save_path: 保存路径（可选）
        title: 图标题（可选）
        overlay_policy: 若提供，则叠加策略箭头（如均匀行为策略）
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    fig, ax = plt.subplots(figsize=(10, 7))
    env.canvas = fig
    env.ax = ax

    ax.set_xlim(-0.5, env.env_size[0] - 0.5)
    ax.set_ylim(-0.5, env.env_size[1] - 0.5)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.set_xticks(np.arange(env.env_size[0]) + 0.5)
    ax.set_yticks(np.arange(env.env_size[1]) + 0.5)
    ax.grid(True, linestyle="-", color="gray", linewidth=1)
    ax.xaxis.set_ticks_position('top')
    ax.tick_params(
        bottom=False, left=False, right=False, top=False,
        labelbottom=False, labelleft=False, labeltop=False
    )

    ax.add_patch(
        patches.Rectangle(
            (env.target_state[0]-0.5, env.target_state[1]-0.5), 1, 1,
            linewidth=1, edgecolor=env.color_target, facecolor=env.color_target
        )
    )
    for forbidden_state in env.forbidden_states:
        ax.add_patch(
            patches.Rectangle(
                (forbidden_state[0]-0.5, forbidden_state[1]-0.5), 1, 1,
                linewidth=1, edgecolor=env.color_forbid, facecolor=env.color_forbid
            )
        )

    # 可选叠加策略箭头（与行为策略图一致）
    if overlay_policy is not None:
        from matplotlib import patches as mpatches
        for state_idx in range(env.num_states):
            st = env.idx_to_state[state_idx]
            if st in env.forbidden_states:
                continue
            x, y = st
            for a_idx, prob in enumerate(overlay_policy[state_idx]):
                if prob > 0.0:
                    dx, dy = env.action_space[a_idx]
                    if (dx, dy) != (0, 0):
                        ax.add_patch(
                            mpatches.FancyArrow(
                                x, y,
                                dx=(0.1 + prob / 2) * dx,
                                dy=(0.1 + prob / 2) * dy,
                                color=env.color_policy,
                                width=0.001,
                                head_width=0.05,
                                alpha=0.5
                            )
                        )
                    else:
                        ax.add_patch(
                            mpatches.Circle(
                                (x, y), radius=0.07,
                                facecolor=env.color_policy,
                                edgecolor=env.color_policy,
                                linewidth=1, fill=False, alpha=0.5
                            )
                        )

    # 绘制轨迹：含随机偏移；支持“停留但动作非零”的短线段
    def _jitter(p: Tuple[int, int], scale: float = 0.15) -> Tuple[float, float]:
        return (
            float(p[0]) + float(np.random.uniform(-scale, scale)),
            float(p[1]) + float(np.random.uniform(-scale, scale))
        )

    if len(state_sequence) > 1:
        use_actions = action_sequence is not None and len(action_sequence) == len(state_sequence) - 1
        for i in range(len(state_sequence) - 1):
            s_cur = state_sequence[i]
            s_nxt = state_sequence[i + 1]

            if use_actions:
                a_dx, a_dy = action_sequence[i]
                if s_cur == s_nxt and (a_dx, a_dy) != (0, 0):
                    sx, sy = _jitter(s_cur)
                    seg_len = 1.0 / 3.0
                    tx = sx + a_dx * seg_len
                    ty = sy + a_dy * seg_len
                    ax.plot([sx, tx], [sy, ty], color=env.color_trajectory, linewidth=1.2)
                    continue

            if 'draw_random_line' in globals():
                draw_random_line(ax, s_cur, s_nxt, color=env.color_trajectory, linewidth=1.2, alpha=0.8)
            else:
                sx, sy = _jitter(s_cur)
                tx, ty = _jitter(s_nxt)
                ax.plot([sx, tx], [sy, ty], color=env.color_trajectory, linewidth=1.2)

    # 起止标记
    ax.plot(state_sequence[0][0], state_sequence[0][1], 'o', color='darkgreen', markersize=12)
    ax.plot(state_sequence[-1][0], state_sequence[-1][1], '*', color='red', markersize=14)

    ttl = title or f"Episode Trajectory ({len(state_sequence)-1} steps)"
    ax.set_title(ttl, fontsize=14, fontweight='bold')

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure: {save_path}")
    else:
        plt.show()
    plt.close(fig)


def compute_state_value_error(V_learned, V_optimal):
    """
    计算学习到的状态值函数与最优状态值函数之间的误差。
    
    参数:
        V_learned: 学习到的状态值函数字典 {state: value}
        V_optimal: 最优状态值函数字典 {state: value}
    
    返回:
        error: 平均绝对误差
    """
    errors = []
    for state in V_learned:
        if state in V_optimal:
            errors.append(abs(V_learned[state] - V_optimal[state]))
    return np.mean(errors) if errors else 0.0


def plot_state_value_error_curve(error_history, save_path=None):
    """
    绘制状态值误差曲线。
    
    参数:
        error_history: 误差历史列表
        save_path: 保存路径（可选）
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(error_history, linewidth=2)
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('State Value Error', fontsize=12)
    ax.set_title('State Value Error vs Episode', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    _save_or_show(save_path)
