# examples/example_assignment_2.py
import sys
sys.path.append("..")

import numpy as np
import matplotlib.pyplot as plt

from src.grid_world import GridWorld
from examples.arguments import args  # 与 A1 相同的参数入口

# ===== 复用 A1 的两种策略（保持一致，不引入新库） =====
from collections import deque

def build_deterministic_policy(env):
    num_states = env.num_states
    policy = np.zeros((num_states, len(env.action_space)))
    stay_idx = env.action_space.index((0,0))

    target = tuple(env.target_state)
    queue = deque([target])
    parent = {target: None}

    while queue:
        current = queue.popleft()
        for action in env.action_space:
            if action == (0,0):  # 不考虑 stay 的反向
                continue
            prev = (current[0] - action[0], current[1] - action[1])
            if (0 <= prev[0] < env.env_size[0] and
                0 <= prev[1] < env.env_size[1] and
                prev not in env.forbidden_states and
                prev not in parent):
                parent[prev] = current
                queue.append(prev)

    for s in range(num_states):
        st = env.idx_to_state[s]
        if st in env.forbidden_states or st == target or st not in parent:
            policy[s, stay_idx] = 1.0
            continue
        nxt = parent[st]
        dx, dy = nxt[0] - st[0], nxt[1] - st[1]
        a = (dx, dy)
        if a in env.action_space:
            policy[s, env.action_space.index(a)] = 1.0
        else:
            policy[s, stay_idx] = 1.0
    return policy

def build_stochastic_policy(env):
    num_states = env.num_states
    policy = np.zeros((num_states, len(env.action_space)))
    weights = {(1,0):0.5, (0,1):0.3, (0,-1):0.1, (-1,0):0.05, (0,0):0.05}
    stay_idx = env.action_space.index((0,0))
    for s in range(num_states):
        st = env.idx_to_state[s]
        if st == tuple(env.target_state) or st in env.forbidden_states:
            policy[s, stay_idx] = 1.0
        else:
            probs = np.array([weights.get(a,0.0) for a in env.action_space], dtype=float)
            policy[s,:] = probs / probs.sum()
    return policy
# =======================================================

# —— 基于环境精确构造 r_pi 与 P_pi（不做任何外部假设） ——
def make_r_pi_and_P_pi(env, policy_matrix):
    """
    r_pi[s] = E[R | s, a~pi(·|s)]
    P_pi[s, s'] = Pr(s' | s, a~pi(·|s))
    环境一步转移是确定性的（给定 (s,a) 唯一下一状态），因此对每个动作只会把质量
    聚到一个 s' 上；对随机策略按动作概率加权。
    """
    S = env.num_states
    A = len(env.action_space)
    r_pi = np.zeros(S)
    P_pi = np.zeros((S, S))

    for s in range(S):
        st = env.idx_to_state[s]
        for a_idx, a in enumerate(env.action_space):
            pi = policy_matrix[s, a_idx]
            if pi == 0.0:
                continue
            # 用环境的真实一步动力学获取 (s,a) 的 s' 与 r
            next_state, reward = env._get_next_state_and_reward(st, a)
            s_next = env.state_to_idx[next_state]
            r_pi[s] += pi * reward
            P_pi[s, s_next] += pi
    return r_pi, P_pi

# —— 闭式求解：解 (I - γP)V = r （用 solve 代替显式求逆） ——
def policy_eval_closed_form(r_pi, P_pi, gamma):
    I = np.eye(P_pi.shape[0])
    V = np.linalg.solve(I - gamma * P_pi, r_pi)
    return V

# —— 迭代求解：V^{k+1} = r + γPV^k ——
def policy_eval_iterative(r_pi, P_pi, gamma, tol=1e-8, max_iters=10000):
    V = np.zeros_like(r_pi)
    for k in range(max_iters):
        V_new = r_pi + gamma * P_pi.dot(V)
        if np.max(np.abs(V_new - V)) < tol:
            return V_new, k+1
        V = V_new
    return V, max_iters

# —— 把价值贴到现有网格可视化上（沿用你的接口） ——
def show_values_on_grid(env, policy_matrix, values, title="Values on Grid", precision=2, save_path=None):
    env.reset()           # 先 reset 才有 traj
    env.render()
    env.add_policy(policy_matrix)
    env.add_state_values(values, precision=precision)
    plt.title(title)

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to {save_path}")
    else:
        plt.show()



def pretty_print(name, arr, precision=3):
    np.set_printoptions(precision=precision, suppress=True)
    print(f"\n{name} =\n{arr}")

def maybe_save_csv(filename, arr):
    try:
        np.savetxt(filename, arr, delimiter=",")
        print(f"Saved: {filename}")
    except Exception as e:
        print(f"Warning: failed to save {filename}: {e}")
def show_policy_only(env, policy_matrix, title="Policy (arrows only)", save_path=None):
    env.reset()
    env.render()
    env.add_policy(policy_matrix)
    plt.title(title)

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Policy figure saved to {save_path}")
    else:
        plt.show()

def run_once(policy_builder, policy_name):
    print("\n" + "="*80)
    print(f"{policy_name}")
    print("="*80)

    env = GridWorld()
    policy = policy_builder(env)

    # r_pi, P_pi
    r_pi, P_pi = make_r_pi_and_P_pi(env, policy)

    # 输出/保存
    pretty_print("r_pi", r_pi)
    pretty_print("P_pi", P_pi)
    maybe_save_csv(f"r_{policy_name}.csv", r_pi)
    maybe_save_csv(f"P_{policy_name}.csv", P_pi)
    # 绘制策略 (仅箭头)
    show_policy_only(env, policy, title=f"{policy_name}: Policy Only",
                    save_path=f"{policy_name}_policy.png")

    gamma = args.discount_rate  # 与 A1 相同来源
    # 闭式
    V_closed = policy_eval_closed_form(r_pi, P_pi, gamma)
    pretty_print("V_closed_form", V_closed)

    # 迭代
    V_iter, iters = policy_eval_iterative(r_pi, P_pi, gamma)
    pretty_print(f"V_iterative (converged in {iters} iters)", V_iter)

    # 可视化（两种解各画一张）
    show_values_on_grid(env, policy, V_closed, 
                        title=f"{policy_name}: Closed-form V", 
                        precision=2,
                        save_path=f"{policy_name}_closed.png")

    show_values_on_grid(env, policy, V_iter,   
                        title=f"{policy_name}: Iterative V", 
                        precision=2,
                        save_path=f"{policy_name}_iter.png")



def main():
    # 与 A1 完全相同的两种策略
    run_once(build_deterministic_policy, "Deterministic")
    run_once(build_stochastic_policy,   "Stochastic")

if __name__ == "__main__":
    main()
