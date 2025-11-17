# examples/assignment_3.py
import sys
import os
from typing import Dict, List, Tuple

sys.path.append("..")

import numpy as np

# 项目依赖（假定存在）
from src.grid_world import GridWorld
from examples.arguments import args
from src.utils import value_iteration, visualize_optimal_policy, visualize_state_values, visualize_state_values_and_policy, visualize_value_evolution, create_probability_and_reward_functions_from_env, sample_probability_and_reward_functions

# 全局变量定义
OUTPUT_DIR = "../../../assignment_notes/assignment_3/output"


def run_value_iteration():
    """
    运行价值迭代算法并生成结果。
    """
    # 创建网格世界环境
    env = GridWorld(
        env_size=args.env_size,
        start_state=args.start_state,
        target_state=args.target_state,
        forbidden_states=args.forbidden_states
    )
    
    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("=" * 80)
    print("作业3：价值迭代算法")
    print("=" * 80)
    
    # 打印环境设置
    print("\n1. 环境设置:")
    print(f"   环境大小: {args.env_size}")
    print(f"   起始状态: {args.start_state}")
    print(f"   目标状态: {args.target_state}")
    print(f"   禁止状态: {args.forbidden_states}")
    print(f"   目标奖励: {args.reward_target}")
    print(f"   禁区奖励: {args.reward_forbidden}")
    print(f"   步进奖励: {args.reward_step}")
    print(f"   折扣因子: {args.discount_rate}")
    
    # 从环境创建概率和奖励函数
    p_r, p_s = create_probability_and_reward_functions_from_env(env)
    
    # 获取状态和动作
    states = list(env.idx_to_state.values())
    actions = env.action_space
    
    print("\n2. 价值迭代算法实现:")
    print("   算法通过迭代更新值函数来寻找最优策略:")
    print("   1. 初始化值函数 V(s) = 0 for all s")
    print("   2. 重复直到收敛:")
    print("      a) 对于每个状态 s:")
    print("         i) 对于每个动作 a:")
    print("            - 计算 Q(s,a) = Σs' P(s'|s,a) [R(s,a,s') + γ V(s')]")
    print("         ii) 更新 V(s) = max_a Q(s,a)")
    print("         iii) 更新策略 π(s) = arg max_a Q(s,a)")
    
    # 执行价值迭代
    gamma = args.discount_rate
    epsilon = 1e-6
    V_dict, policy_dict, iterations, V_history, policy_history = value_iteration(
        states, actions, p_r, p_s, gamma, epsilon
    )
    
    # 将字典格式的值函数和策略转换为数组格式
    V = np.array([V_dict[env.idx_to_state[i]] for i in range(len(env.idx_to_state))])
    policy_matrix = np.zeros((len(states), len(actions)))
    
    for i, state in enumerate(states):
        state_idx = env.state_to_idx[state]
        best_action = policy_dict[state]
        if best_action in actions:
            action_idx = actions.index(best_action)
            policy_matrix[state_idx, action_idx] = 1.0
    
    print(f"\n3. 算法结果:")
    print(f"   迭代次数: {iterations}")
    print(f"   收敛阈值: {epsilon}")
    
    # 打印最优值函数
    print(f"\n4. 最优值函数:")
    for i, state in enumerate(states):
        if state not in env.forbidden_states and state != tuple(env.target_state):
            print(f"   状态 {state}: {V[i]:.2f}")
    
    # 可视化最优策略
    print(f"\n5. 最优策略可视化:")
    visualize_optimal_policy(
        env, 
        policy_matrix, 
        save_path=f"{OUTPUT_DIR}/optimal_policy.png"
    )
    print(f"   最优策略已保存到: {OUTPUT_DIR}/optimal_policy.png")
    
    # 可视化最优状态值
    print(f"\n6. 最优状态值可视化:")
    visualize_state_values(
        env, 
        V, 
        precision=2, 
        save_path=f"{OUTPUT_DIR}/optimal_state_values.png"
    )
    print(f"   最优状态值已保存到: {OUTPUT_DIR}/optimal_state_values.png")
    
    # 可视化值函数演变
    print(f"\n7. 值函数演变可视化:")
    visualize_value_evolution(
        env, 
        V_history, 
        policy_history,
        gamma, 
        max_figures=10, 
        save_path_prefix=f"{OUTPUT_DIR}/value_evolution"
    )
    print(f"   值函数演变已保存到: {OUTPUT_DIR}/value_evolution_*.png")
    
    print(f"\n8. 实验观察:")
    print("   通过价值迭代算法，我们得到了网格世界环境的最优策略和最优值函数。")
    print("   - 最优策略指导智能体如何行动以最大化累积奖励")
    print("   - 最优值函数表示每个状态的长期价值")
    print("   - 值函数演变展示了算法的收敛过程")
    
    # 演示采样方法
    print(f"\n9. 采样方法演示:")
    print("   通过采样方法创建概率和奖励函数:")
    p_r_sampled, p_s_sampled = sample_probability_and_reward_functions(env, num_samples=5000)
    
    # 使用采样函数再次运行价值迭代
    V_dict_sampled, policy_dict_sampled, iterations_sampled, V_history_sampled, policy_history_sampled = value_iteration(
        states, actions, p_r_sampled, p_s_sampled, gamma, epsilon
    )
    
    # 将字典格式的值函数和策略转换为数组格式
    V_sampled = np.array([V_dict_sampled[env.idx_to_state[i]] for i in range(len(env.idx_to_state))])
    
    print(f"   使用采样函数的价值迭代完成，迭代次数: {iterations_sampled}")
    
    # 比较两种方法的结果
    print(f"\n10. 方法比较:")
    difference = np.abs(V - V_sampled)
    max_diff = np.max(difference)
    mean_diff = np.mean(difference)
    print(f"    最大差异: {max_diff:.6f}")
    print(f"    平均差异: {mean_diff:.6f}")
    
    if max_diff < 1e-3:
        print("    两种方法的结果基本一致，说明采样方法有效。")
    else:
        print("    两种方法的结果存在较大差异，可能需要增加采样次数。")
    
    return V, policy_matrix, iterations, V_history


def main():
    run_value_iteration()


if __name__ == "__main__":
    main()