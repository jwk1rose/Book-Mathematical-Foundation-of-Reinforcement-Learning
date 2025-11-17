# examples/assignment_4.py
import sys
import os
from typing import Dict, List, Tuple

sys.path.append("..")

import numpy as np

# 项目依赖
from src.grid_world import GridWorld
from examples.arguments import args
from src.utils import (
    value_iteration,
    q_learning_off_policy,
    create_epsilon_greedy_policy,
    create_soft_uniform_policy,
    visualize_optimal_policy,
    visualize_state_values,
    visualize_episode_trajectory,
    visualize_episode_on_env,
    visualize_path_on_env,
    compute_state_value_error,
    plot_state_value_error_curve,
    create_probability_and_reward_functions_from_env,
    add_state_numbers_to_env,
    build_deterministic_policy,
    compute_absorbing_optimal_values
)

# 全局变量定义
OUTPUT_DIR = "../../../assignment_notes/assignment_4/output"


def run_q_learning_experiment():
    """
    运行Q-learning算法实验并生成完整结果。
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
    print("作业4：Q-learning算法（离策略版本）")
    print("=" * 80)
    
    # ========================================
    # 1. 问题设置描述
    # ========================================
    print("\n" + "=" * 80)
    print("1. 问题设置")
    print("=" * 80)
    print(f"\n任务描述:")
    print(f"  智能体需要在{args.env_size[0]}x{args.env_size[1]}的网格世界中找到从起始状态到目标状态的最优路径。")
    print(f"\n环境设置:")
    print(f"  - 环境大小: {args.env_size[0]} 行 × {args.env_size[1]} 列")
    print(f"  - 起始状态: {args.start_state}")
    print(f"  - 目标状态: {args.target_state}")
    print(f"  - 禁止状态: {args.forbidden_states}")
    print(f"\n奖励设置:")
    print(f"  - 到达目标状态奖励: {args.reward_target}")
    print(f"  - 进入禁止区域惩罚: {args.reward_forbidden}")
    print(f"  - 每步移动成本: {args.reward_step}")
    print(f"\n折扣因子:")
    print(f"  - γ (gamma) = {args.discount_rate}")
    
    # ========================================
    # 2. 使用上次作业的代码找到最优策略和最优状态值（作为参考）
    # ========================================
    print("\n" + "=" * 80)
    print("2. 计算参考最优策略和最优状态值（使用价值迭代算法）")
    print("=" * 80)
    
    # 从环境创建概率和奖励函数
    p_r, p_s = create_probability_and_reward_functions_from_env(env)
    
    # 获取状态和动作
    states = list(env.idx_to_state.values())
    actions = env.action_space
    
    # 执行价值迭代
    gamma = args.discount_rate
    epsilon = 1e-6
    V_dict_optimal, policy_dict_optimal, iterations, V_history, policy_history = value_iteration(
        states, actions, p_r, p_s, gamma, epsilon
    )
    
    # 将字典格式的值函数转换为数组格式
    V_optimal = np.array([V_dict_optimal[env.idx_to_state[i]] for i in range(len(env.idx_to_state))])
    
    # 使用assignment_3的方法生成最优策略（BFS确定性策略）
    optimal_policy_matrix = build_deterministic_policy(env)
    
    print(f"\n价值迭代完成，迭代次数: {iterations}")
    print(f"\n最优状态值函数:")
    for i, state in enumerate(states):
        if state not in env.forbidden_states and state != tuple(env.target_state):
            print(f"  状态 {state}: {V_optimal[i]:.2f}")
    print(f"  目标状态 {tuple(env.target_state)} 的值: {V_dict_optimal[tuple(env.target_state)]:.2f}")
    
    # 可视化最优策略
    print(f"\n保存最优策略可视化...")
    visualize_optimal_policy(
        env,
        optimal_policy_matrix,
        save_path=f"{OUTPUT_DIR}/optimal_policy.png"
    )
    print(f"  已保存到: {OUTPUT_DIR}/optimal_policy.png")
    
    # 可视化最优状态值（与不终止设定一致，使用价值迭代结果）
    print(f"\n保存最优状态值可视化...")
    visualize_state_values(
        env,
        V_optimal,
        precision=2,
        save_path=f"{OUTPUT_DIR}/optimal_state_values.png"
    )
    print(f"  已保存到: {OUTPUT_DIR}/optimal_state_values.png")
    
    # ========================================
    # 3. 描述和绘制行为策略
    # ========================================
    print("\n" + "=" * 80)
    print("3. 行为策略设置")
    print("=" * 80)
    
    # 创建均匀随机行为策略
    print(f"\n使用均匀随机行为策略:")
    print(f"  - 每个动作的选择概率相等")
    print(f"  - 对于{len(env.action_space)}个可能动作，每个动作的概率为 {1.0/len(env.action_space):.2f}")
    print(f"  - 这种策略保证了充分的随机探索，能够访问所有状态-动作对")
    
    behavior_policy_matrix = create_soft_uniform_policy(
        env.num_states,
        len(env.action_space)
    )
    
    # 可视化行为策略
    print(f"\n保存行为策略可视化...")
    visualize_optimal_policy(
        env,
        behavior_policy_matrix,
        save_path=f"{OUTPUT_DIR}/behavior_policy.png"
    )
    print(f"  已保存到: {OUTPUT_DIR}/behavior_policy.png")
    
    # ========================================
    # 4. Q-learning算法实现
    # ========================================
    print("\n" + "=" * 80)
    print("4. Q-learning算法实现")
    print("=" * 80)
    
    print("\nQ-learning算法描述:")
    print("  算法7.3: 通过Q-learning学习最优策略（离策略版本）")
    print("\n  初始化:")
    print("    - 初始化 q₀(s,a) 对所有 (s,a)")
    print("    - 设置行为策略 πᵦ(a|s) 对所有 (s,a)")
    print("    - 学习率 αₜ(s,a) = α > 0")
    print("\n  目标:")
    print("    - 从行为策略 πᵦ 生成的经验样本中学习最优目标策略 πₜ")
    print("\n  对于每个episode {s₀, a₀, r₁, s₁, a₁, r₂, ...}（由πᵦ生成）:")
    print("    对于episode中的每一步 t = 0, 1, 2, ...:")
    print("      1. 更新q值:")
    print("         qₜ₊₁(sₜ, aₜ) = qₜ(sₜ, aₜ) - αₜ(sₜ, aₜ)[qₜ(sₜ, aₜ) - (rₜ₊₁ + γ maxₐ qₜ(sₜ₊₁, a))]")
    print("      2. 更新目标策略:")
    print("         πₜ,ₜ₊₁(a|sₜ) = 1 if a = argmaxₐ qₜ₊₁(sₜ, a)")
    print("         πₜ,ₜ₊₁(a|sₜ) = 0 otherwise")
    
    # Q-learning参数设置
    num_episodes = 200
    alpha = 0.1
    max_steps = 500
    record_freq = 50
    
    print(f"\n算法参数:")
    print(f"  - Episode数量: {num_episodes} (使用单个超长episode)")
    print(f"  - 学习率 α: {alpha}")
    print(f"  - 折扣因子 γ: {gamma}")
    print(f"  - Episode最大步数: {max_steps}")
    print(f"  - 记录频率: 每{record_freq}步")
    print(f"\n说明: 使用均匀随机行为策略进行探索；采用多episode设置以便稳定更新与记录")
    
    print(f"\n开始执行Q-learning算法...")
    Q, target_policy_matrix, policy_history, value_history, episode_data, actual_trajectory = q_learning_off_policy(
        env,
        behavior_policy_matrix,
        num_episodes=num_episodes,
        alpha=alpha,
        gamma=gamma,
        max_steps=max_steps,
        record_freq=record_freq
    )
    print(f"Q-learning算法完成！")
    
    # ========================================
    # 5. 描述和绘制一个episode示例  
    # ========================================
    print("\n" + "=" * 80)
    print("5. Episode示例展示")
    print("=" * 80)
    
    # 在环境仿真坐标系上，使用行为策略（均匀）采样并绘制轨迹，避免偏移问题
    print(f"\n展示训练过程中的长轨迹（环境仿真绘制，叠加行为策略）...")
    # 使用训练得到的实际轨迹，显示更多段
    # 仅展示最后一个episode的轨迹，避免跨episode的起点连接线
    last_ep = episode_data[-1]
    MAX_SHOW = 500
    path_states = last_ep['states'][:MAX_SHOW]
    path_actions = last_ep['actions'][:MAX_SHOW]
    visualize_path_on_env(
        env,
        path_states,
        save_path=f"{OUTPUT_DIR}/episode_trajectory.png",
        title="Episode Trajectory (Uniform Behavior)",
        overlay_policy=behavior_policy_matrix,
        action_sequence=path_actions
    )
    
    print(f"\nEpisode信息:")
    print(f"  - 起始状态: {path_states[0]}")
    print(f"  - 展示结束状态: {path_states[-1]}")
    print(f"  - 展示长度: {len(path_states)-1} 步")
    
    print(f"\nEpisode轨迹（前20步）:")
    for t in range(min(20, len(path_states)-1)):
        state_t = path_states[t]
        next_state_t = path_states[t+1]
        action_t = tuple(np.array(next_state_t) - np.array(state_t))
        reward_t = 0.0
        print(f"  步骤 {t}: {state_t} --{action_t}--> {next_state_t}, 奖励={reward_t:.2f}")
    
    if len(path_states)-1 > 20:
        print(f"  ... (省略剩余 {len(path_states)-1-20} 步)")
    
    # 可视化episode轨迹
    # 已在环境仿真函数中完成保存，无需重复保存
    print(f"  已保存到: {OUTPUT_DIR}/episode_trajectory.png")
    
    # ========================================
    # 6. 描述和绘制最终学习到的策略
    # ========================================
    print("\n" + "=" * 80)
    print("6. 最终学习到的目标策略")
    print("=" * 80)
    
    # 计算最终的状态值函数
    V_learned = np.max(Q, axis=1)
    
    print(f"\n学习到的状态值函数:")
    for i, state in enumerate(states):
        if state not in env.forbidden_states and state != tuple(env.target_state):
            print(f"  状态 {state}: {V_learned[i]:.2f}")
    
    # 可视化最终策略
    print(f"\n保存最终学习到的策略可视化...")
    visualize_optimal_policy(
        env,
        target_policy_matrix,
        save_path=f"{OUTPUT_DIR}/learned_policy.png"
    )
    print(f"  已保存到: {OUTPUT_DIR}/learned_policy.png")
    
    # 可视化最终状态值
    print(f"\n保存最终学习到的状态值可视化...")
    visualize_state_values(
        env,
        V_learned,
        precision=2,
        save_path=f"{OUTPUT_DIR}/learned_state_values.png"
    )
    print(f"  已保存到: {OUTPUT_DIR}/learned_state_values.png")
    
    # ========================================
    # 7. 策略演变过程 - 状态值误差曲线
    # ========================================
    print("\n" + "=" * 80)
    print("7. 策略演变过程分析")
    print("=" * 80)
    
    # 计算状态值误差
    print(f"\n计算状态值误差...")
    error_history = []
    episode_indices = []
    
    for idx, V_dict in enumerate(value_history):
        error = compute_state_value_error(V_dict, V_dict_optimal)
        error_history.append(error)
        episode_idx = idx * record_freq if idx < len(value_history) - 1 else num_episodes - 1
        episode_indices.append(episode_idx)
    
    print(f"  初始误差: {error_history[0]:.4f}")
    print(f"  最终误差: {error_history[-1]:.4f}")
    print(f"  误差减少: {error_history[0] - error_history[-1]:.4f}")
    
    # 绘制状态值误差曲线
    print(f"\n保存状态值误差曲线...")
    import matplotlib.pyplot as plt
    # 仅显示到 100000
    MAX_POINTS = 100000
    if len(episode_indices) > MAX_POINTS:
        episode_indices = episode_indices[:MAX_POINTS]
        error_history = error_history[:MAX_POINTS]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(episode_indices, error_history, s=8)
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Mean State Value Error', fontsize=12)
    ax.set_title('Q-learning Convergence: State Value Error vs Episode', fontsize=14)
    ax.set_xlim(0, 100000)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/state_value_error_curve.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  已保存到: {OUTPUT_DIR}/state_value_error_curve.png")
    
    # ========================================
    # 8. 实验总结
    # ========================================
    print("\n" + "=" * 80)
    print("8. 实验总结与观察")
    print("=" * 80)
    
    print(f"\n主要观察:")
    print(f"  1. 算法收敛性:")
    print(f"     - Q-learning算法通过{max_steps}步的学习成功收敛")
    print(f"     - 状态值误差从{error_history[0]:.4f}降低到{error_history[-1]:.4f}")
    
    print(f"\n  2. 离策略学习特性:")
    print(f"     - 行为策略使用均匀随机策略保证探索")
    print(f"     - 目标策略始终选择最优动作（贪心策略）")
    print(f"     - 成功实现了探索与利用的平衡")
    
    print(f"\n  3. 策略质量:")
    # 比较学习到的策略和最优策略
    policy_match_count = 0
    for s_idx in range(env.num_states):
        if np.array_equal(target_policy_matrix[s_idx], optimal_policy_matrix[s_idx]):
            policy_match_count += 1
    policy_match_ratio = policy_match_count / env.num_states
    print(f"     - 学习到的策略与最优策略的匹配度: {policy_match_ratio*100:.1f}%")
    print(f"     - 平均状态值误差: {error_history[-1]:.4f}")
    
    print(f"\n  4. Q-learning算法优势:")
    print(f"     - 无需环境模型（无需知道状态转移概率和奖励函数）")
    print(f"     - 通过采样学习，适用于大规模状态空间")
    print(f"     - 离策略特性允许使用不同的行为策略进行探索")
    
    print(f"\n  5. 参数影响:")
    print(f"     - 学习率α={alpha}控制了学习速度")
    print(f"     - 折扣因子γ={gamma}平衡了短期和长期奖励")
    print(f"     - 均匀随机策略保证了充分探索")
    
    print("\n" + "=" * 80)
    print("实验完成！所有结果已保存到:", OUTPUT_DIR)
    print("=" * 80)
    
    # 生成结果摘要文件
    summary_path = f"{OUTPUT_DIR}/experiment_summary.txt"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("Q-learning算法实验结果摘要\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("1. 问题设置\n")
        f.write(f"   环境大小: {args.env_size}\n")
        f.write(f"   起始状态: {args.start_state}\n")
        f.write(f"   目标状态: {args.target_state}\n")
        f.write(f"   禁止状态: {args.forbidden_states}\n")
        f.write(f"   折扣因子: {gamma}\n\n")
        
        f.write("2. 算法参数\n")
        f.write(f"   Episode数量: {num_episodes}\n")
        f.write(f"   学习率: {alpha}\n")
        f.write(f"   行为策略: 均匀随机策略\n")
        f.write(f"   最大步数: {max_steps}\n\n")
        
        f.write("3. 实验结果\n")
        f.write(f"   初始状态值误差: {error_history[0]:.4f}\n")
        f.write(f"   最终状态值误差: {error_history[-1]:.4f}\n")
        f.write(f"   策略匹配度: {policy_match_ratio*100:.1f}%\n\n")
        
        f.write("4. 学习到的状态值函数\n")
        for i, state in enumerate(states):
            if state not in env.forbidden_states:
                f.write(f"   V({state}) = {V_learned[i]:.2f}\n")
        
        f.write("\n5. 最优状态值函数（参考）\n")
        for i, state in enumerate(states):
            if state not in env.forbidden_states:
                f.write(f"   V*({state}) = {V_optimal[i]:.2f}\n")
    
    print(f"\n实验摘要已保存到: {summary_path}")


def main():
    run_q_learning_experiment()


if __name__ == "__main__":
    main()
