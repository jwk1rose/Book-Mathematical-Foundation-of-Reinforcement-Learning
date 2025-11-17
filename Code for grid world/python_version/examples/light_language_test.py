import math
import random
import matplotlib.pyplot as plt


# --- 1. 定义灯语强度 ---
class SignalStrength:
    RED_FLASH = 3  # 极佳位置 (High Priority)
    YELLOW_ON = 2  # 普通位置 (Medium Priority)
    GREEN_IDLE = 1  # 弃权/无效 (Low Priority)

    @staticmethod
    def to_string(val):
        if val == 3: return "🔴"
        if val == 2: return "🟡"
        return "🟢"


# --- 2. 分布式机器人 (支持 Gossip 中继) ---
class DistributedRobot:
    def __init__(self, robot_id, x, y):
        self.id = robot_id
        self.pos = (x, y)

        # --- 自身硬性条件 ---
        self.my_raw_bid = 0  # 我实际的物理评估 (灯语等级)

        # --- 认知状态 (关键) ---
        # 格式: (SignalStrength, -ID)
        # 使用 -ID 是因为 Python tuple 比较时，先比第一项，若相同比第二项。
        # 我们希望 ID 越小优先级越高，但在 max() 函数中数值越大越好，所以取负号。
        # 例如: ID 1 -> -1, ID 5 -> -5.   -1 > -5，所以 ID 1 胜出。
        self.knowledge_best_bid = (-1, -9999)

        self.is_active = True  # 我是否认为自己是当前的 Winner

    def evaluate_task(self, task_x, task_y):
        """ 初始阶段：计算距离并生成初始出价 """
        dist = math.sqrt((self.pos[0] - task_x) ** 2 + (self.pos[1] - task_y) ** 2)

        # 量化逻辑
        if dist < 30:
            self.my_raw_bid = SignalStrength.RED_FLASH
        elif dist < 60:
            self.my_raw_bid = SignalStrength.YELLOW_ON
        else:
            self.my_raw_bid = SignalStrength.GREEN_IDLE

        # 初始时刻，我认为全网最强就是我自己
        self.knowledge_best_bid = (self.my_raw_bid, -self.id)
        self.is_active = True

    def calculate_next_knowledge(self, neighbors):
        """
        Gossip 核心: 收集邻居信息，计算下一时刻的认知
        注意：这里不立即修改 self.knowledge，而是返回新值，用于同步更新
        """
        # 收集所有邻居目前知道的最强信息
        candidates = [self.knowledge_best_bid]  # 别忘了自己当前的认知

        for n in neighbors:
            # 即使邻居 inactive，它维护的 knowledge_best_bid 也是有价值的
            candidates.append(n.knowledge_best_bid)

        # 选出那个最强的 (Max-Consensus)
        # 逻辑: 先比灯光强度，强度一样比 ID (负ID越大越好)
        best_candidate = max(candidates)

        return best_candidate

    def update_state(self, new_knowledge):
        """ 应用更新并判断自己是否胜出 """
        self.knowledge_best_bid = new_knowledge

        # 判断逻辑: 如果"全网最强" 等于 "我的原始出价"，那我就是 Winner
        my_bid_tuple = (self.my_raw_bid, -self.id)

        if self.knowledge_best_bid == my_bid_tuple:
            self.is_active = True
        else:
            self.is_active = False

    def get_display_color(self):
        # 可视化：如果我是 active，显示我的等级颜色；否则显示绿色(已放弃)
        if self.is_active:
            return self.my_raw_bid
        else:
            return SignalStrength.GREEN_IDLE


# --- 3. 环境与仿真控制器 ---
class DecentralizedSwarm:
    def __init__(self, num_robots=15):
        self.robots = []
        # 随机分布
        for i in range(num_robots):
            self.robots.append(DistributedRobot(i, random.uniform(0, 100), random.uniform(0, 100)))

        # 通讯半径: 设置得适中，故意制造"非全连接"网络，测试中继能力
        self.comm_radius = 35.0

    def get_neighbors(self, robot):
        neighbors = []
        for other in self.robots:
            if robot.id == other.id: continue
            dist = math.sqrt((robot.pos[0] - other.pos[0]) ** 2 + (robot.pos[1] - other.pos[1]) ** 2)
            if dist <= self.comm_radius:
                neighbors.append(other)
        return neighbors

    def run_consensus(self, task_x, task_y, max_rounds=15):
        print(f"任务发布于: ({task_x:.1f}, {task_y:.1f}) | 通讯半径: {self.comm_radius}")

        # 1. 初始评估
        for r in self.robots:
            r.evaluate_task(task_x, task_y)

        history = []

        # 2. 循环迭代
        for round_idx in range(max_rounds):
            # --- 记录快照 ---
            snapshot = []
            active_ids = []
            for r in self.robots:
                snapshot.append({
                    'x': r.pos[0], 'y': r.pos[1],
                    'color': r.get_display_color(),
                    'id': r.id,
                    'active': r.is_active,
                    'knowledge': r.knowledge_best_bid  # 调试用
                })
                if r.is_active: active_ids.append(r.id)
            history.append(snapshot)

            print(f"--- Round {round_idx:2d} --- 活跃ID: {active_ids}")

            # 检查收敛：如果只剩1个活跃者，且持续稳定（这里简单起见，剩1个就停）
            # 在实际中通常会多跑几轮以确保传播完成
            if len(active_ids) == 1 and round_idx > 2:
                print(f"✅ 共识达成! 胜者: Robot {active_ids[0]}")
                break
            if len(active_ids) == 0:
                print("❌ 异常: 全员放弃 (距离太远?)")
                break

            # --- 同步更新步骤 (Synchronous Update) ---

            # A. 计算阶段：所有机器人根据当前邻居计算"下一时刻认知"
            next_knowledges = {}
            for r in self.robots:
                neighbors = self.get_neighbors(r)
                next_knowledges[r.id] = r.calculate_next_knowledge(neighbors)

            # B. 更新阶段：所有机器人同时应用更新
            for r in self.robots:
                r.update_state(next_knowledges[r.id])

        return history

    def visualize_steps(self, task_pos, history):
        """ 可视化: 选取 起始、中间、结束 三个关键帧 """
        total_rounds = len(history)
        indices = [0, total_rounds // 2, total_rounds - 1]
        # 去重并排序，防止轮数太少导致报错
        indices = sorted(list(set(indices)))

        cols = len(indices)
        fig, axes = plt.subplots(1, cols, figsize=(5 * cols, 5))
        if cols == 1: axes = [axes]

        for i, idx in enumerate(indices):
            ax = axes[i]
            data = history[idx]

            ax.set_xlim(0, 100);
            ax.set_ylim(0, 100)
            ax.grid(True, linestyle=':', alpha=0.5)
            ax.set_title(f"Round {idx}")

            # 画任务
            ax.scatter(task_pos[0], task_pos[1], marker='*', s=300, c='purple', label='Task')

            # 画机器人
            for bot in data:
                # 颜色逻辑
                c = 'lightgreen'  # 默认放弃
                s = 60
                lw = 1
                edge = 'gray'
                z = 5

                if bot['active']:
                    if bot['color'] == SignalStrength.RED_FLASH:
                        c = 'red';
                        s = 200;
                        lw = 2;
                        edge = 'black';
                        z = 10
                    elif bot['color'] == SignalStrength.YELLOW_ON:
                        c = 'orange';
                        s = 150;
                        lw = 2;
                        edge = 'black';
                        z = 10

                # 画点
                ax.scatter(bot['x'], bot['y'], c=c, s=s, edgecolors=edge, linewidths=lw, zorder=z)
                # 画ID
                ax.text(bot['x'], bot['y'], str(bot['id']), fontsize=9, fontweight='bold', zorder=z + 1)

                # 可选：画出机器人之间的连线，表示网络连通性 (第一张图画一下)
                if i == 0 and False:
                    pass

        plt.tight_layout()
        plt.show()


# --- 运行主程序 ---
if __name__ == "__main__":
    # 设置随机种子以便复现 (可选)
    # random.seed(42)

    # 1. 初始化：增加机器人数量，确保大概率形成连通图
    swarm = DecentralizedSwarm(num_robots=15)

    # 2. 任务位置
    tx, ty = 50, 50

    # 3. 运行
    hist = swarm.run_consensus(tx, ty, max_rounds=15)

    # 4. 绘图
    swarm.visualize_steps((tx, ty), hist)