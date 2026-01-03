import matplotlib
matplotlib.use('Agg')

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
import torch.nn as nn
import torch.optim as optim
import random
import math
import heapq
from collections import deque, namedtuple
import copy
import time
import os
from tqdm import tqdm


# ==========================================
# 1. Configuration
# ==========================================
class Config:
    # --- Experiment Settings ---
    TRAIN_STEPS = 10000    # 訓練總步數 (非 Episode，因為是連續流)
    EVAL_INTERVAL = 1000   # 每幾步評估一次
    SEED = 42
    
    # --- Environment Dimensions ---
    NUM_MACHINES = 5
    NUM_OPS_PER_JOB = 5
    
    # --- Job Arrival (Poisson) ---
    ARRIVAL_RATE = 0.15    # Lambda: 平均每 1/0.15 = 6.6 單位時間來一個工件
    DUE_DATE_FACTOR = 1.5  # Due Date = Arrival + Factor * Total_Proc_Time
    
    # --- Machine Deterioration (Discrete Multi-state) ---
    K_STATES = 5           # 狀態 0 (New) ~ 5 (Failed)
    STATE_DEGRADE_PROB = 0.2 # 每次加工後狀態惡化的基礎機率
    PROC_TIME_PENALTY = 0.1  # 狀態每增加 1，加工時間增加 10%
    
    # --- Dual Failure Modes ---
    BREAKDOWN_RATE = 0.005 # 隨機故障率 (Poisson)
    
    # --- Maintenance Specs ---
    TIME_PM = 10           # 預防性維護 (完美修復: State -> 0)
    TIME_CM = 30           # 強制大修 (狀態 K -> 0)
    TIME_MINIMAL = 5       # 最小修復 (隨機故障 -> 狀態不變)
    
    Q_LIMIT = 2            # 維修工數量限制
    
    # --- Reward Weights ---
    W_TARDINESS = 1.0
    W_MAINT_COST = 0.5
    PENALTY_INVALID = 10.0 # 資源不足卻想修的懲罰
    
    # --- RL Hyperparameters ---
    LR = 1e-4
    GAMMA = 0.99
    BUFFER_SIZE = 10000
    BATCH_SIZE = 64
    EPSILON_START = 1.0
    EPSILON_END = 0.05
    EPSILON_DECAY = 5000
    TARGET_UPDATE = 200
    HIDDEN_DIM = 128

# ==========================================
# 2. Core Classes
# ==========================================
class Job:
    def __init__(self, job_id, arrival_time, ops_times):
        self.id = job_id
        self.arrival_time = arrival_time
        self.ops_times = ops_times # Base processing times
        self.num_ops = len(ops_times)
        self.current_op_idx = 0
        self.finished = False
        self.completion_time = 0
        
        # 設定交期 (Due Date)
        total_proc = sum(ops_times)
        self.due_date = arrival_time + total_proc * Config.DUE_DATE_FACTOR

    def get_base_proc_time(self):
        if self.finished: return 0
        return self.ops_times[self.current_op_idx]
    
    def get_tardiness(self, current_time):
        # 若未完成，計算當前延遲；若已完成，計算最終延遲
        ref_time = self.completion_time if self.finished else current_time
        return max(0, ref_time - self.due_date)

class Machine:
    def __init__(self, m_id):
        self.id = m_id
        self.state = 0       # 0 (New) to K (Failed)
        self.status = 0      # 0: Idle, 1: Busy, 2: Down/Maint
        self.age_accum = 0.0 # 累積加工時間 (影響狀態轉移機率)
        self.history = []    # 畫圖用

    def get_actual_proc_time(self, base_time):
        # 狀態越差，加工越慢 (Ghaleb et al. 概念)
        # Time = Base * (1 + state * penalty)
        return base_time * (1.0 + self.state * Config.PROC_TIME_PENALTY)

    def degrade(self):
        # 模擬老化：基於累積加工時間的機率性轉移
        # 簡單模型：機率隨 age_accum 增加
        prob = Config.STATE_DEGRADE_PROB + (self.age_accum * 0.01)
        if random.random() < prob and self.state < Config.K_STATES:
            self.state += 1
            self.age_accum = 0 # 狀態改變後，累積歸零重新計算
        return self.state >= Config.K_STATES # 回傳是否故障

    def repair_perfect(self):
        self.state = 0
        self.age_accum = 0
        
    def repair_minimal(self):
        # 最小修復：狀態不變
        pass

# ==========================================
# 3. Advanced Event-Driven Environment
# ==========================================
class AdvancedDFJSPEnv(gym.Env):
    def __init__(self):
        super(AdvancedDFJSPEnv, self).__init__()
        
        # Action Space: 8 Discrete Actions
        # [Rule (4)] x [Maint (2)]
        # Rules: 0:FIFO, 1:SPT, 2:EDD, 3:SRPT
        # Maint: 0:No, 1:PM
        self.action_space = spaces.Discrete(8)
        
        # State Space: 
        # Machine Feats (3*M): [State/K, Status, Norm_Age] * M
        # Buffer Feats (3): [Queue_Len, Avg_Tardiness, Avg_Proc_Time]
        # Resource Feats (1): [Avail_Crew_Ratio]
        self.state_dim = (3 * Config.NUM_MACHINES) + 3 + 1
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.state_dim,), dtype=np.float32)
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            
        self.machines = [Machine(i) for i in range(Config.NUM_MACHINES)]
        self.job_buffer = []     # 等待加工的工件 (Visible Jobs)
        self.active_jobs = []    # 系統中所有未完成工件 (計算 Reward 用)
        self.finished_jobs = []  # 已完成工件
        
        self.now = 0.0
        self.event_queue = []    # (time, type, data...)
        self.avail_crews = Config.Q_LIMIT
        self.job_counter = 0
        
        # 初始事件：第一個工件到達
        self._schedule_next_arrival()
        
        # 快轉到第一個決策點
        self.decision_machine_id = self._resume_simulation()
        
        return self._get_state(), {}

    def _schedule_next_arrival(self):
        # Poisson Arrival: Inter-arrival time ~ Exponential(lambda)
        inter_arrival = random.expovariate(Config.ARRIVAL_RATE)
        arrival_time = self.now + inter_arrival
        heapq.heappush(self.event_queue, (arrival_time, 'JOB_ARRIVAL', None))

    def _resume_simulation(self):
        """
        DES 核心：推進時間直到需要 Agent 決策 (Machine Idle & Buffer not empty)
        """
        while True:
            # 檢查是否觸發決策條件
            # 條件：有機器閒置 且 有工件在排隊
            idle_machines = [m.id for m in self.machines if m.status == 0]
            if idle_machines and self.job_buffer:
                return idle_machines[0] # 回傳第一台閒置機器 ID 供 Agent 決策

            if not self.event_queue:
                # 應該不會發生 (因為有無限工件流)，防呆
                self._schedule_next_arrival()
                continue
                
            # 取出事件
            time_stamp, event_type, data = heapq.heappop(self.event_queue)
            self.now = time_stamp
            
            if event_type == 'JOB_ARRIVAL':
                # 生成新工件
                ops = [random.randint(1, 10) for _ in range(Config.NUM_OPS_PER_JOB)]
                new_job = Job(self.job_counter, self.now, ops)
                self.job_counter += 1
                
                self.job_buffer.append(new_job)
                self.active_jobs.append(new_job)
                
                # 安排下一個到達
                self._schedule_next_arrival()
                
            elif event_type == 'JOB_FINISH':
                m_id, j_id = data
                machine = self.machines[m_id]
                job = next((j for j in self.active_jobs if j.id == j_id), None)
                
                # 更新工件
                if job:
                    job.current_op_idx += 1
                    if job.current_op_idx >= job.num_ops:
                        job.finished = True
                        job.completion_time = self.now
                        self.active_jobs.remove(job)
                        self.finished_jobs.append(job)
                    else:
                        # 工件回到 Buffer 等待下一道工序
                        self.job_buffer.append(job)
                
                # 更新機器狀態 (老化檢查)
                machine.status = 0 # Idle
                is_failed = machine.degrade()
                
                if is_failed:
                    # 觸發老化故障 -> 強制大修 (CM)
                    # 檢查資源
                    if self.avail_crews > 0:
                        self.avail_crews -= 1
                        machine.status = 2 # Down
                        finish_time = self.now + Config.TIME_CM
                        heapq.heappush(self.event_queue, (finish_time, 'MAINT_FINISH', (m_id, 'CM')))
                        machine.history.append((-1, -1, self.now, finish_time, 'CM'))
                    else:
                        # 資源不足，進入等待隊列 (簡化：直接推遲 1.0 秒再檢查)
                        machine.status = 2 # Down (Waiting)
                        heapq.heappush(self.event_queue, (self.now + 1.0, 'WAIT_RESOURCE', m_id))

            elif event_type == 'RANDOM_BREAKDOWN':
                # 隨機故障發生
                m_id = data
                machine = self.machines[m_id]
                
                # 只有在加工中才會壞 (簡化假設)
                if machine.status == 1: 
                    # 找到正在做的工件，推遲其完成時間
                    # 這裡簡化處理：直接插入最小修復時間，不中斷工件
                    # 實務上應該要 Split job，這裡為了代碼簡潔，視為機器被鎖定
                    if self.avail_crews > 0:
                        self.avail_crews -= 1
                        machine.status = 2
                        finish_time = self.now + Config.TIME_MINIMAL
                        heapq.heappush(self.event_queue, (finish_time, 'MAINT_FINISH', (m_id, 'MINIMAL')))
                        machine.history.append((-1, -1, self.now, finish_time, 'MINIMAL'))
                        
                        # 原本的 JOB_FINISH 事件需要推遲
                        # (這部分實作較複雜，此處簡化：假設隨機故障只發生在 Idle 時或不影響當前工件完成時間)
                        # 為了保持 DES 簡單，我們假設隨機故障只在 Idle 時被偵測到
                    else:
                        # 沒人修，稍後再試
                        heapq.heappush(self.event_queue, (self.now + 5.0, 'RANDOM_BREAKDOWN', m_id))

            elif event_type == 'MAINT_FINISH':
                m_id, m_type = data
                machine = self.machines[m_id]
                self.avail_crews += 1
                machine.status = 0 # Idle
                
                if m_type == 'PM' or m_type == 'CM':
                    machine.repair_perfect()
                elif m_type == 'MINIMAL':
                    machine.repair_minimal()
            
            elif event_type == 'WAIT_RESOURCE':
                m_id = data
                # 再次檢查資源
                if self.avail_crews > 0:
                    self.avail_crews -= 1
                    self.machines[m_id].status = 2
                    finish_time = self.now + Config.TIME_CM
                    heapq.heappush(self.event_queue, (finish_time, 'MAINT_FINISH', (m_id, 'CM')))
                    self.machines[m_id].history.append((-1, -1, self.now, finish_time, 'CM'))
                else:
                    heapq.heappush(self.event_queue, (self.now + 1.0, 'WAIT_RESOURCE', m_id))

            # 隨機生成故障事件 (Poisson Process)
            # 每個時間步都有小機率發生
            if random.random() < Config.BREAKDOWN_RATE:
                target_m = random.choice(self.machines)
                if target_m.status == 1: # 只有忙碌時會壞
                    # 簡化：插入一個故障事件
                    # 實作細節：這裡略過複雜的工件中斷邏輯，僅作為概念展示
                    pass 

    def step(self, action):
        machine = self.machines[self.decision_machine_id]
        
        # 解析動作
        rule_idx = action // 2      # 0~3
        do_pm = (action % 2 == 1)   # True/False
        
        reward = 0
        done = False
        
        # 1. 執行維護決策
        if do_pm:
            if self.avail_crews > 0:
                # 執行 PM
                self.avail_crews -= 1
                machine.status = 2
                finish_time = self.now + Config.TIME_PM
                heapq.heappush(self.event_queue, (finish_time, 'MAINT_FINISH', (machine.id, 'PM')))
                machine.history.append((-1, -1, self.now, finish_time, 'PM'))
                
                # 扣除維護成本
                reward -= Config.W_MAINT_COST
                
                # 機器進入維護，本回合結束，快轉
                self.decision_machine_id = self._resume_simulation()
                return self._get_state(), reward, False, False, {}
            else:
                # 資源不足卻想修 -> 懲罰並強制轉為生產
                reward -= Config.PENALTY_INVALID
                do_pm = False # 強制取消
        
        # 2. 執行派工決策
        if not self.job_buffer:
            # 異常：無工件 (應被 resume 過濾)
            self.decision_machine_id = self._resume_simulation()
            return self._get_state(), reward, False, False, {}
            
        # 應用規則選擇工件
        selected_job = self._apply_rule(rule_idx, self.job_buffer)
        self.job_buffer.remove(selected_job) # 從 Buffer 移除
        
        # 計算加工時間
        base_time = selected_job.get_base_proc_time()
        actual_time = machine.get_actual_proc_time(base_time)
        
        # 更新狀態
        machine.status = 1 # Busy
        machine.age_accum += actual_time
        finish_time = self.now + actual_time
        
        # 記錄
        machine.history.append((selected_job.id, selected_job.current_op_idx, self.now, finish_time, 'JOB'))
        heapq.heappush(self.event_queue, (finish_time, 'JOB_FINISH', (machine.id, selected_job.id)))
        
        # 3. 計算 Tardiness Reward
        # 針對系統中所有 Active Jobs 計算當前延遲懲罰
        current_tardiness = sum([j.get_tardiness(self.now) for j in self.active_jobs])
        reward -= Config.W_TARDINESS * current_tardiness
        
        # 4. 快轉
        self.decision_machine_id = self._resume_simulation()
        
        return self._get_state(), reward, False, False, {}

    def _apply_rule(self, rule_idx, jobs):
        if rule_idx == 0: # FIFO
            return min(jobs, key=lambda j: j.arrival_time)
        elif rule_idx == 1: # SPT
            return min(jobs, key=lambda j: j.get_base_proc_time())
        elif rule_idx == 2: # EDD
            return min(jobs, key=lambda j: j.due_date)
        elif rule_idx == 3: # SRPT (Remaining)
            return min(jobs, key=lambda j: sum(j.ops_times[j.current_op_idx:]))
        return jobs[0]

    def _get_state(self):
        # 1. Machine Features (3 * M)
        m_feats = []
        for m in self.machines:
            m_feats.extend([
                m.state / Config.K_STATES,     # Normalized State
                1.0 if m.status == 1 else 0.0, # Is Busy?
                0.0 # Placeholder for age_accum if needed
            ])
            
        # 2. Buffer Features (3)
        if self.job_buffer:
            q_len = len(self.job_buffer) / 20.0 # Norm
            avg_tard = np.mean([j.get_tardiness(self.now) for j in self.job_buffer]) / 50.0
            avg_proc = np.mean([j.get_base_proc_time() for j in self.job_buffer]) / 20.0
        else:
            q_len, avg_tard, avg_proc = 0, 0, 0
            
        # 3. Resource Features (1)
        crew_ratio = self.avail_crews / Config.Q_LIMIT
        
        state = np.array(m_feats + [q_len, avg_tard, avg_proc, crew_ratio], dtype=np.float32)
        return state

# ==========================================
# 4. Agent & Training
# ==========================================
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    def forward(self, x): return self.net(x)

class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.policy_net = DQN(state_dim, action_dim)
        self.target_net = DQN(state_dim, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=Config.LR)
        self.memory = deque(maxlen=Config.BUFFER_SIZE)
        self.steps = 0
        self.action_dim = action_dim
        
    def select_action(self, state, training=True):
        eps = Config.EPSILON_END + (Config.EPSILON_START - Config.EPSILON_END) * \
              math.exp(-1. * self.steps / Config.EPSILON_DECAY)
        self.steps += 1
        if training and random.random() < eps: return random.randrange(self.action_dim)
        with torch.no_grad(): return self.policy_net(torch.FloatTensor(state)).argmax().item()

    def update(self):
        if len(self.memory) < Config.BATCH_SIZE: return
        batch = random.sample(self.memory, Config.BATCH_SIZE)
        state, action, reward, next_state, done = zip(*batch)
        
        state = torch.FloatTensor(np.array(state))
        action = torch.LongTensor(action).unsqueeze(1)
        reward = torch.FloatTensor(reward).unsqueeze(1)
        next_state = torch.FloatTensor(np.array(next_state))
        done = torch.FloatTensor(done).unsqueeze(1)
        
        q_val = self.policy_net(state).gather(1, action)
        next_q = self.target_net(next_state).max(1)[0].unsqueeze(1)
        expected_q = reward + Config.GAMMA * next_q * (1 - done)
        
        loss = nn.MSELoss()(q_val, expected_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

# ==========================================
# 5. Main Execution
# ==========================================
def run_advanced_experiment():
    print("Starting Advanced Integrated Scheduling (Sim-to-Real)...")
    env = AdvancedDFJSPEnv()
    agent = DQNAgent(env.observation_space.shape[0], env.action_space.n)
    
    rewards = []
    avg_tardiness = []
    
    state, _ = env.reset(seed=Config.SEED)
    total_reward = 0
    
    # 連續訓練迴圈 (Continuous Training)
    pbar = tqdm(range(Config.TRAIN_STEPS))
    for step in pbar:
        action = agent.select_action(state)
        next_state, r, done, _, _ = env.step(action)
        
        agent.memory.append((state, action, r, next_state, done))
        agent.update()
        
        state = next_state
        total_reward += r
        
        if step % Config.TARGET_UPDATE == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
            
        if step % Config.EVAL_INTERVAL == 0:
            # 記錄當前平均延遲
            if env.finished_jobs:
                avg_t = np.mean([j.get_tardiness(j.completion_time) for j in env.finished_jobs[-50:]])
                avg_tardiness.append(avg_t)
                pbar.set_description(f"Step {step} | Avg Tardiness: {avg_t:.2f} | R: {total_reward:.1f}")
            rewards.append(total_reward)
            total_reward = 0 # Reset for next interval

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(avg_tardiness)
    plt.title("Advanced Model: Average Tardiness Trend")
    plt.xlabel("Evaluation Interval (x1000 steps)")
    plt.ylabel("Avg Tardiness (Lower is Better)")
    plt.grid(True)
    plt.savefig("advanced_tardiness.png")
    print("Saved advanced_tardiness.png")
    
    # Gantt Chart (Snapshot)
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.tab20.colors
    # 只畫最後一段時間的甘特圖
    time_window_start = env.now - 200
    for m in env.machines:
        for h in m.history:
            start, end, type_ = h[2], h[3], h[4]
            if end < time_window_start: continue # Skip old history
            
            dur = end - start
            if 'JOB' in type_:
                job_id = h[0]
                ax.add_patch(mpatches.Rectangle((start, m.id*10), dur, 9, facecolor=colors[job_id%20], edgecolor='black'))
                if dur > 2:
                    ax.text(start+dur/2, m.id*10+4.5, f"J{job_id}", ha='center', va='center', color='white', fontsize=8)
            elif 'PM' in type_:
                ax.add_patch(mpatches.Rectangle((start, m.id*10), dur, 9, facecolor='green', hatch='//', edgecolor='black'))
                ax.text(start+dur/2, m.id*10+4.5, "PM", ha='center', va='center', color='white', fontsize=8)
            elif 'CM' in type_:
                ax.add_patch(mpatches.Rectangle((start, m.id*10), dur, 9, facecolor='red', hatch='xx', edgecolor='black'))
                ax.text(start+dur/2, m.id*10+4.5, "CM", ha='center', va='center', color='white', fontsize=8)
                
    ax.set_yticks([i*10+5 for i in range(Config.NUM_MACHINES)])
    ax.set_yticklabels([f'M{i}' for i in range(Config.NUM_MACHINES)])
    ax.set_xlabel("Time")
    ax.set_xlim(time_window_start, env.now)
    ax.set_title(f"Advanced Schedule Snapshot (Time {env.now:.0f})")
    plt.savefig("advanced_gantt.png")
    print("Saved advanced_gantt.png")

if __name__ == "__main__":
    run_advanced_experiment()