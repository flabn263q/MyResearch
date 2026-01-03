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

# 防呆機制
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterator, *args, **kwargs):
        return iterator

# ==========================================
# 1. Configuration
# ==========================================
class Config:
    # --- Experiment Settings ---
    # [修正 4] 增加訓練步數以確保收斂
    TRAIN_STEPS = 50000    
    EVAL_INTERVAL = 1000   
    SEED = 42
    
    # --- Environment Dimensions ---
    NUM_MACHINES = 5
    NUM_OPS_PER_JOB = 5
    
    # --- Job Arrival (Poisson) ---
    ARRIVAL_RATE = 0.15    
    DUE_DATE_FACTOR = 1.5  
    
    # --- Machine Deterioration ---
    K_STATES = 5           
    STATE_DEGRADE_PROB = 0.2 
    PROC_TIME_PENALTY = 0.1  
    
    # --- Dual Failure Modes ---
    BREAKDOWN_RATE = 0.005 # Lambda for breakdown
    
    # --- Maintenance Specs ---
    TIME_PM = 10           
    TIME_CM = 30           
    TIME_MINIMAL = 5       
    
    Q_LIMIT = 2            
    
    # --- Reward Weights ---
    W_TARDINESS = 1.0
    W_MAINT_COST = 0.5
    # PENALTY_INVALID 已移除，改用 Masking
    
    # --- RL Hyperparameters ---
    LR = 1e-4
    GAMMA = 0.99
    BUFFER_SIZE = 10000
    BATCH_SIZE = 64
    EPSILON_START = 1.0
    EPSILON_END = 0.05
    # [修正 4] 調整 Decay 讓 Agent 有更多時間利用策略
    EPSILON_DECAY = 10000 
    TARGET_UPDATE = 200
    HIDDEN_DIM = 128

# ==========================================
# 2. Core Classes
# ==========================================
class Job:
    def __init__(self, job_id, arrival_time, ops_times):
        self.id = job_id
        self.arrival_time = arrival_time
        self.ops_times = ops_times 
        self.num_ops = len(ops_times)
        self.current_op_idx = 0
        self.finished = False
        self.completion_time = 0
        
        total_proc = sum(ops_times)
        self.due_date = arrival_time + total_proc * Config.DUE_DATE_FACTOR

    def get_base_proc_time(self):
        if self.finished: return 0
        return self.ops_times[self.current_op_idx]
    
    def get_tardiness(self, current_time):
        ref_time = self.completion_time if self.finished else current_time
        return max(0, ref_time - self.due_date)

class Machine:
    def __init__(self, m_id):
        self.id = m_id
        self.state = 0       
        self.status = 0      # 0: Idle, 1: Busy, 2: Down/Maint
        self.age_accum = 0.0 
        self.history = []    

    def get_actual_proc_time(self, base_time):
        return base_time * (1.0 + self.state * Config.PROC_TIME_PENALTY)

    def degrade(self):
        prob = Config.STATE_DEGRADE_PROB + (self.age_accum * 0.01)
        if random.random() < prob and self.state < Config.K_STATES:
            self.state += 1
            self.age_accum = 0 
        return self.state >= Config.K_STATES 

    def repair_perfect(self):
        self.state = 0
        self.age_accum = 0
        
    def repair_minimal(self):
        pass

# ==========================================
# 3. Advanced Event-Driven Environment
# ==========================================
class AdvancedDFJSPEnv(gym.Env):
    def __init__(self):
        super(AdvancedDFJSPEnv, self).__init__()
        
        # Action Space: 8 Discrete Actions
        # 0-3: Rules without PM
        # 4-7: Rules with PM
        self.action_space = spaces.Discrete(8)
        
        # State Space
        self.state_dim = (3 * Config.NUM_MACHINES) + 3 + 1
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.state_dim,), dtype=np.float32)
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            
        self.machines = [Machine(i) for i in range(Config.NUM_MACHINES)]
        self.job_buffer = []     
        self.active_jobs = []    
        self.finished_jobs = []  
        
        self.now = 0.0
        self.event_queue = []    
        self.avail_crews = Config.Q_LIMIT
        self.job_counter = 0
        
        # 初始事件
        self._schedule_next_arrival()
        # [修正 1] 初始安排隨機故障
        self._schedule_next_breakdown()
        
        # 快轉到第一個決策點
        self.decision_machine_id = self._resume_simulation()
        
        return self._get_state(), {}

    def _schedule_next_arrival(self):
        inter_arrival = random.expovariate(Config.ARRIVAL_RATE)
        arrival_time = self.now + inter_arrival
        heapq.heappush(self.event_queue, (arrival_time, 'JOB_ARRIVAL', None))

    # [修正 1] 新增隨機故障排程
    def _schedule_next_breakdown(self):
        # 全系統共享一個故障流 (Poisson Process)
        inter_breakdown = random.expovariate(Config.BREAKDOWN_RATE)
        fail_time = self.now + inter_breakdown
        heapq.heappush(self.event_queue, (fail_time, 'RANDOM_BREAKDOWN', None))

    def _resume_simulation(self):
        while True:
            # 決策觸發條件
            idle_machines = [m.id for m in self.machines if m.status == 0]
            if idle_machines and self.job_buffer:
                return idle_machines[0] 

            if not self.event_queue:
                self._schedule_next_arrival()
                continue
                
            time_stamp, event_type, data = heapq.heappop(self.event_queue)
            self.now = time_stamp
            
            if event_type == 'JOB_ARRIVAL':
                ops = [random.randint(1, 10) for _ in range(Config.NUM_OPS_PER_JOB)]
                new_job = Job(self.job_counter, self.now, ops)
                self.job_counter += 1
                self.job_buffer.append(new_job)
                self.active_jobs.append(new_job)
                self._schedule_next_arrival()
                
            elif event_type == 'JOB_FINISH':
                m_id, j_id = data
                machine = self.machines[m_id]
                job = next((j for j in self.active_jobs if j.id == j_id), None)
                
                if job:
                    job.current_op_idx += 1
                    if job.current_op_idx >= job.num_ops:
                        job.finished = True
                        job.completion_time = self.now
                        self.active_jobs.remove(job)
                        self.finished_jobs.append(job)
                    else:
                        self.job_buffer.append(job)
                
                machine.status = 0 # Idle
                is_failed = machine.degrade()
                
                if is_failed:
                    # 老化故障 -> 強制 CM
                    if self.avail_crews > 0:
                        self.avail_crews -= 1
                        machine.status = 2 
                        finish_time = self.now + Config.TIME_CM
                        heapq.heappush(self.event_queue, (finish_time, 'MAINT_FINISH', (m_id, 'CM')))
                        machine.history.append((-1, -1, self.now, finish_time, 'CM'))
                    else:
                        machine.status = 2 
                        heapq.heappush(self.event_queue, (self.now + 1.0, 'WAIT_RESOURCE', m_id))

            elif event_type == 'RANDOM_BREAKDOWN':
                # [修正 1] 處理隨機故障
                # 隨機挑選一台機器，如果是忙碌狀態則故障
                # 如果沒有機器在忙，這次故障就"pass"掉，但仍需排程下一次
                busy_machines = [m for m in self.machines if m.status == 1]
                
                if busy_machines:
                    target_m = random.choice(busy_machines)
                    if self.avail_crews > 0:
                        self.avail_crews -= 1
                        target_m.status = 2
                        finish_time = self.now + Config.TIME_MINIMAL
                        heapq.heappush(self.event_queue, (finish_time, 'MAINT_FINISH', (target_m.id, 'MINIMAL')))
                        target_m.history.append((-1, -1, self.now, finish_time, 'MINIMAL'))
                        # 簡化：假設故障不影響當前工件的完成時間 (或視為工件被延後)
                    else:
                        # 沒人修，稍後再試 (Retry breakdown event)
                        heapq.heappush(self.event_queue, (self.now + 5.0, 'RANDOM_BREAKDOWN', None))
                        # 注意：這裡不呼叫 _schedule_next_breakdown，因為這是 Retry
                        continue 
                
                # 安排下一次隨機故障
                self._schedule_next_breakdown()

            elif event_type == 'MAINT_FINISH':
                m_id, m_type = data
                machine = self.machines[m_id]
                self.avail_crews += 1
                machine.status = 0 
                
                if m_type == 'PM' or m_type == 'CM':
                    machine.repair_perfect()
                elif m_type == 'MINIMAL':
                    machine.repair_minimal()
            
            elif event_type == 'WAIT_RESOURCE':
                m_id = data
                if self.avail_crews > 0:
                    self.avail_crews -= 1
                    self.machines[m_id].status = 2
                    finish_time = self.now + Config.TIME_CM
                    heapq.heappush(self.event_queue, (finish_time, 'MAINT_FINISH', (m_id, 'CM')))
                    self.machines[m_id].history.append((-1, -1, self.now, finish_time, 'CM'))
                else:
                    heapq.heappush(self.event_queue, (self.now + 1.0, 'WAIT_RESOURCE', m_id))

    def step(self, action):
        machine = self.machines[self.decision_machine_id]
        
        rule_idx = action % 4      
        do_pm = (action >= 4)   
        
        reward = 0
        
        # 1. 執行維護決策
        # [修正 2] 由於有 Masking，這裡不需要再檢查資源不足的情況
        # Agent 選了 PM，代表一定有資源
        if do_pm:
            self.avail_crews -= 1
            machine.status = 2
            finish_time = self.now + Config.TIME_PM
            heapq.heappush(self.event_queue, (finish_time, 'MAINT_FINISH', (machine.id, 'PM')))
            machine.history.append((-1, -1, self.now, finish_time, 'PM'))
            
            reward -= Config.W_MAINT_COST
            
            self.decision_machine_id = self._resume_simulation()
            return self._get_state(), reward, False, False, {}
        
        # 2. 執行派工決策
        if not self.job_buffer:
            self.decision_machine_id = self._resume_simulation()
            return self._get_state(), reward, False, False, {}
            
        selected_job = self._apply_rule(rule_idx, self.job_buffer)
        self.job_buffer.remove(selected_job) 
        
        base_time = selected_job.get_base_proc_time()
        actual_time = machine.get_actual_proc_time(base_time)
        
        machine.status = 1 
        machine.age_accum += actual_time
        finish_time = self.now + actual_time
        
        machine.history.append((selected_job.id, selected_job.current_op_idx, self.now, finish_time, 'JOB'))
        heapq.heappush(self.event_queue, (finish_time, 'JOB_FINISH', (machine.id, selected_job.id)))
        
        # 3. 計算 Tardiness Reward
        current_tardiness = sum([j.get_tardiness(self.now) for j in self.active_jobs])
        reward -= Config.W_TARDINESS * current_tardiness
        
        self.decision_machine_id = self._resume_simulation()
        
        return self._get_state(), reward, False, False, {}

    def _apply_rule(self, rule_idx, jobs):
        if rule_idx == 0: # FIFO
            return min(jobs, key=lambda j: j.arrival_time)
        elif rule_idx == 1: # SPT
            return min(jobs, key=lambda j: j.get_base_proc_time())
        elif rule_idx == 2: # EDD
            return min(jobs, key=lambda j: j.due_date)
        elif rule_idx == 3: # SRPT
            return min(jobs, key=lambda j: sum(j.ops_times[j.current_op_idx:]))
        return jobs[0]

    def _get_state(self):
        # 1. Machine Features
        m_feats = []
        for m in self.machines:
            m_feats.extend([
                m.state / Config.K_STATES,     
                1.0 if m.status == 1 else 0.0, 
                0.0 
            ])
            
        # 2. Buffer Features
        # [修正 3] 使用 tanh 進行數值穩定化
        if self.job_buffer:
            q_len = np.tanh(len(self.job_buffer) / 10.0) 
            avg_tard = np.tanh(np.mean([j.get_tardiness(self.now) for j in self.job_buffer]) / 50.0)
            avg_proc = np.tanh(np.mean([j.get_base_proc_time() for j in self.job_buffer]) / 20.0)
        else:
            q_len, avg_tard, avg_proc = 0, 0, 0
            
        # 3. Resource Features
        crew_ratio = self.avail_crews / Config.Q_LIMIT
        
        state = np.array(m_feats + [q_len, avg_tard, avg_proc, crew_ratio], dtype=np.float32)
        
        # [修正 2] 產生 Action Mask
        # 0-3: No PM (Always valid if buffer not empty)
        # 4-7: PM (Valid only if crew > 0)
        mask = np.ones(8, dtype=np.float32)
        if self.avail_crews <= 0:
            mask[4:] = 0.0 # 封鎖 PM 動作
            
        return state, mask

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
        
    def select_action(self, state, mask, training=True):
        eps = Config.EPSILON_END + (Config.EPSILON_START - Config.EPSILON_END) * \
              math.exp(-1. * self.steps / Config.EPSILON_DECAY)
        self.steps += 1
        
        # [修正 2] 支援 Masking 的 Epsilon-Greedy
        if training and random.random() < eps:
            valid_indices = [i for i, m in enumerate(mask) if m == 1.0]
            return random.choice(valid_indices)
        
        with torch.no_grad():
            q_values = self.policy_net(torch.FloatTensor(state))
            # Masking: 將無效動作 Q 值設為極小
            inf_mask = (torch.FloatTensor(mask) - 1) * 1e9
            masked_q = q_values + inf_mask
            return masked_q.argmax().item()

    def update(self):
        if len(self.memory) < Config.BATCH_SIZE: return
        batch = random.sample(self.memory, Config.BATCH_SIZE)
        state, action, reward, next_state, done, mask, next_mask = zip(*batch)
        
        state = torch.FloatTensor(np.array(state))
        action = torch.LongTensor(action).unsqueeze(1)
        reward = torch.FloatTensor(reward).unsqueeze(1)
        next_state = torch.FloatTensor(np.array(next_state))
        done = torch.FloatTensor(done).unsqueeze(1)
        # next_mask 用於 Double DQN 或未來擴充，目前標準 DQN 暫不需要
        
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
    
    # [修正 2] 接收 mask
    (state, mask), _ = env.reset(seed=Config.SEED)
    total_reward = 0
    
    pbar = tqdm(range(Config.TRAIN_STEPS))
    for step in pbar:
        # [修正 2] 傳入 mask
        action = agent.select_action(state, mask)
        (next_state, next_mask), r, done, _, _ = env.step(action)
        
        # 儲存 mask 資訊 (雖然標準 DQN update 沒用到 next_mask，但存著好)
        agent.memory.append((state, action, r, next_state, done, mask, next_mask))
        agent.update()
        
        state = next_state
        mask = next_mask
        total_reward += r
        
        if step % Config.TARGET_UPDATE == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
            
        if step % Config.EVAL_INTERVAL == 0:
            if env.finished_jobs:
                avg_t = np.mean([j.get_tardiness(j.completion_time) for j in env.finished_jobs[-50:]])
                avg_tardiness.append(avg_t)
                pbar.set_description(f"Step {step} | Avg Tardiness: {avg_t:.2f} | R: {total_reward:.1f}")
            rewards.append(total_reward)
            total_reward = 0 

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
    time_window_start = env.now - 200
    for m in env.machines:
        for h in m.history:
            start, end, type_ = h[2], h[3], h[4]
            if end < time_window_start: continue 
            
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
            elif 'MINIMAL' in type_:
                ax.add_patch(mpatches.Rectangle((start, m.id*10), dur, 9, facecolor='orange', hatch='..', edgecolor='black'))
                ax.text(start+dur/2, m.id*10+4.5, "MR", ha='center', va='center', color='black', fontsize=8)
                
    ax.set_yticks([i*10+5 for i in range(Config.NUM_MACHINES)])
    ax.set_yticklabels([f'M{i}' for i in range(Config.NUM_MACHINES)])
    ax.set_xlabel("Time")
    ax.set_xlim(time_window_start, env.now)
    ax.set_title(f"Advanced Schedule Snapshot (Time {env.now:.0f})")
    plt.savefig("advanced_gantt.png")
    print("Saved advanced_gantt.png")

if __name__ == "__main__":
    run_advanced_experiment()