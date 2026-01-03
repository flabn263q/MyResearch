import matplotlib
matplotlib.use('Agg')

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
import torch.nn as nn
import torch.optim as optim
import random
import math
import heapq
from collections import deque
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
    STATE_DEGRADE_PROB = 0.1 # [調整] 降低機率，配合 age_accum 重置邏輯
    PROC_TIME_PENALTY = 0.1  
    
    # --- Dual Failure Modes ---
    BREAKDOWN_RATE = 0.005 
    
    # --- Maintenance Specs ---
    TIME_PM = 10           
    TIME_CM = 30           
    TIME_MINIMAL = 5       
    
    MAX_CREWS = 2          
    
    # --- Action Definition ---
    ACTION_PM = 4          
    
    # --- Reward Weights & Scaling ---
    # [修正 4] 移除 W_TARDINESS (Completion)，只保留 Step Penalty
    W_MAINT_COST = 0.5
    W_STEP_PENALTY = 1.0  # 強化過程懲罰
    REWARD_SCALE = 10.0
    
    # --- Normalization Factors ---
    NORM_Q_LEN = 10.0
    NORM_TARDINESS = 50.0
    NORM_PROC_TIME = 20.0
    NORM_AGE_ACCUM = 50.0 
    NORM_REMAIN_TIME = 20.0 # [新增]
    NORM_FAIL_COUNT = 5.0   # [新增]
    
    # --- RL Hyperparameters ---
    LR = 1e-4
    GAMMA = 0.99
    BUFFER_SIZE = 20000
    BATCH_SIZE = 128
    EPSILON_START = 1.0
    EPSILON_END = 0.05
    EPSILON_DECAY = 15000
    TARGET_UPDATE = 200
    HIDDEN_DIM = 128
    GRAD_CLIP = 10.0
    
    # --- Memory Management ---
    MAX_HISTORY_LEN = 1000 

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ==========================================
# 2. Efficient Replay Buffer
# ==========================================
class ReplayBuffer:
    def __init__(self, capacity, state_dim):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, 1), dtype=np.int64)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        self.masks = np.zeros((capacity, 5), dtype=np.float32)
        self.next_masks = np.zeros((capacity, 5), dtype=np.float32)

    def push(self, state, action, reward, next_state, done, mask, next_mask):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done
        self.masks[self.ptr] = mask
        self.next_masks[self.ptr] = next_mask
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        return (
            self.states[ind],
            self.actions[ind],
            self.rewards[ind],
            self.next_states[ind],
            self.dones[ind],
            self.masks[ind],
            self.next_masks[ind]
        )

    def __len__(self):
        return self.size

# ==========================================
# 3. Core Classes
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
        self.status = 0      # 0: Idle, 1: Busy, 2: Down/Maint, 3: Waiting
        self.age_accum = 0.0 
        self.history = []    
        self.finish_time = 0.0 # [修正 1] 統一管理釋放時間
        self.failure_count = 0 # [修正 1] 故障計數

    def get_actual_proc_time(self, base_time):
        return base_time * (1.0 + self.state * Config.PROC_TIME_PENALTY)

    def degrade(self):
        # [修正 1] 物理校準：機率隨「當前狀態累積時間」增加
        prob = Config.STATE_DEGRADE_PROB + (self.age_accum * 0.005)
        if random.random() < prob and self.state < Config.K_STATES:
            self.state += 1
            self.age_accum = 0 # [修正 1] 狀態改變，累積歸零
        return self.state >= Config.K_STATES 

    def repair_perfect(self):
        self.state = 0
        self.age_accum = 0
        self.failure_count = 0 # PM 後故障計數歸零 (可選)
        
    def repair_minimal(self):
        self.failure_count += 1 # [修正 1] 增加故障計數
    
    def clean_history(self, current_time):
        cutoff = current_time - 500 
        self.history = [h for h in self.history if h[3] > cutoff]

# ==========================================
# 4. Advanced Event-Driven Environment
# ==========================================
class AdvancedDFJSPEnv(gym.Env):
    def __init__(self):
        super(AdvancedDFJSPEnv, self).__init__()
        
        self.action_space = spaces.Discrete(5)
        
        # [修正 1] State Dim: (7 * M) + 3 + 1
        # Machine: [State, Idle, Busy, Down, Age, Remain, FailCount]
        self.state_dim = (7 * Config.NUM_MACHINES) + 3 + 1
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.state_dim,), dtype=np.float32)
        
    def reset(self, seed=None, options=None):
        self.machines = [Machine(i) for i in range(Config.NUM_MACHINES)]
        self.job_buffer = []     
        self.active_jobs = []    
        self.finished_jobs = []  
        
        self.now = 0.0
        self.event_queue = []    
        self.avail_crews = Config.MAX_CREWS
        self.job_counter = 0
        
        self.repair_queue = deque() 
        
        self._schedule_next_arrival()
        self._schedule_next_breakdown()
        
        self.decision_machine_id = self._resume_simulation()
        
        return self._get_state(), {}

    def _schedule_next_arrival(self):
        inter_arrival = random.expovariate(Config.ARRIVAL_RATE)
        arrival_time = self.now + inter_arrival
        heapq.heappush(self.event_queue, (arrival_time, 'JOB_ARRIVAL', None))

    def _schedule_next_breakdown(self):
        inter_breakdown = random.expovariate(Config.BREAKDOWN_RATE)
        fail_time = self.now + inter_breakdown
        heapq.heappush(self.event_queue, (fail_time, 'RANDOM_BREAKDOWN', None))

    def _resume_simulation(self):
        while True:
            idle_machines = [m.id for m in self.machines if m.status == 0]
            if idle_machines and self.job_buffer:
                return random.choice(idle_machines)

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
                
                if self.now < machine.finish_time:
                    heapq.heappush(self.event_queue, (machine.finish_time, 'JOB_FINISH', data))
                    continue

                for i in range(len(machine.history) - 1, -1, -1):
                    entry = machine.history[i]
                    if entry[0] == j_id and entry[4] == 'JOB':
                        machine.history[i] = (entry[0], entry[1], entry[2], self.now, entry[4])
                        break

                job = next((j for j in self.active_jobs if j.id == j_id), None)
                
                if job:
                    job.current_op_idx += 1
                    if job.current_op_idx >= job.num_ops:
                        job.finished = True
                        job.completion_time = self.now
                        self.active_jobs.remove(job)
                        self.finished_jobs.append(job)
                        # [修正 4] 移除這裡的 Tardiness 扣分，避免雙重計算
                    else:
                        self.job_buffer.append(job)
                
                machine.status = 0 
                is_failed = machine.degrade()
                
                if is_failed:
                    self._request_maintenance(machine, 'CM', Config.TIME_CM)

            elif event_type == 'RANDOM_BREAKDOWN':
                self._schedule_next_breakdown()
                target_m = random.choice(self.machines)
                
                if target_m.status == 1: 
                    self._request_maintenance(target_m, 'MINIMAL', Config.TIME_MINIMAL)

            elif event_type == 'MAINT_FINISH':
                m_id, m_type = data
                machine = self.machines[m_id]
                self.avail_crews += 1
                
                if self.now < machine.finish_time:
                    machine.status = 1 
                else:
                    machine.status = 0 
                
                if m_type == 'PM' or m_type == 'CM':
                    machine.repair_perfect()
                elif m_type == 'MINIMAL':
                    machine.repair_minimal()
                
                if self.repair_queue:
                    next_m_id, next_type, next_dur, next_busy, request_time = self.repair_queue.popleft()
                    wait_time = self.now - request_time
                    self._start_maintenance(self.machines[next_m_id], next_type, next_dur, 
                                          was_busy=next_busy, wait_time=wait_time)

    def _request_maintenance(self, machine, m_type, duration):
        was_busy = (machine.status == 1)
        
        if self.avail_crews > 0:
            self._start_maintenance(machine, m_type, duration, was_busy=was_busy, wait_time=0)
        else:
            machine.status = 3 
            self.repair_queue.append((machine.id, m_type, duration, was_busy, self.now))

    def _start_maintenance(self, machine, m_type, duration, was_busy, wait_time):
        self.avail_crews -= 1
        
        if machine.status == 1 or was_busy:
            total_delay = duration + wait_time
            machine.finish_time += total_delay
        
        machine.status = 2 
        finish_time = self.now + duration
        heapq.heappush(self.event_queue, (finish_time, 'MAINT_FINISH', (machine.id, m_type)))
        
        machine.history.append((-1, -1, self.now, finish_time, m_type))
        
        if wait_time > 0:
            machine.history.append((-1, -1, self.now - wait_time, self.now, 'WAIT'))

    def step(self, action):
        machine = self.machines[self.decision_machine_id]
        
        is_pm = (action == Config.ACTION_PM)
        rule_idx = action if not is_pm else 0
        
        reward = 0.0
        
        # [修正 4] 稠密獎勵 (Dense Reward)
        if self.active_jobs:
            current_avg_tardiness = np.mean([j.get_tardiness(self.now) for j in self.active_jobs])
            reward -= Config.W_STEP_PENALTY * current_avg_tardiness
        
        if is_pm:
            self._start_maintenance(machine, 'PM', Config.TIME_PM, was_busy=(machine.status==1), wait_time=0)
            reward -= Config.W_MAINT_COST
            
            self.decision_machine_id = self._resume_simulation()
            return self._get_state(), reward / Config.REWARD_SCALE, False, False, {}
        
        if not self.job_buffer:
            self.decision_machine_id = self._resume_simulation()
            return self._get_state(), reward / Config.REWARD_SCALE, False, False, {}
            
        selected_job = self._apply_rule(rule_idx, self.job_buffer)
        self.job_buffer.remove(selected_job) 
        
        base_time = selected_job.get_base_proc_time()
        actual_time = machine.get_actual_proc_time(base_time)
        
        machine.status = 1 
        machine.age_accum += actual_time
        finish_time = self.now + actual_time
        machine.finish_time = finish_time 
        
        machine.history.append((selected_job.id, selected_job.current_op_idx, self.now, finish_time, 'JOB'))
        heapq.heappush(self.event_queue, (finish_time, 'JOB_FINISH', (machine.id, selected_job.id)))
        
        self.decision_machine_id = self._resume_simulation()
        
        return self._get_state(), reward / Config.REWARD_SCALE, False, False, {}

    def _apply_rule(self, rule_idx, jobs):
        if rule_idx == 0: return min(jobs, key=lambda j: j.arrival_time)
        elif rule_idx == 1: return min(jobs, key=lambda j: j.get_base_proc_time())
        elif rule_idx == 2: return min(jobs, key=lambda j: j.due_date)
        elif rule_idx == 3: return min(jobs, key=lambda j: sum(j.ops_times[j.current_op_idx:]))
        return jobs[0]

    def _get_state(self):
        m_feats = []
        for m in self.machines:
            # [修正 2] One-Hot Encoding
            is_idle = 1.0 if m.status == 0 else 0.0
            is_busy = 1.0 if m.status == 1 else 0.0
            is_down = 1.0 if m.status >= 2 else 0.0
            
            # [修正 1] 新增 Remaining Time 與 Failure Count
            remain_time = max(0, m.finish_time - self.now)
            
            m_feats.extend([
                m.state / Config.K_STATES,     
                is_idle, is_busy, is_down,
                np.tanh(m.age_accum / Config.NORM_AGE_ACCUM),
                np.tanh(remain_time / Config.NORM_REMAIN_TIME),
                np.tanh(m.failure_count / Config.NORM_FAIL_COUNT)
            ])
            
        if self.job_buffer:
            q_len = np.tanh(len(self.job_buffer) / Config.NORM_Q_LEN) 
            avg_tard = np.tanh(np.mean([j.get_tardiness(self.now) for j in self.job_buffer]) / Config.NORM_TARDINESS)
            avg_proc = np.tanh(np.mean([j.get_base_proc_time() for j in self.job_buffer]) / Config.NORM_PROC_TIME)
        else:
            q_len, avg_tard, avg_proc = 0, 0, 0
            
        crew_ratio = self.avail_crews / Config.MAX_CREWS
        
        state = np.array(m_feats + [q_len, avg_tard, avg_proc, crew_ratio], dtype=np.float32)
        
        mask = np.ones(5, dtype=np.float32)
        if self.avail_crews <= 0:
            mask[Config.ACTION_PM] = 0.0 
        
        if not self.job_buffer:
            mask[0:4] = 0.0
            
        return state, mask

    def clean_history(self):
        for m in self.machines:
            m.clean_history(self.now)
        
        # [修正 3] 清理 finished_jobs
        if len(self.finished_jobs) > 1000:
            self.finished_jobs = self.finished_jobs[-1000:]

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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.policy_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=Config.LR)
        self.memory = ReplayBuffer(Config.BUFFER_SIZE, state_dim)
        self.steps = 0
        self.action_dim = action_dim
        
    def select_action(self, state, mask, training=True):
        eps = Config.EPSILON_END + (Config.EPSILON_START - Config.EPSILON_END) * \
              math.exp(-1. * self.steps / Config.EPSILON_DECAY)
        self.steps += 1
        
        if training and random.random() < eps:
            valid_indices = [i for i, m in enumerate(mask) if m == 1.0]
            if not valid_indices: return 0 
            return random.choice(valid_indices)
        
        with torch.no_grad():
            state_t = torch.FloatTensor(state).to(self.device)
            mask_t = torch.FloatTensor(mask).to(self.device)
            
            q_values = self.policy_net(state_t)
            inf_mask = (mask_t - 1) * 1e9
            masked_q = q_values + inf_mask
            return masked_q.argmax().item()

    def update(self):
        if len(self.memory) < Config.BATCH_SIZE: return
        
        states, actions, rewards, next_states, dones, masks, next_masks = self.memory.sample(Config.BATCH_SIZE)
        
        state = torch.FloatTensor(states).to(self.device)
        action = torch.LongTensor(actions).to(self.device)
        reward = torch.FloatTensor(rewards).to(self.device)
        next_state = torch.FloatTensor(next_states).to(self.device)
        done = torch.FloatTensor(dones).to(self.device)
        next_mask = torch.FloatTensor(next_masks).to(self.device)
        
        q_val = self.policy_net(state).gather(1, action)
        
        with torch.no_grad():
            next_q_values = self.target_net(next_state)
            inf_mask = (next_mask - 1) * 1e9
            masked_next_q = next_q_values + inf_mask
            next_q = masked_next_q.max(1)[0].unsqueeze(1)
            
        expected_q = reward + Config.GAMMA * next_q * (1 - done)
        
        loss = nn.MSELoss()(q_val, expected_q)
        self.optimizer.zero_grad()
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), Config.GRAD_CLIP)
        
        self.optimizer.step()

# ==========================================
# 5. Main Execution
# ==========================================
def run_advanced_experiment():
    set_seed(Config.SEED)
    
    print(f"Starting Advanced Integrated Scheduling (Sim-to-Real) on {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}...")
    env = AdvancedDFJSPEnv()
    agent = DQNAgent(env.observation_space.shape[0], env.action_space.n)
    
    rewards = []
    avg_tardiness = []
    
    (state, mask), _ = env.reset(seed=Config.SEED)
    total_reward = 0
    
    pbar = tqdm(range(Config.TRAIN_STEPS))
    for step in pbar:
        action = agent.select_action(state, mask)
        (next_state, next_mask), r, done, _, _ = env.step(action)
        
        agent.memory.push(state, action, r, next_state, done, mask, next_mask)
        agent.update()
        
        state = next_state
        mask = next_mask
        total_reward += r
        
        if step % Config.TARGET_UPDATE == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
            
        if step % Config.EVAL_INTERVAL == 0:
            env.clean_history()
            
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
            elif 'WAIT' in type_:
                ax.add_patch(mpatches.Rectangle((start, m.id*10), dur, 9, facecolor='gray', alpha=0.5))
                ax.text(start+dur/2, m.id*10+4.5, "W", ha='center', va='center', color='white', fontsize=8)
                
    ax.set_yticks([i*10+5 for i in range(Config.NUM_MACHINES)])
    ax.set_yticklabels([f'M{i}' for i in range(Config.NUM_MACHINES)])
    ax.set_xlabel("Time")
    ax.set_xlim(time_window_start, env.now)
    ax.set_title(f"Advanced Schedule Snapshot (Time {env.now:.0f})")
    plt.savefig("advanced_gantt.png")
    print("Saved advanced_gantt.png")
    
    # Save the trained model
    torch.save(agent.policy_net.state_dict(), "advanced_dqn_model.pth")
    print("Model saved as advanced_dqn_model.pth")

if __name__ == "__main__":
    run_advanced_experiment()
