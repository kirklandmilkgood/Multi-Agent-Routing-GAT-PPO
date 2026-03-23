import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import random
from collections import deque
import time


# 單一代理人的 Q 網路 (Agent Q-Network)
class AgentQNetwork(nn.Module):
    def __init__(self, obs_dim, hidden_dim, action_dim):
        super(AgentQNetwork, self).__init__()
        # 採用標準多層感知機 (MLP)
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q_out = nn.Linear(hidden_dim, action_dim)

    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        return self.q_out(x)

# VDN 價值混和器 (VDN Mixer)
class VDNMixer(nn.Module):
    def __init__(self):
        super(VDNMixer, self).__init__()

    def forward(self, agent_qs):
        # VDN 的核心假設 - 團隊總價值 (Q_tot) 等於各個 Agent 價值 (Q_i) 的線性總和
        # agent_qs 的維度: (Batch_Size, Num_Agents)
        # 沿著 Agent 維度 (dim=1) 進行加總
        return torch.sum(agent_qs, dim=1, keepdim=True)

# 經驗回放池 (Replay Buffer)
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, obs, action, reward, next_obs, done, avail_actions, next_avail_actions):
        # 將環境互動的 trajectory 存入 memory
        self.buffer.append((obs, action, reward, next_obs, done, avail_actions, next_avail_actions))

    def sample(self, batch_size):
        # 隨機打亂並 sample ，打破資料的時間相關性 (Off-Policy)
        batch = random.sample(self.buffer, batch_size)
        obs, action, reward, next_obs, done, avail_actions, next_avail_actions = zip(*batch)
        return (np.array(obs), np.array(action), np.array(reward), 
                np.array(next_obs), np.array(done), 
                np.array(avail_actions), np.array(next_avail_actions))

    def __len__(self):
        return len(self.buffer)


# VDN 演算法 (Agent adapter)
class VDNAgent:
    def __init__(self, env, hidden_dim=64, lr=2e-3, gamma=0.99, buffer_capacity=10000, batch_size=32):
        self.env = env
        self.gamma = gamma
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Epsilon-Greedy 探索率設定
        self.epsilon = 1.0        # 初始 100% 隨機探索
        self.epsilon_min = 0.05   # 訓練後期保留 5% 的隨機性
        self.target_update_steps = 200 # Target 網路硬更新頻率
        self.global_step = 0

        # 狀態維度加入 Agent ID (One-hot encoding)
        state_dim = len(env._get_state())
        # obs = [自身位置, 自身預算] + [全局狀態] + [Agent_ID_One_Hot]
        self.obs_dim = 2 + state_dim + env.num_agents 
        self.action_dim = env.num_nodes

        # 初始化 Eval 與 Target 網路
        self.eval_q_net = AgentQNetwork(self.obs_dim, hidden_dim, self.action_dim).to(self.device)
        self.target_q_net = AgentQNetwork(self.obs_dim, hidden_dim, self.action_dim).to(self.device)
        self.target_q_net.load_state_dict(self.eval_q_net.state_dict())

        self.mixer = VDNMixer().to(self.device)
        self.target_mixer = VDNMixer().to(self.device)
        self.target_mixer.load_state_dict(self.mixer.state_dict())

        self.optimizer = optim.Adam(list(self.eval_q_net.parameters()) + list(self.mixer.parameters()), lr=lr)
        self.buffer = ReplayBuffer(buffer_capacity)

    def _format_state(self, state):
        agent_state = np.array(state, dtype=np.float32)

        obs = []
        for i in range(self.env.num_agents):
            agent_pos = self.env.agent_positions[i]
            agent_budget = self.env.remaining_agent_budgets[i]
            
            # Agent ID One-hot
            agent_id = np.zeros(self.env.num_agents)
            agent_id[i] = 1.0
            
            local_obs = np.concatenate([[agent_pos, agent_budget], agent_state, agent_id])
            obs.append(local_obs)

        return np.array(obs, dtype=np.float32)

    def _get_available_actions(self):
        """合法動作遮罩"""
        available_actions = np.zeros((self.env.num_agents, self.env.num_nodes), dtype=np.float32)
        for i in range(self.env.num_agents):
            curr = self.env.agent_positions[i]
            for j in range(self.env.num_nodes):
                # 判斷連通性與預算
                cost = self.env.graph[curr].get(j, {}).get('weight', None) if j != curr else 0
                is_valid_move = (cost is not None) and \
                                (self.env.remaining_agent_budgets[i] >= cost) and \
                                (self.env.remaining_total_budget >= cost)
                # 判斷是否有獎勵
                has_reward = self.env.node_rewards[j][0] > 0 or j == curr
                
                if is_valid_move and has_reward:
                    available_actions[i][j] = 1.0
                    
            # 若無路可走，待在原地
            if np.sum(available_actions[i]) == 0:
                available_actions[i][curr] = 1.0
        return available_actions

    def get_action(self, obs, avail_actions, epsilon=0.0):
        """根據 Epsilon-Greedy 與 Mask 選擇動作"""
        actions = []
        obs_tensor = torch.tensor(obs, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            q_values = self.eval_q_net(obs_tensor).cpu().numpy()

        for i in range(self.env.num_agents):
            valid_action_indices = np.where(avail_actions[i] == 1.0)[0]
            
            if np.random.rand() < epsilon:
                # 隨機探索，從合法的動作中隨機選一個
                action = np.random.choice(valid_action_indices)
            else:
                # 貪婪利用，將不合法動作 Q 值設為極小值，強制選出合法的最大 Q 值
                masked_q_values = q_values[i].copy()
                masked_q_values[avail_actions[i] == 0.0] = -1e9
                action = np.argmax(masked_q_values)
                
            actions.append(action)
        return np.array(actions)

    def _update(self):
        """從 Replay Buffer 抽樣並執行 Double DQN 更新"""
        if len(self.buffer) < self.batch_size:
            return 0.0

        # 從 buffer 抽樣 batch 資料
        obs_b, action_b, reward_b, next_obs_b, done_b, avail_actions_b, next_avail_actions_b = self.buffer.sample(self.batch_size)

        # 轉為 tensor (shape: batch_size, num_agents, ...)
        obs_b = torch.tensor(obs_b, dtype=torch.float32).to(self.device)
        action_b = torch.tensor(action_b, dtype=torch.long).unsqueeze(-1).to(self.device)
        reward_b = torch.tensor(reward_b, dtype=torch.float32).unsqueeze(-1).to(self.device) 
        next_obs_b = torch.tensor(next_obs_b, dtype=torch.float32).to(self.device)
        done_b = torch.tensor(done_b, dtype=torch.float32).unsqueeze(-1).to(self.device)     
        next_avail_actions_b = torch.tensor(next_avail_actions_b, dtype=torch.float32).to(self.device)

        # 計算目前的 Q_tot (Eval 網路)
        q_evals = []
        for i in range(self.env.num_agents):
            q_values = self.eval_q_net(obs_b[:, i, :])
            chosen_q = q_values.gather(1, action_b[:, i, :]) # 取出實際執行動作的 Q 值
            q_evals.append(chosen_q)
        q_evals = torch.cat(q_evals, dim=1)
        q_tot_eval = self.mixer(q_evals)

        # 使用 Double DQN 邏輯計算 Target Q_tot
        q_targets = []
        with torch.no_grad():
            for i in range(self.env.num_agents):
                # 用 Eval Net 評估下一步的所有動作，並套用 Mask 找出最佳動作 (Argmax)
                q_eval_next = self.eval_q_net(next_obs_b[:, i, :])
                q_eval_next[next_avail_actions_b[:, i, :] == 0.0] = -1e9
                argmax_action = q_eval_next.argmax(dim=1, keepdim=True)
                
                # 用 Target Net 計算該最佳動作的真實 Q 值
                q_next_values = self.target_q_net(next_obs_b[:, i, :])
                max_q_next = q_next_values.gather(1, argmax_action)
                q_targets.append(max_q_next)
                
            q_targets = torch.cat(q_targets, dim=1)
            q_tot_target_next = self.target_mixer(q_targets)

            # 計算 TD Target: y = r + gamma * max(Q_next) * (1 - done)
            targets = reward_b + self.gamma * q_tot_target_next * (1 - done_b)

        # 計算 MSE Loss 並反向傳播
        loss = F.mse_loss(q_tot_eval, targets)
        self.optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪 (Gradient Clipping)
        nn.utils.clip_grad_norm_(self.eval_q_net.parameters(), max_norm=10)
        self.optimizer.step()

        return loss.item()

    def train(self, num_episodes=50):
        logs = []
        start_time = time.time()
        
        # 計算線性衰減的 Epsilon 步長，在 80% 的訓練期內降到最低值
        epsilon_decay_step = (1.0 - self.epsilon_min) / (num_episodes * 0.8)

        for episode in range(num_episodes):
            state, _ = self.env.reset()
            obs = self._format_state(state)
            avail_actions = self._get_available_actions()
            
            done = False
            total_reward = 0
            total_loss = 0
            step_count = 0

            while not done:
                actions = self.get_action(obs, avail_actions, self.epsilon)
                next_state, reward, done, _, _ = self.env.step(actions)
                next_obs = self._format_state(next_state)
                next_avail_actions = self._get_available_actions()
                
                self.buffer.push(obs, actions, reward, next_obs, done, avail_actions, next_avail_actions)
                
                loss = self._update()
                total_loss += loss

                obs = next_obs
                avail_actions = next_avail_actions
                total_reward += reward
                step_count += 1
                self.global_step += 1

                # 定期將 Eval 網路的權重複製給 Target 網路
                if self.global_step % self.target_update_steps == 0:
                    self.target_q_net.load_state_dict(self.eval_q_net.state_dict())
                    self.target_mixer.load_state_dict(self.mixer.state_dict())

            # 每個 Episode 結束後線性衰減探索率
            self.epsilon = max(self.epsilon_min, self.epsilon - epsilon_decay_step)

            avg_loss = total_loss / step_count if step_count > 0 else 0
            logs.append([episode, total_reward, avg_loss])
            
            if episode % 10 == 0:
                print(f"[VDN-Baseline | Episode {episode}] Total Reward: {total_reward:.2f} | Avg Loss: {avg_loss:.4f} | Epsilon: {self.epsilon:.2f}")

        print(f"訓練時間：{time.time() - start_time:.2f} 秒")
        return pd.DataFrame(logs, columns=["Episode", "Total Reward", "Loss"])

    def evaluate(self, num_runs=10):
        """評估階段：關閉隨機性 (epsilon=0.0)，進行純確定性 (Argmax) 推論"""
        all_rewards, all_costs = [], []
        start_time = time.time()
        
        print(f"\n=== VDN 評估結果（執行 {num_runs} 次） ===")
        for run in range(num_runs):
            state, _ = self.env.reset()
            obs = self._format_state(state)
            avail_actions = self._get_available_actions()
            
            done = False
            total_cost = 0
            agent_rewards = [0] * self.env.num_agents
            collected_nodes = set([0]) 
            
            while not done:
                # 評估時 epsilon 設為 0
                actions = self.get_action(obs, avail_actions, epsilon=0.0)
                next_state, _, done, info, _ = self.env.step(actions)
                obs = self._format_state(next_state)
                avail_actions = self._get_available_actions()
                
                for (agent_idx, from_node, to_node) in info.get("executed_moves", []):
                    if self.env.graph.has_edge(from_node, to_node):
                        total_cost += self.env.graph[from_node][to_node]['weight']
                    if to_node not in collected_nodes and self.env.node_rewards[to_node][0] > 0:
                        agent_rewards[agent_idx] += self.env.node_rewards[to_node][0]
                        collected_nodes.add(to_node)
                        
            total_reward = sum(agent_rewards)
            all_rewards.append(total_reward)
            all_costs.append(total_cost)
            print(f"[Run {run}] Total Reward: {total_reward} | Total Cost: {total_cost}")
            
        print(f"=== 平均結果 ===")
        print(f"Average Reward: {np.mean(all_rewards):.2f} | Average Cost: {np.mean(all_costs):.2f}")
        print(f"VDN 總評估時間：{time.time() - start_time:.2f} 秒")
        return pd.DataFrame({"Total Reward": all_rewards, "Total Cost": all_costs})