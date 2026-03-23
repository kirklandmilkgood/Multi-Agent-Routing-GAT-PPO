import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import random
from collections import deque
import time

# ==========================================
# 1. DGN 核心神經網路 (包含 GCN 通訊層)
# ==========================================
class DGNNetwork(nn.Module):
    def __init__(self, obs_dim, hidden_dim, action_dim, num_agents):
        super(DGNNetwork, self).__init__()
        self.num_agents = num_agents
        
        # 1. 觀測值編碼器 (Encoder)：將原始未正規化的特徵轉換為隱藏向量
        self.encoder = nn.Linear(obs_dim, hidden_dim)
        
        # 2. ✨ 圖卷積層 (GCN Layer)：負責 Agent 之間的訊息傳遞
        self.gcn_weight = nn.Linear(hidden_dim, hidden_dim)
        
        # 3. 決策輸出層 (Q-Network)：綜合自己的情報與鄰居情報後，輸出 Q 值
        self.q_out = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), # 拼接自身特徵與通訊特徵
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, obs_batch):
        """
        obs_batch 形狀: (Batch_Size, Num_Agents, Obs_Dim)
        """
        # Step 1: 獨立編碼每個 Agent 的觀測值
        # h 形狀: (Batch_Size, Num_Agents, Hidden_Dim)
        h = F.relu(self.encoder(obs_batch)) 
        
        # Step 2: Agent 之間的 GCN 訊息傳遞 (Message Passing)
        # 這裡假設所有 Agent 都能互相通訊 (Fully Connected Communication Graph)
        # 我們將所有 Agent 的特徵取平均 (Mean Aggregation)，等同於乘以正規化後的鄰接矩陣 D^(-1/2) A D^(-1/2)
        mean_neighbor_h = h.mean(dim=1, keepdim=True) # (Batch_Size, 1, Hidden_Dim)
        
        # 將鄰居情報廣播給所有 Agent，並通過 GCN 權重矩陣
        # h_comm 形狀: (Batch_Size, Num_Agents, Hidden_Dim)
        h_comm = F.relu(self.gcn_weight(mean_neighbor_h.expand(-1, self.num_agents, -1)))
        
        # Step 3: 將「自身的情報 h」與「通訊得來的情報 h_comm」拼接起來
        h_final = torch.cat([h, h_comm], dim=-1) # (Batch_Size, Num_Agents, Hidden_Dim * 2)
        
        # Step 4: 輸出 Q 值
        q_values = self.q_out(h_final) # (Batch_Size, Num_Agents, Action_Dim)
        return q_values

# ==========================================
# 2. 經驗回放池 (與 VDN 相同)
# ==========================================
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, obs, action, reward, next_obs, done, avail_actions, next_avail_actions):
        self.buffer.append((obs, action, reward, next_obs, done, avail_actions, next_avail_actions))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        obs, action, reward, next_obs, done, avail_actions, next_avail_actions = zip(*batch)
        return (np.array(obs), np.array(action), np.array(reward), 
                np.array(next_obs), np.array(done), 
                np.array(avail_actions), np.array(next_avail_actions))

    def __len__(self):
        return len(self.buffer)

# ==========================================
# 3. DGN 代理人演算法
# ==========================================
class DGNAgent:
    def __init__(self, env, hidden_dim=64, lr=1e-3, gamma=0.99, buffer_capacity=10000, batch_size=32):
        self.env = env
        self.gamma = gamma
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.target_update_steps = 200
        self.global_step = 0

        self.state_dim = len(env._get_state())
        self.obs_dim = 2 + self.state_dim + env.num_agents 
        self.action_dim = env.num_nodes

        # 初始化 DGN 網路 (Eval 與 Target)
        self.eval_net = DGNNetwork(self.obs_dim, hidden_dim, self.action_dim, env.num_agents).to(self.device)
        self.target_net = DGNNetwork(self.obs_dim, hidden_dim, self.action_dim, env.num_agents).to(self.device)
        self.target_net.load_state_dict(self.eval_net.state_dict())

        self.optimizer = optim.Adam(self.eval_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_capacity)

    def _format_state(self, state):
        """✨ 依照您的要求，完全保持未正規化的原始數據"""
        obs = []
        for i in range(self.env.num_agents):
            # 原始位置與原始預算 (不做任何除法)
            raw_pos = self.env.agent_positions[i]
            raw_budget = self.env.remaining_agent_budgets[i]
            
            agent_id = np.zeros(self.env.num_agents)
            agent_id[i] = 1.0
            
            # 拼接未正規化的觀測值
            local_obs = np.concatenate([[raw_pos, raw_budget], state, agent_id])
            obs.append(local_obs)
            
        return np.array(obs, dtype=np.float32)

    def _get_available_actions(self):
        """產生合法動作遮罩 (與所有 Baseline 完全對齊)"""
        available_actions = np.zeros((self.env.num_agents, self.env.num_nodes), dtype=np.float32)
        for i in range(self.env.num_agents):
            curr = self.env.agent_positions[i]
            for j in range(self.env.num_nodes):
                cost = self.env.graph[curr].get(j, {}).get('weight', None) if j != curr else 0
                is_valid_move = (cost is not None) and \
                                (self.env.remaining_agent_budgets[i] >= cost) and \
                                (self.env.remaining_total_budget >= cost)
                has_reward = self.env.node_rewards[j][0] > 0 or j == curr
                
                if is_valid_move and has_reward:
                    available_actions[i][j] = 1.0
                    
            if np.sum(available_actions[i]) == 0:
                available_actions[i][curr] = 1.0
        return available_actions

    def get_action(self, obs, avail_actions, epsilon=0.0):
        actions = []
        # 因為 DGN 需要一次吃進所有 Agent 的觀測值來做 GCN 通訊，所以要增加一個 Batch 維度 (unsqueeze)
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # q_values 形狀: (1, Num_Agents, Action_Dim)
            q_values = self.eval_net(obs_tensor).squeeze(0).cpu().numpy()

        for i in range(self.env.num_agents):
            valid_action_indices = np.where(avail_actions[i] == 1.0)[0]
            if np.random.rand() < epsilon:
                action = np.random.choice(valid_action_indices)
            else:
                masked_q_values = q_values[i].copy()
                masked_q_values[avail_actions[i] == 0.0] = -1e9
                action = np.argmax(masked_q_values)
            actions.append(action)
        return np.array(actions)

    def _update(self):
        if len(self.buffer) < self.batch_size:
            return 0.0

        obs_b, action_b, reward_b, next_obs_b, done_b, avail_actions_b, next_avail_actions_b = self.buffer.sample(self.batch_size)

        obs_b = torch.tensor(obs_b, dtype=torch.float32).to(self.device)
        action_b = torch.tensor(action_b, dtype=torch.long).unsqueeze(-1).to(self.device)
        reward_b = torch.tensor(reward_b, dtype=torch.float32).unsqueeze(-1).to(self.device)
        next_obs_b = torch.tensor(next_obs_b, dtype=torch.float32).to(self.device)
        done_b = torch.tensor(done_b, dtype=torch.float32).unsqueeze(-1).to(self.device)
        next_avail_actions_b = torch.tensor(next_avail_actions_b, dtype=torch.float32).to(self.device)

        # 1. 評估目前的 Q 值 (前向傳播會觸發 GCN 通訊)
        q_evals = self.eval_net(obs_b) # (Batch, Num_Agents, Action_Dim)
        q_evals = q_evals.gather(2, action_b).squeeze(-1) # (Batch, Num_Agents)

        # 2. 計算目標 Q 值 (Double DQN 邏輯)
        with torch.no_grad():
            q_next_eval = self.eval_net(next_obs_b)
            q_next_eval[next_avail_actions_b == 0.0] = -1e9
            argmax_action = q_next_eval.argmax(dim=2, keepdim=True)
            
            q_next_target = self.target_net(next_obs_b)
            max_q_next = q_next_target.gather(2, argmax_action).squeeze(-1)
            
            # 我們將每個 Agent 自己收集到的 reward 分配給自己
            targets = reward_b + self.gamma * max_q_next * (1 - done_b)

        # 3. 計算 MSE Loss 並加總所有 Agent 的誤差
        loss = F.mse_loss(q_evals, targets)
        
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.eval_net.parameters(), max_norm=5)
        self.optimizer.step()

        return loss.item()

    def train(self, num_episodes=500): # ✨ 預設調高為 500 以利觀察收斂
        logs = []
        start_time = time.time()
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

                if self.global_step % self.target_update_steps == 0:
                    self.target_net.load_state_dict(self.eval_net.state_dict())

            self.epsilon = max(self.epsilon_min, self.epsilon - epsilon_decay_step)
            avg_loss = total_loss / step_count if step_count > 0 else 0
            logs.append([episode, total_reward, avg_loss])
            
            if episode % 10 == 0:
                print(f"[DGN-Baseline | Episode {episode}] Total Reward: {total_reward:.2f} | Avg Loss: {avg_loss:.4f} | Epsilon: {self.epsilon:.2f}")

        print(f"訓練時間：{time.time() - start_time:.2f} 秒")
        return pd.DataFrame(logs, columns=["Episode", "Total Reward", "Loss"])

    def evaluate(self, num_runs=10):
        all_rewards, all_costs = [], []
        start_time = time.time()
        print(f"\n=== DGN 評估結果（執行 {num_runs} 次） ===")
        for run in range(num_runs):
            state, _ = self.env.reset()
            obs = self._format_state(state)
            avail_actions = self._get_available_actions()
            
            done = False
            total_cost = 0
            agent_rewards = [0] * self.env.num_agents
            collected_nodes = set([0]) 
            
            while not done:
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
        print(f"DGN 總評估時間：{time.time() - start_time:.2f} 秒")
        return pd.DataFrame({"Total Reward": all_rewards, "Total Cost": all_costs})