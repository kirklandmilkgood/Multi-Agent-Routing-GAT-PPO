import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import random
from collections import deque
import time

# Relation Kernel (多頭注意力機制)
class RelationKernel(nn.Module):
    def __init__(self, hidden_dim, num_heads=8):
        super(RelationKernel, self).__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.tau = 0.25 # 縮放因子 (Scaling factor)
        
        # 產生 Query, Key, Value 的投影矩陣
        self.W_Q = nn.Linear(hidden_dim, hidden_dim)
        self.W_K = nn.Linear(hidden_dim, hidden_dim)
        self.W_V = nn.Linear(hidden_dim, hidden_dim)
        
        # 非線性轉換函數 sigma
        self.sigma = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
    def forward(self, h):
        """
        h 形狀: (Batch_Size, Num_Agents, Hidden_Dim)
        """
        batch_size, num_agents, _ = h.size()
        
        # 投影並拆分為多頭 (Batch_Size, Heads, Num_Agents, Head_Dim)
        Q = self.W_Q(h).view(batch_size, num_agents, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_K(h).view(batch_size, num_agents, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_V(h).view(batch_size, num_agents, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 計算注意力能量值 e_ij = tau * Q * K^T
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.tau # (Batch, Heads, Num_Agents, Num_Agents)
        
        # 取得注意力權重 alpha_ij
        attn_weights = F.softmax(scores, dim=-1)
        
        # 將 Value 以注意力權重進行加權總和
        out = torch.matmul(attn_weights, V) # (Batch, Heads, Num_Agents, Head_Dim)
        
        # 將多個頭拼接並送入 sigma 函數
        out = out.transpose(1, 2).contiguous().view(batch_size, num_agents, -1)
        out = self.sigma(out)
        
        return out, attn_weights


# DGN 完整神經網路架構
class DGNNetwork(nn.Module):
    def __init__(self, obs_dim, hidden_dim, action_dim, num_agents):
        super(DGNNetwork, self).__init__()
        self.num_agents = num_agents
        
        # 觀測值編碼器 (Encoder)
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 堆疊兩層卷積層 (Relation Kernels) 以逐步擴大視野
        self.conv1 = RelationKernel(hidden_dim, num_heads=8)
        self.conv2 = RelationKernel(hidden_dim, num_heads=8)
        
        # 決策輸出層 (Q-Network) 
        # DenseNet，將 h0, h1, h2 拼接，輸入維度為 hidden_dim * 3
        self.q_out = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, obs_batch):
        #  獨立編碼
        h0 = self.encoder(obs_batch)
        
        #  多頭注意力訊息傳遞
        h1, attn1 = self.conv1(h0)
        h2, attn2 = self.conv2(h1)
        
        # DenseNet 特徵拼接 (整合自身與多層次鄰居的資訊)
        h_concat = torch.cat([h0, h1, h2], dim=-1)
        
        # 輸出 Q 值
        q_values = self.q_out(h_concat)
        
        # 回傳 Q 值以及最高層的注意力權重
        return q_values, attn2

# 經驗回放池
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

# DGN 代理人演算法
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
        self.lambda_reg = 0.03 # 正則化係數

        self.state_dim = len(env._get_state())
        self.obs_dim = 2 + self.state_dim + env.num_agents 
        self.action_dim = env.num_nodes

        self.eval_net = DGNNetwork(self.obs_dim, hidden_dim, self.action_dim, env.num_agents).to(self.device)
        self.target_net = DGNNetwork(self.obs_dim, hidden_dim, self.action_dim, env.num_agents).to(self.device)
        self.target_net.load_state_dict(self.eval_net.state_dict())

        self.optimizer = optim.Adam(self.eval_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_capacity)

    def _format_state(self, state):
        obs = []
        for i in range(self.env.num_agents):
            raw_pos = self.env.agent_positions[i]
            raw_budget = self.env.remaining_agent_budgets[i]
            agent_id = np.zeros(self.env.num_agents)
            agent_id[i] = 1.0
            local_obs = np.concatenate([[raw_pos, raw_budget], state, agent_id])
            obs.append(local_obs)
        return np.array(obs, dtype=np.float32)

    def _get_available_actions(self):
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
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # 取出 q_values，忽略注意力權重
            q_values, _ = self.eval_net(obs_tensor)
            q_values = q_values.squeeze(0).cpu().numpy()

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

        # 評估目前的 Q 值與取得當前注意力權重
        q_evals, current_attn = self.eval_net(obs_b) 
        q_evals = q_evals.gather(2, action_b).squeeze(-1) 

        # 計算目標 Q 值與取得時序正則化目標權重
        with torch.no_grad():
            q_next_eval, target_attn = self.eval_net(next_obs_b) # 使用 eval_net 來產出 target_attn
            q_next_eval[next_avail_actions_b == 0.0] = -1e9
            argmax_action = q_next_eval.argmax(dim=2, keepdim=True)
            
            q_next_target, _ = self.target_net(next_obs_b)
            max_q_next = q_next_target.gather(2, argmax_action).squeeze(-1)
            
            targets = reward_b + self.gamma * max_q_next * (1 - done_b)

        # TD Error (MSE Loss)
        td_loss = F.mse_loss(q_evals, targets)
        
        # Temporal Relation Regularization (KL Divergence Loss)
        # 計算 KL 散度確保維度穩健: sum(P * log(P/Q))
        epsilon_val = 1e-8
        target_attn_detached = target_attn.detach()
        kl_loss = torch.sum(target_attn_detached * (torch.log(target_attn_detached + epsilon_val) - torch.log(current_attn + epsilon_val)), dim=-1).mean()
        
        # 總損失
        total_loss = td_loss + self.lambda_reg * kl_loss
        
        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.eval_net.parameters(), max_norm=5)
        self.optimizer.step()

        return total_loss.item()

    def train(self, num_episodes=500):
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