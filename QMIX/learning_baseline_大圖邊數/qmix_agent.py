import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import random
from collections import deque
import time


# Agent Q-Network
class AgentQNetwork(nn.Module):
    def __init__(self, obs_dim, hidden_dim, action_dim):
        super(AgentQNetwork, self).__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q_out = nn.Linear(hidden_dim, action_dim)

    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        return self.q_out(x)

# QMIX 核心：超網路與混和網路 (Mixing Network)
class QMIXMixer(nn.Module):
    def __init__(self, num_agents, state_dim, embed_dim=32, hypernet_embed=64):
        super(QMIXMixer, self).__init__()
        self.n_agents = num_agents
        self.state_dim = state_dim
        self.embed_dim = embed_dim

        # 超網路 1 (Hypernetwork W1)：負責生成第一層的權重 W1
        # 輸出維度需為 (n_agents * embed_dim) 以便組裝成矩陣
        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, hypernet_embed),
            nn.ReLU(),
            nn.Linear(hypernet_embed, self.n_agents * self.embed_dim)
        )
        # 超網路 2：生成第一層的偏置 b1
        self.hyper_b1 = nn.Linear(state_dim, self.embed_dim)

        # 超網路 3 (Hypernetwork W2)：負責生成第二層的權重 W2
        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim, hypernet_embed),
            nn.ReLU(),
            nn.Linear(hypernet_embed, self.embed_dim * 1)
        )
        # 超網路 4：生成第二層的偏置 b2 (這層可以用深一點的網路)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, 1)
        )

    def forward(self, agent_qs, states):
        """
        agent_qs: shape (Batch_Size, Num_Agents)
        states: shape (Batch_Size, State_Dim)
        """
        b_size = agent_qs.size(0)
        # 變形為 (Batch, 1, Num_Agents) 方便做矩陣乘法
        agent_qs = agent_qs.view(-1, 1, self.n_agents)

        # 單調性約束 - 所有的權重 W 必須經過 torch.abs() 保證大於等於零
        w1 = torch.abs(self.hyper_w1(states))
        w1 = w1.view(-1, self.n_agents, self.embed_dim) # 變形為 (Batch, Num_Agents, Embed)
        b1 = self.hyper_b1(states).view(-1, 1, self.embed_dim)

        w2 = torch.abs(self.hyper_w2(states))
        w2 = w2.view(-1, self.embed_dim, 1) # 變形為 (Batch, Embed, 1)
        b2 = self.hyper_b2(states).view(-1, 1, 1)

        # 第一層混和：隱藏層 = ELU(Q * W1 + b1)
        hidden = F.elu(torch.bmm(agent_qs, w1) + b1)
        # 第二層混和：總價值 Q_tot = hidden * W2 + b2
        q_tot = torch.bmm(hidden, w2) + b2

        return q_tot.view(b_size, 1)

# Replay Buffer - 新增 state 儲存
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, obs, action, reward, next_state, next_obs, done, avail_actions, next_avail_actions):
        # QMIX 需要全局 state 來輸入給 Hypernetwork，必須多存 state 與 next_state
        self.buffer.append((state, obs, action, reward, next_state, next_obs, done, avail_actions, next_avail_actions))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, obs, action, reward, next_state, next_obs, done, avail_actions, next_avail_actions = zip(*batch)
        return (np.array(state), np.array(obs), np.array(action), np.array(reward), 
                np.array(next_state), np.array(next_obs), np.array(done), 
                np.array(avail_actions), np.array(next_avail_actions))

    def __len__(self):
        return len(self.buffer)


# QMIX 演算法 (Agent adapter)
class QMIXAgent:
    def __init__(self, env, hidden_dim=64, lr=1e-3, gamma=0.99, buffer_capacity=10000, batch_size=32):
        self.env = env
        self.gamma = gamma
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Epsilon-Greedy 探索率
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.target_update_steps = 200
        self.global_step = 0

        # 維度計算
        self.state_dim = len(env._get_state())
        # obs = [自身位置, 自身預算] + [全局狀態] + [Agent_ID_One_Hot]
        self.obs_dim = 2 + self.state_dim + env.num_agents 
        self.action_dim = env.num_nodes

        # 初始化 Q 網路
        self.eval_q_net = AgentQNetwork(self.obs_dim, hidden_dim, self.action_dim).to(self.device)
        self.target_q_net = AgentQNetwork(self.obs_dim, hidden_dim, self.action_dim).to(self.device)
        self.target_q_net.load_state_dict(self.eval_q_net.state_dict())

        # 初始化 QMIX 混和網路
        self.mixer = QMIXMixer(env.num_agents, self.state_dim).to(self.device)
        self.target_mixer = QMIXMixer(env.num_agents, self.state_dim).to(self.device)
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
        return np.array(obs, dtype=np.float32), agent_state

    def _get_available_actions(self):
        """產生合法動作遮罩 (與先前完全一致以求公平)"""
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
        obs_tensor = torch.tensor(obs, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            q_values = self.eval_q_net(obs_tensor).cpu().numpy()

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
        """執行 QMIX 的 Double DQN 更新"""
        if len(self.buffer) < self.batch_size:
            return 0.0

        # sample，多出了 state_b 與 next_state_b
        state_b, obs_b, action_b, reward_b, next_state_b, next_obs_b, done_b, avail_actions_b, next_avail_actions_b = self.buffer.sample(self.batch_size)

        state_b = torch.tensor(state_b, dtype=torch.float32).to(self.device)
        next_state_b = torch.tensor(next_state_b, dtype=torch.float32).to(self.device)
        obs_b = torch.tensor(obs_b, dtype=torch.float32).to(self.device)
        action_b = torch.tensor(action_b, dtype=torch.long).unsqueeze(-1).to(self.device)
        reward_b = torch.tensor(reward_b, dtype=torch.float32).unsqueeze(-1).to(self.device) 
        next_obs_b = torch.tensor(next_obs_b, dtype=torch.float32).to(self.device)
        done_b = torch.tensor(done_b, dtype=torch.float32).unsqueeze(-1).to(self.device)     
        next_avail_actions_b = torch.tensor(next_avail_actions_b, dtype=torch.float32).to(self.device)

        # 計算目前的 Q_tot (使用 Mixer，傳入 Global State)
        q_evals = []
        for i in range(self.env.num_agents):
            q_values = self.eval_q_net(obs_b[:, i, :])
            chosen_q = q_values.gather(1, action_b[:, i, :])
            q_evals.append(chosen_q)
        q_evals = torch.cat(q_evals, dim=1) # (Batch, Num_Agents)
        q_tot_eval = self.mixer(q_evals, state_b) # 餵入全局 state

        # 計算 target Q_tot (Double DQN 邏輯)
        q_targets = []
        with torch.no_grad():
            for i in range(self.env.num_agents):
                # Eval 網路挑選動作
                q_eval_next = self.eval_q_net(next_obs_b[:, i, :])
                q_eval_next[next_avail_actions_b[:, i, :] == 0.0] = -1e9
                argmax_action = q_eval_next.argmax(dim=1, keepdim=True)
                
                # Target 網路評估價值
                q_next_values = self.target_q_net(next_obs_b[:, i, :])
                max_q_next = q_next_values.gather(1, argmax_action)
                q_targets.append(max_q_next)
                
            q_targets = torch.cat(q_targets, dim=1)
            q_tot_target_next = self.target_mixer(q_targets, next_state_b) # 餵入下一步全局 state

            targets = reward_b + self.gamma * q_tot_target_next * (1 - done_b)

        # 反向傳播
        loss = F.mse_loss(q_tot_eval, targets)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.eval_q_net.parameters(), max_norm=5)
        self.optimizer.step()

        return loss.item()

    def train(self, num_episodes=50):
        logs = []
        start_time = time.time()
        epsilon_decay_step = (1.0 - self.epsilon_min) / (num_episodes * 0.8)

        for episode in range(num_episodes):
            state, _ = self.env.reset()
            obs, state= self._format_state(state)
            avail_actions = self._get_available_actions()
            
            done = False
            total_reward = 0
            total_loss = 0
            step_count = 0

            while not done:
                actions = self.get_action(obs, avail_actions, self.epsilon)
                next_state, reward, done, _, _ = self.env.step(actions)
                next_obs, next_state = self._format_state(next_state)
                next_avail_actions = self._get_available_actions()
                
                # 必須把原始 state 放入 Buffer 中
                self.buffer.push(state, obs, actions, reward, next_state, next_obs, done, avail_actions, next_avail_actions)
                
                loss = self._update()
                total_loss += loss

                state = next_state
                obs = next_obs
                avail_actions = next_avail_actions
                total_reward += reward
                step_count += 1
                self.global_step += 1

                if self.global_step % self.target_update_steps == 0:
                    self.target_q_net.load_state_dict(self.eval_q_net.state_dict())
                    self.target_mixer.load_state_dict(self.mixer.state_dict())

            self.epsilon = max(self.epsilon_min, self.epsilon - epsilon_decay_step)

            avg_loss = total_loss / step_count if step_count > 0 else 0
            logs.append([episode, total_reward, avg_loss])
            
            if episode % 10 == 0:
                print(f"[QMIX-Baseline | Episode {episode}] Total Reward: {total_reward:.2f} | Avg Loss: {avg_loss:.4f} | Epsilon: {self.epsilon:.2f}")

        print(f"訓練時間：{time.time() - start_time:.2f} 秒")
        return pd.DataFrame(logs, columns=["Episode", "Total Reward", "Loss"])

    def evaluate(self, num_runs=10):
        """評估階段：100% 確定性策略 (Epsilon=0.0)"""
        all_rewards, all_costs = [], []
        start_time = time.time()
        
        print(f"\n=== QMIX 評估結果（執行 {num_runs} 次） ===")
        for run in range(num_runs):
            state, _ = self.env.reset()
            obs, _ = self._format_state(state)
            avail_actions = self._get_available_actions()
            
            done = False
            total_cost = 0
            agent_rewards = [0] * self.env.num_agents
            collected_nodes = set([0]) 
            
            while not done:
                actions = self.get_action(obs, avail_actions, epsilon=0.0)
                next_state, _, done, info, _ = self.env.step(actions)
                obs, _ = self._format_state(next_state)
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
        print(f"QMIX 總評估時間：{time.time() - start_time:.2f} 秒")
        return pd.DataFrame({"Total Reward": all_rewards, "Total Cost": all_costs})