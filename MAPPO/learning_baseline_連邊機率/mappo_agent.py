import torch
import numpy as np
import pandas as pd
import time
from gym import spaces
from onpolicy.algorithms.r_mappo.r_mappo import R_MAPPO
from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy

class DummyArgs:
    """
    模擬 MAPPO 初始化所需的 hyperparameter
    """
    def __init__(self, hidden_size=64):
        self.hidden_size = hidden_size
        self.lr = 2e-3             # 對齊 GAT-PPO 的學習率
        self.critic_lr = 2e-3
        self.opti_eps = 1e-5
        self.weight_decay = 0
        self.gain = 0.01
        self.use_orthogonal = True

        self.use_feature_normalization = False  # 關閉正規化，保持最原始的 MLP 狀態
        self.use_ReLU = True                    # 使用 ReLU 激活函數
        self.stacked_frames = 1                 # 預設不堆疊特徵幀
        self.layer_N = 1                        # MLP 的隱藏層層數
        
        # RNN 相關設定
        self.use_naive_recurrent_policy = False
        self.use_recurrent_policy = True
        self.recurrent_N = 1
        
        # PPO 核心參數
        self.algorithm_name = "rmappo"
        self.clip_param = 0.2      # 對齊 GAT-PPO 的 clip_epsilon
        self.ppo_epoch = 4         # 對齊 GAT-PPO 的 epochs
        self.num_mini_batch = 1
        self.data_chunk_length = 10
        self.value_loss_coef = 1.0
        self.entropy_coef = 0.01
        self.max_grad_norm = 10.0
        
        # MAPPO 穩定性優化參數
        self.huber_delta = 10.0
        self.use_max_grad_norm = True
        self.use_clipped_value_loss = True
        self.use_huber_loss = True
        self.use_popart = False
        self.use_valuenorm = False
        self.use_value_active_masks = False
        self.use_policy_active_masks = False

class MAPPOAgent:
    """
    相容於 MATP 環境的 MAPPO adapter
    它負責將環境的 graph 資訊壓平為向量資訊，並生成 action mask 給 MAPPO
    """
    def __init__(self, env, hidden_dim=64):
        self.env = env
        self.args = DummyArgs(hidden_size=hidden_dim)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = 0.99
        
        # 定義狀態與動作空間維度
        # cent_obs (全局狀態): 整個環境的一維陣列長度
        state_dim = len(env._get_state())
        # obs (局部狀態): 讓 Agent 知道自己在哪以及自己剩多少錢(預算)，將這些特徵拼接在全局狀態前面
        obs_dim = 2 + state_dim 
        
        self.obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,))
        self.share_obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(state_dim,))
        self.act_space = spaces.Discrete(env.num_nodes)

        # 初始化 MAPPO strategy 與 optimizer
        self.policy = R_MAPPOPolicy(self.args, self.obs_space, self.share_obs_space, self.act_space, device=self.device)
        self.trainer = R_MAPPO(self.args, self.policy, device=self.device)

    def _format_state(self, state):
        """
        將 MATP 的環境 state 轉換為 MAPPO 需要的 obs 與 cent_obs
        回傳維度: obs (num_agents, obs_dim), cent_obs (num_agents, state_dim)
        """
        # 中心化 Critic 看到的都是全局視野
        cent_obs = np.array([state for _ in range(self.env.num_agents)]) 
        
        obs = []
        for i in range(self.env.num_agents):
            agent_pos = self.env.agent_positions[i]
            agent_budget = self.env.remaining_agent_budgets[i]
            # 去中心化 Actor 看到的視野為自己的位置、自己的預算及全局地圖
            local_obs = np.concatenate([[agent_pos, agent_budget], state])
            obs.append(local_obs)
            
        return np.array(obs, dtype=np.float32), np.array(cent_obs, dtype=np.float32)

    def _get_available_actions(self):
        """
        產生 MAPPO 的 Action Mask
        將 GAT 透過 edge_index 處理的拓撲限制，轉換為適用 MAPPO 的 [num_agents, num_nodes] mask array
        1 = 合法可走, 0 = 非法 (設為 -inf)
        """
        available_actions = np.zeros((self.env.num_agents, self.env.num_nodes), dtype=np.float32)
        
        for i in range(self.env.num_agents):
            curr = self.env.agent_positions[i]
            for j in range(self.env.num_nodes):
                # 檢查條件：圖拓撲是否有連通的邊 (若待在原地 j==curr 則視為 0 成本)
                cost = self.env.graph[curr].get(j, {}).get('weight', None) if j != curr else 0
                
                # 檢查條件：個人與全局預算是否足夠
                is_valid_move = (cost is not None) and \
                                (self.env.remaining_agent_budgets[i] >= cost) and \
                                (self.env.remaining_total_budget >= cost)
                
                # 檢查條件：對齊 GAT 邏輯 (只准走向有獎勵的節點，或是待在原地)
                has_reward = self.env.node_rewards[j][0] > 0 or j == curr
                
                if is_valid_move and has_reward:
                    available_actions[i][j] = 1.0
                    
            # 極端情況處理：如果所有節點都去不了（或沒預算了），強迫選擇待在原地
            if np.sum(available_actions[i]) == 0:
                available_actions[i][curr] = 1.0
                
        return available_actions

    def get_action(self, state, rnn_states_actor, rnn_states_critic, masks, deterministic=False):
        """
        訓練與評估階段時呼叫
        取得 MAPPO 的預測動作與 value
        """
        obs, cent_obs = self._format_state(state)
        available_actions = self._get_available_actions()
        
        # 由於關閉 RNN，給予全零的 Dummy RNN States
        #rnn_states_actor = np.zeros((self.env.num_agents, 1, self.args.hidden_size), dtype=np.float32)
        #rnn_states_critic = np.zeros((self.env.num_agents, 1, self.args.hidden_size), dtype=np.float32)
        # masks = np.ones((self.env.num_agents, 1), dtype=np.float32)

        with torch.no_grad():
            values, actions, action_log_probs, next_rnn_states_actor, next_rnn_states_critic = self.policy.get_actions(
                cent_obs, obs, rnn_states_actor, rnn_states_critic, masks, available_actions, deterministic
            )
        
        # 回傳形狀 - actions: (num_agents, 1), log_probs: (num_agents, 1), values: (num_agents, 1)
        return (actions.cpu().numpy(), action_log_probs.cpu().numpy(), values.cpu().numpy(), next_rnn_states_actor.cpu().numpy(), next_rnn_states_critic.cpu().numpy())

    def train(self, num_episodes=50):
        logs = []
        start_time = time.time()
        
        for episode in range(num_episodes):
            state, _ = self.env.reset() # 丟棄 graph_data，因為 MAPPO 不用 GNN
            done = False
            
            # 建立 Trajectory Buffer
            b_obs, b_cent_obs, b_actions = [], [], []
            b_log_probs, b_values, b_rewards = [], [], []
            b_masks, b_avail_actions = [], []

            b_rnn_actor, b_rnn_critic = [], [] 

            # 每一回合開始時，初始化記憶體與 Mask
            rnn_states_actor = np.zeros((self.env.num_agents, 1, self.args.hidden_size), dtype=np.float32)
            rnn_states_critic = np.zeros((self.env.num_agents, 1, self.args.hidden_size), dtype=np.float32)
            masks = np.ones((self.env.num_agents, 1), dtype=np.float32)

            # 收集軌跡 (Rollout)
            while not done:
                obs, cent_obs = self._format_state(state)
                avail_actions = self._get_available_actions()

                # 記錄當步的記憶與 Mask
                b_rnn_actor.append(rnn_states_actor)
                b_rnn_critic.append(rnn_states_critic)
                b_masks.append(masks)

                actions, log_probs, values, next_rnn_actor, next_rnn_critic = self.get_action(state, rnn_states_actor, rnn_states_critic, masks, deterministic=False)
                
                # 把 MAPPO 輸出的 shape (num_agents, 1) 壓平餵給 env
                next_state, reward, done, _, _ = self.env.step(actions.flatten())
                
                b_obs.append(obs)
                b_cent_obs.append(cent_obs)
                b_actions.append(actions)
                b_log_probs.append(log_probs)
                b_values.append(values)
                b_avail_actions.append(avail_actions)
                
                # 所有 Agent 獲得一樣的總和獎勵
                b_rewards.append(np.array([[reward]] * self.env.num_agents, dtype=np.float32))
                
                state = next_state
                rnn_states_actor = next_rnn_actor
                rnn_states_critic = next_rnn_critic
                masks = np.array([[0.0 if done else 1.0]] * self.env.num_agents, dtype=np.float32)

            # 計算 Returns 與 Advantages
            returns = []
            G = np.zeros((self.env.num_agents, 1), dtype=np.float32)
            for step in reversed(range(len(b_rewards))):
                G = b_rewards[step] + self.gamma * G * b_masks[step]
                returns.insert(0, G)
                
            returns_np = np.array(returns)    # shape: (T, num_agents, 1)
            values_np = np.array(b_values)    # shape: (T, num_agents, 1)
            advs_np = returns_np - values_np  # 計算 Advantage

            advs_np = (advs_np - advs_np.mean()) / (advs_np.std() + 1e-5)

            # 維度對齊與 PPO 更新
            # MAPPO 的 ppo_update 期望所有資料的 Batch 維度合併 (T * num_agents, ...)
            B = len(b_obs) * self.env.num_agents
            
            sample = (
                torch.tensor(np.array(b_cent_obs).reshape(B, -1), dtype=torch.float32),
                torch.tensor(np.array(b_obs).reshape(B, -1), dtype=torch.float32),
                torch.tensor(np.array(b_rnn_actor).reshape(B, 1, self.args.hidden_size), dtype=torch.float32),  # 傳入 Actor 記憶
                torch.tensor(np.array(b_rnn_critic).reshape(B, 1, self.args.hidden_size), dtype=torch.float32), # 傳入 Critic 記憶
                torch.tensor(np.array(b_actions).reshape(B, 1)),
                torch.tensor(values_np.reshape(B, 1)),
                torch.tensor(returns_np.reshape(B, 1)),
                torch.tensor(np.array(b_masks).reshape(B, 1), dtype=torch.float32),
                torch.ones((B, 1), dtype=torch.float32), # active_masks
                torch.tensor(np.array(b_log_probs).reshape(B, 1)),
                torch.tensor(advs_np.reshape(B, 1)),     # adv_targ
                torch.tensor(np.array(b_avail_actions).reshape(B, self.env.num_nodes), dtype=torch.float32)
            )
            
            # 執行 PPO 優化 iteration
            for _ in range(self.args.ppo_epoch):
                value_loss, _, policy_loss, _, _, _ = self.trainer.ppo_update(sample)
                
            total_reward = sum([r[0][0] for r in b_rewards])
            logs.append([episode, total_reward, policy_loss.item() + value_loss.item()])
            
            if episode % 10 == 0:
                print(f"[MAPPO-Baseline | Episode {episode}] Total Reward: {total_reward:.2f} | Total Loss: {policy_loss.item() + value_loss.item():.4f}")

        print(f"訓練時間：{time.time() - start_time:.2f} 秒")
        return pd.DataFrame(logs, columns=["Episode", "Total Reward", "Loss"])

    def evaluate(self, num_runs=10):
        """評估階段"""
        all_rewards, all_costs = [], []
        start_time = time.time()
        
        print(f"\n=== MAPPO 評估結果（執行 {num_runs} 次） ===")
        for run in range(num_runs):
            state, _ = self.env.reset()
            done = False
            total_cost = 0
            agent_rewards = [0] * self.env.num_agents
            collected_nodes = set([0]) 

            # 初始化記憶
            rnn_states_actor = np.zeros((self.env.num_agents, 1, self.args.hidden_size), dtype=np.float32)
            rnn_states_critic = np.zeros((self.env.num_agents, 1, self.args.hidden_size), dtype=np.float32)
            masks = np.ones((self.env.num_agents, 1), dtype=np.float32)
            
            while not done:
                actions, _, _, next_rnn_actor, next_rnn_critic = self.get_action(state, rnn_states_actor, rnn_states_critic, masks, deterministic=False)
                state, _, done, info, _ = self.env.step(actions.flatten())

                # 更新記憶與 Mask
                rnn_states_actor = next_rnn_actor
                rnn_states_critic = next_rnn_critic
                masks = np.array([[0.0 if done else 1.0]] * self.env.num_agents, dtype=np.float32)
                
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
        print(f"MAPPO 總評估時間：{time.time() - start_time:.2f} 秒")
        return pd.DataFrame({"Total Reward": all_rewards, "Total Cost": all_costs})