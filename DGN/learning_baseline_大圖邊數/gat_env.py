import os
import numpy as np
import networkx as nx
import torch
import gym
import math
import pandas as pd
from torch_geometric.data import Data


def generate_er_graph_with_edge_count(num_nodes, num_edges, max_attempts=1000):
    attempts = 0
    while attempts < max_attempts:
        G = nx.gnm_random_graph(num_nodes, num_edges)
        if nx.is_connected(G):
            for u, v in G.edges():
                G[u][v]['weight'] = np.random.randint(1, 10)
            return G
        attempts += 1
    print(f"無法在 {max_attempts} 次內產生連通圖 (n={num_nodes}, m={num_edges}) \n")

    safe_m = int(0.55 * num_nodes * math.log(num_nodes))
    attempts = 0
    for _ in range(max_attempts):
        G = nx.gnm_random_graph(num_nodes, safe_m)
        if nx.is_connected(G):
            for u, v in G.edges():
                G[u][v]['weight'] = np.random.randint(1, 10)
            print(f"產生 n ={num_nodes}, m={safe_m} 的替代連通圖")
            return G
        attempts += 1
    raise ValueError(f"無法在 {max_attempts} 次內產生替代連通圖 (n={num_nodes}, m={safe_m})")


def load_graph_from_excel(file_path):
    if not os.path.exists(file_path):
        print("評估檔案不存在")
        return None
    print("已載入評估圖檔\n")
    df = pd.read_excel(file_path)
    G = nx.Graph()
    for _, row in df.iterrows():
        u, v, w = int(row['From']), int(row['To']), float(row['Weight'])
        G.add_edge(u, v, weight=w)
    return G


class MultiAgentTSPEnv(gym.Env):
    def __init__(self, num_nodes=50, num_edges=None, num_agents=2, total_budget=100, individual_budget=100,
                 eval_file_path=None, dynamic_traffic=False, change_prob=0.10):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.num_agents = num_agents
        self.total_budget = total_budget
        self.individual_budget = individual_budget
        self.eval_file_path = eval_file_path
        self.fixed_graph = None
        self.fixed_rewards = None
        self.input_dim = 4  

        # dynamic 開關與機率
        self.dynamic_traffic = dynamic_traffic
        self.change_prob = change_prob

        if eval_file_path:
            self.fixed_graph = load_graph_from_excel(eval_file_path)
            self.num_nodes = self.fixed_graph.number_of_nodes()

        self.reset()
    
    # dynamic 機制函數
    def _update_traffic_step_by_step(self):
        for u, v, data in self.graph.edges(data=True):
            # 確保已記錄該條路的初始靜態權重 (t=0)
            if 'base_weight' not in data:
                data['base_weight'] = data['weight']
                
            # 以 change_prob 的機率觸發路況變化
            if np.random.rand() < self.change_prob:
                delta = int(np.random.choice([-2, -1, 1, 2]))
                new_weight = data['weight'] + delta
                # bounded limits: 1 to 10
                self.graph[u][v]['weight'] = max(1, min(10, new_weight))

    def _build_graph_data(self):
        node_features = []
        for i in range(self.num_nodes):
            reward = self.node_rewards[i][0] if self.visited[i] > 0 else 0
            visited_flag = 1 if self.visited[i] == 0 else 0
            is_agent_here = 1 if i in self.agent_positions else 0
            has_reward = 1 if (self.visited[i] > 0 and self.node_rewards[i][0] > 0) else 0
            node_features.append([reward, visited_flag, is_agent_here, has_reward])

        x = torch.tensor(node_features, dtype=torch.float32)
        edge_index = []
        for u, v in self.graph.edges():
            edge_index.append([u, v])
            edge_index.append([v, u])
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        return Data(x=x, edge_index=edge_index)

    def reset(self):
        if self.fixed_graph is not None:
            self.graph = self.fixed_graph.copy()
        elif self.num_edges:
            self.graph = generate_er_graph_with_edge_count(self.num_nodes, self.num_edges)
        else:
            self.graph = nx.erdos_renyi_graph(self.num_nodes, 0.6)
            while not nx.is_connected(self.graph):
                self.graph = nx.erdos_renyi_graph(self.num_nodes, 0.6)
            for u, v in self.graph.edges():
                self.graph[u][v]['weight'] = np.random.randint(1, 10)

        # 新的 episode 開始時，恢復基準狀態，並記錄 base_weight
        for u, v, data in self.graph.edges(data=True):
            if 'base_weight' in data:
                # 若圖已被 dynamic weight 污染過，強制恢復初始權重
                self.graph[u][v]['weight'] = data['base_weight']
            else:
                # 初始化記錄
                data['base_weight'] = data['weight']

        self.node_rewards = {i: (np.random.randint(5, 20), 1) for i in self.graph.nodes()}
        self.agent_positions = np.array([0] * self.num_agents)
        self.remaining_total_budget = self.total_budget
        self.remaining_agent_budgets = [self.individual_budget] * self.num_agents
        self.visited = {node: self.node_rewards[node][1] for node in self.graph.nodes()}
        self.visit_count = {node: 0 for node in self.graph.nodes()}
        self.start_positions = [0] * self.num_agents
        for pos in self.start_positions:
            self.node_rewards[pos] = (0, 0)
            self.visited[pos] = 0
        self.current_step = 0
        self.max_steps = 500
        return self._get_state(), self._build_graph_data()

    def _get_state(self):
        reward_vector = np.array([self.node_rewards[i][0] if self.visited[i] > 0 else 0
                                  for i in range(self.num_nodes)])
        visited_vector = np.array([self.visited[i] for i in range(self.num_nodes)])
        return np.concatenate([
            self.agent_positions,
            reward_vector,
            visited_vector,
            [self.remaining_total_budget],
            self.remaining_agent_budgets
        ]).astype(np.float32)

    def step(self, actions):
        self.current_step += 1
        rewards = 0
        move_costs = [0] * self.num_agents
        valid_actions = [False] * self.num_agents

        for i, a in enumerate(actions):
            curr = self.agent_positions[i]
            cost = self.graph[curr].get(a, {}).get('weight', None)
            if cost is None or self.remaining_agent_budgets[i] < cost:
                continue
            move_costs[i] = cost
            valid_actions[i] = True

        total_step_cost = sum(c for c, v in zip(move_costs, valid_actions) if v)
        if total_step_cost > self.remaining_total_budget:
            return self._get_state(), 0, True, {"executed_moves": []}, self._build_graph_data()

        executed_moves = []
        for i, a in enumerate(actions):
            if not valid_actions[i]:
                continue
            prev = self.agent_positions[i]
            cost = move_costs[i]
            self.agent_positions[i] = a
            self.visit_count[a] += 1

            if a in self.start_positions:
                rewards -= 20
            elif self.visited[a] > 0:
                rewards += self.node_rewards[a][0] + 10
                self.visited[a] = 0
            else:
                penalty = min(self.visit_count[a]**2, 10)
                rewards -= 10 * penalty

            self.remaining_agent_budgets[i] -= cost
            self.remaining_total_budget -= cost
            executed_moves.append((i, prev, a))

        # agent 結算移動後，整個路網的交通狀況隨機變化
        if self.dynamic_traffic:
            self._update_traffic_step_by_step()

        done = (self.remaining_total_budget <= 0) or (self.current_step >= self.max_steps)
        return self._get_state(), rewards, done, {"executed_moves": executed_moves}, self._build_graph_data()
