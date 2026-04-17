import numpy as np
import networkx as nx
import torch
import gym
from torch_geometric.data import Data
import pandas as pd
import os

def generate_random_graph(num_nodes, edge_prob=0.2):
    G = nx.erdos_renyi_graph(num_nodes, p=edge_prob)
    while not nx.is_connected(G):
        G = nx.erdos_renyi_graph(num_nodes, p=edge_prob)
    for u, v in G.edges():
        G[u][v]['weight'] = np.random.randint(1, 10)
    return G

def load_graph_from_excel(file_path):
    if not os.path.exists(file_path):
        print("評估檔案不存在")
        return None

    df = pd.read_excel(file_path)
    G = nx.Graph()
    print("已載入評估圖檔\n")
    for _, row in df.iterrows():
        u, v, w = int(row['From']), int(row['To']), float(row['Weight'])
        G.add_edge(u, v, weight=w)
    return G

class MultiAgentTSPEnv(gym.Env):
    def __init__(self, num_nodes=50, num_agents=2, total_budget=100, individual_budget=100,
                 eval_file_path=None):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_agents = num_agents
        self.total_budget = total_budget
        self.individual_budget = individual_budget
        self.eval_file_path = eval_file_path
        self.fixed_graph = None

        if self.eval_file_path:
            self.fixed_graph = load_graph_from_excel(self.eval_file_path)
            self.num_nodes = self.fixed_graph.number_of_nodes()

        self.reset()

    def _build_graph_data(self):
        node_features = []
        for i in range(self.num_nodes):
            reward = self.node_rewards.get(i, (0,))[0] if self.visited.get(i, 0) > 0 else 0
            visited_flag = 1 if self.visited.get(i, 0) == 0 else 0
            is_agent_here = 1 if i in self.agent_positions else 0
            has_reward = 1 if (self.visited.get(i, 0) > 0 and reward > 0) else 0
            node_features.append([reward, visited_flag, is_agent_here, has_reward])

        x = torch.tensor(node_features, dtype=torch.float32)
        edge_index = []
        for u, v in self.graph.edges():
            edge_index.append([u, v])
            edge_index.append([v, u])
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        return Data(x=x, edge_index=edge_index)

    def reset(self):
        self.graph = self.fixed_graph if self.fixed_graph else generate_random_graph(self.num_nodes)
        self.node_rewards = {}

        for node in self.graph.nodes():
            self.node_rewards[node] = (np.random.randint(5, 20), 1)

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
        self.max_steps = 200
        return self._get_state(), self._build_graph_data()

    def _get_state(self):
        reward_vector = np.array([self.node_rewards.get(i, (0,))[0] if self.visited.get(i, 0) > 0 else 0
                                  for i in range(self.num_nodes)])
        visited_vector = np.array([self.visited.get(i, 0) for i in range(self.num_nodes)])
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
            elif self.visited.get(a, 0) > 0:
                rewards += self.node_rewards[a][0] + 10
                self.visited[a] = 0
            else:
                penalty = min(self.visit_count[a] ** 2, 10)
                rewards -= 10 * penalty

            self.remaining_agent_budgets[i] -= cost
            self.remaining_total_budget -= cost
            executed_moves.append((i, prev, a))

        done = (self.remaining_total_budget <= 0) or (self.current_step >= self.max_steps)
        return self._get_state(), rewards, done, {"executed_moves": executed_moves}, self._build_graph_data()
