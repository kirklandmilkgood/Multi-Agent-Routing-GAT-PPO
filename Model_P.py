import numpy as np
import networkx as nx
from sklearn.cluster import KMeans
import random
import time
import os
import pandas as pd

class RewardBudgetHeuristic:
    def __init__(self, graph: nx.Graph, node_rewards: dict, total_budget: int, per_agent_budget: int, num_agents: int = 3, seed: int = 42, fix_layout: bool = False):
        self.graph = graph
        self.node_rewards = node_rewards
        self.total_budget = total_budget
        self.per_agent_budget = per_agent_budget
        self.num_agents = num_agents
        self.seed = seed
        self.positions = nx.spring_layout(graph, seed=self.seed) if fix_layout else nx.spring_layout(graph)
        self.nodes = list(graph.nodes())
        self.agent_clusters = {}
        self.routes = []
        self.start_nodes = [0] * self.num_agents
        self.node_rewards[0] = (0, 0)

    def cluster_nodes(self):
        coords = np.array([self.positions[v] for v in self.nodes])
        init_centers = np.array([self.positions[0]] * self.num_agents)
        kmeans = KMeans(n_clusters=self.num_agents, init=init_centers, n_init=1).fit(coords)
        labels = kmeans.labels_
        clusters = {i: [] for i in range(self.num_agents)}
        for idx, label in enumerate(labels):
            clusters[label].append(self.nodes[idx])
        self.agent_clusters = clusters

    def greedy_route_within_cluster(self, cluster_nodes, global_budget_left):
        subgraph = self.graph.subgraph(cluster_nodes)
        start = self.start_nodes[self.current_agent]
        path = [start]
        visited = set([start])
        local_budget_left = self.per_agent_budget
        reward_collected = 0
        cost_used = 0

        while True:
            current = path[-1]
            candidates = [(n, subgraph[current][n]['weight']) for n in subgraph.neighbors(current) if n not in visited]
            if not candidates:
                break
           
            candidates = [(n, cost, self.node_rewards[n][0]) for n, cost in candidates]
            candidates = sorted(candidates, key=lambda x: -x[2])  
            for n, cost, _ in candidates:
                if local_budget_left - cost >= 0 and global_budget_left - cost >= 0:
                    path.append(n)
                    visited.add(n)
                    local_budget_left -= cost
                    global_budget_left -= cost
                    cost_used += cost
                    if self.node_rewards[n][1] > 0:
                        reward_collected += self.node_rewards[n][0]
                        self.node_rewards[n] = (0, 0)
                    break
                else:
                    return path, reward_collected, cost_used, global_budget_left
        return path, reward_collected, cost_used, global_budget_left

    def run(self):
        self.cluster_nodes()
        all_routes = []
        global_budget_left = self.total_budget
        for i in range(self.num_agents):
            self.current_agent = i
            if self.start_nodes[i] not in self.agent_clusters[i]:
                self.agent_clusters[i].insert(0, self.start_nodes[i])
            route, reward, spent, global_budget_left = self.greedy_route_within_cluster(self.agent_clusters[i], global_budget_left)
            all_routes.append({
                'agent': i,
                'route': route,
                'reward': reward,
                'cost': spent
            })
        return all_routes

if __name__ == "__main__":
    start_time = time.time()

    #輸入要讀取檔案路徑
    file_path = "/home/lu/paper_code/dataset/ER_Graph_50Nodes.xlsx"
    if os.path.exists(file_path):
        df = pd.read_excel(file_path)
        G = nx.Graph()
        for _, row in df.iterrows():
            G.add_edge(int(row['From']), int(row['To']), weight=int(row['Weight']))
        print("有讀到輸入的檔案!!!")
    else:
        G = nx.erdos_renyi_graph(50, 0.6)
        while not nx.is_connected(G):
            G = nx.erdos_renyi_graph(50, 0.6)
        for (u, v) in G.edges():
            G[u][v]['weight'] = np.random.randint(1, 10)
        print("Noooooo讀到輸入的檔案!!!")

    rewards = {n: (np.random.randint(5, 20), 1) for n in G.nodes()}
    rewards[0] = (0, 0)

    #設定agent數量、總預算、個人預算
    total_budget = 100
    per_agent_budget = 50
    num_agents = 4

    heuristic = RewardBudgetHeuristic(G, node_rewards=rewards, total_budget=total_budget, per_agent_budget=per_agent_budget, num_agents=num_agents)
    results = heuristic.run()

    total_reward = 0
    total_cost = 0
    for res in results:
        print(f"Agent {res['agent']} | Reward: {res['reward']} | Cost: {res['cost']} | Route: {res['route']}")
        total_reward += res['reward']
        total_cost += res['cost']

    print(f"Total Reward Collected by All Agents: {total_reward}")
    print(f"Total Cost Spent by All Agents: {total_cost}")
    print(f"Remaining Budget: {heuristic.total_budget - total_cost}")

    end_time = time.time()
    print(f"Total Execution Time: {end_time - start_time:.2f} seconds")
