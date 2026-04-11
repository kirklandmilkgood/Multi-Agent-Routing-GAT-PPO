import numpy as np
import networkx as nx
from sklearn.cluster import KMeans
import random
import time
import os
import json
import sys
import pandas as pd
from pathlib import Path
from copy import deepcopy

# 通用 dynamic 環境模擬函數
def evaluate_paths_in_dynamic_env(graph, paths, rewards_obj, total_budget, per_agent_budget, change_prob=0.10, target_nodes=None):
    dynamic_G = deepcopy(graph)
    global_budget_left = total_budget
    local_budgets_left = [per_agent_budget] * len(paths)
    
    actual_total_reward = 0
    actual_costs = [0.0] * len(paths)
    global_collected = set()
    
    # 處理各演算法 rewards 格式不同的問題 (dict of int, dict of tuple, or list)
    def get_r(node):
        if isinstance(rewards_obj, dict):
            val = rewards_obj.get(node, 0)
            return val[0] if isinstance(val, tuple) else val
        elif isinstance(rewards_obj, list) or isinstance(rewards_obj, np.ndarray):
            return rewards_obj[node]
        return 0

    max_steps = max([len(p) for p in paths]) if paths else 0
    
    # 初始起點獎勵收集
    for p in paths:
        if p and p[0] not in global_collected:
            if target_nodes is None or p[0] in target_nodes:
                actual_total_reward += get_r(p[0])
            global_collected.add(p[0])
            
    # step by step 模擬
    for step in range(max_steps - 1):
        for i, path in enumerate(paths):
            if step < len(path) - 1 and local_budgets_left[i] > 0 and global_budget_left > 0:
                u = path[step]
                v = path[step+1]
                
                # 承受當下 dynamic 成本
                actual_cost = dynamic_G[u][v]['weight']
                
                if local_budgets_left[i] >= actual_cost and global_budget_left >= actual_cost:
                    local_budgets_left[i] -= actual_cost
                    global_budget_left -= actual_cost
                    actual_costs[i] += actual_cost
                    
                    if v not in global_collected:
                        if target_nodes is None or v in target_nodes:    
                            actual_total_reward += get_r(v)
                        global_collected.add(v)
                else:
                    local_budgets_left[i] = -1 # 標記破產，後續不再移動
                    
        # 路況隨機變動
        for u, v, data in dynamic_G.edges(data=True):
            if 'base_weight' not in data:
                data['base_weight'] = data['weight']
            if np.random.rand() < change_prob:
                delta = int(np.random.choice([-2, -1, 1, 2]))
                new_weight = data['weight'] + delta
                dynamic_G[u][v]['weight'] = max(1, min(10, new_weight))
                
    return actual_total_reward, actual_costs

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
    # 接收總指揮腳本傳來的 config 路徑，若無則預設抓取上層目錄的 config.json
    config_path = sys.argv[1] if len(sys.argv) > 1 else "large_network_config.json"
        
    if not os.path.exists(config_path):
            print(f"找不到設定檔: {config_path}")
            sys.exit(1)

    with open(config_path, 'r', encoding='utf-8') as f:
            configs = json.load(f)["experiments"]

    total_exps = len(configs)
    for i in range(total_exps):
        num_nodes = configs[i]["num_nodes"]
        num_edges = configs[i].get("num_edges", "-")
        num_agents = configs[i]["num_agents"]
        t_budget = configs[i]["total_budget"]
        i_budget = configs[i]["individual_budget"]
        dataset_path = Path(configs[i]["dataset"])
        num_episodes = configs[i]["episodes"]
        dynamic_traffic = True if configs[i]["dynamic"] else False
        print(f"experiment setting: num nodes: {num_nodes}, num edges: {num_edges}, num agents: {num_agents}, total budget: {t_budget}, individual budget: {i_budget}...")
        filtered_parts = [part for part in dataset_path.parts if part != '..']
        # 重新組合路徑
        eval_file_path = Path(*filtered_parts)

        start_time = time.time()

        #輸入要讀取檔案路徑
        if os.path.exists(eval_file_path):
            df = pd.read_excel(eval_file_path)
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
        total_budget = t_budget
        per_agent_budget = i_budget

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

        # 提取路徑並進行動態環境評估
        paths = [res['route'] for res in results]
        if dynamic_traffic:
            dyn_reward, dyn_costs = evaluate_paths_in_dynamic_env(G, paths, rewards, total_budget, per_agent_budget, change_prob=0.10)
            print("\n--- 動態路網評估結果 (change_prob=0.10) ---")
            print(f"ACTUAL Total Reward Collected: {dyn_reward}")
            print(f"ACTUAL Costs per Agent: {dyn_costs}")
            print(f"ACTUAL Total Cost Used: {sum(dyn_costs)}")
