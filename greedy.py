import random
import networkx as nx
import pandas as pd
import numpy as np
import os
import sys
import json
import time
from pathlib import Path
from copy import deepcopy

# 通用 dynamic 環境模擬函數
def evaluate_paths_in_dynamic_env(graph, paths, rewards_obj, total_budget, per_agent_budget, change_prob=0.10):
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


class MATPGreedySolver:
    def __init__(self, graph: nx.Graph, rewards: dict, total_budget: float, num_agents: int, start_nodes: list):
        self.graph = graph
        self.rewards = rewards
        self.total_budget = total_budget
        self.num_agents = num_agents
        self.start_nodes = start_nodes

        self.global_collected = {v: False for v in graph.nodes}

        
        self.paths = [[start_nodes[i]] for i in range(num_agents)]
        self.travel_costs = [0.0 for _ in range(num_agents)]

        
        self.total_reward = 0.0

    def run(self):
        per_agent_budget = self.total_budget / self.num_agents
        G = self.graph

        
        for i in range(self.num_agents):
            current = self.start_nodes[i]
            if not self.global_collected[current]:
                self.total_reward += self.rewards[current]
                self.global_collected[current] = True

        while True:
            best_gain = 0
            best_agent = None
            best_target = None
            best_path = None
            best_path_cost = float('inf')

            for i in range(self.num_agents):
                current = self.paths[i][-1]

                for target in G.nodes:
                    
                    if target == current:
                        continue

                    try:
                        path = nx.shortest_path(G, current, target, weight='weight')
                        cost = nx.path_weight(G, path, weight='weight')
                    except nx.NetworkXNoPath:
                        continue

                    if self.travel_costs[i] + cost > per_agent_budget:
                        continue

                    
                    gain = self.rewards[target]

                    if gain > best_gain:
                        best_gain = gain
                        best_agent = i
                        best_target = target
                        best_path = path
                        best_path_cost = cost

            if best_agent is None:
                break

           
            self.paths[best_agent].extend(best_path[1:])
            self.travel_costs[best_agent] += best_path_cost

            
            if not self.global_collected[best_target]:
                self.total_reward += self.rewards[best_target]
                self.global_collected[best_target] = True
            

        return self.paths, self.total_reward, self.travel_costs


def load_or_generate_graph(file_path="dataset/Large_network.xlsx"): 
    # 輸入要讀取檔案的路徑
    if os.path.exists(file_path):
        df = pd.read_excel(file_path)
        G = nx.Graph()
        for _, row in df.iterrows():
            G.add_edge(int(row['From']), int(row['To']), weight=int(row['Weight']))
        print("有讀到輸入的檔案!!!!!!!!!!")
    else:
        G = nx.erdos_renyi_graph(50, 0.6)
        while not nx.is_connected(G):
            G = nx.erdos_renyi_graph(50, 0.6)
        for (u, v) in G.edges():
            G[u][v]['weight'] = np.random.randint(1, 10)
    return G


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
        num_edges = configs[i]["num_edges"]
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

        G = load_or_generate_graph(eval_file_path)
        rewards = {n: np.random.randint(5, 20) for n in G.nodes()}

        # 設定agent數量、總預算
        total_budget = t_budget
        start_nodes = random.sample(list(G.nodes), num_agents)

        solver = MATPGreedySolver(G, rewards, total_budget, num_agents, start_nodes)
        paths, total_reward, costs = solver.run()

        print("Start Nodes:", start_nodes)
        print("Paths:", paths)
        print("Total Reward Collected:", total_reward)
        print("Costs per Agent:", costs)
        print("Total Cost Used:", sum(costs))

        end_time = time.time()
        print(f"Total Execution Time: {end_time - start_time:.2f} seconds")

        # 進行動態環境評估
        if dynamic_traffic:
            per_agent_budget = total_budget / num_agents
            dyn_reward, dyn_costs = evaluate_paths_in_dynamic_env(G, paths, rewards, total_budget, per_agent_budget, change_prob=0.10)
            print("\n--- 動態路網評估結果 (change_prob=0.10) ---")
            print(f"ACTUAL Total Reward Collected: {dyn_reward}")
            print(f"ACTUAL Costs per Agent: {dyn_costs}")
            print(f"ACTUAL Total Cost Used: {sum(dyn_costs)}")
