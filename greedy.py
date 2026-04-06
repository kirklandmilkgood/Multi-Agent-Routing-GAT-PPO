import random
import networkx as nx
import pandas as pd
import numpy as np
import os
import sys
import json
import time
from pathlib import Path


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
        print(f"experiment setting: num nodes: {num_nodes}, num edges: {num_edges}, num agents: {num_agents}, total budget: {t_budget}, individual budget: {i_budget}...")
        filtered_parts = [part for part in dataset_path.parts if part != '..']
        # 重新組合路徑
        eval_file_path = Path(*filtered_parts)
    
        start_time = time.time()

        G = load_or_generate_graph(eval_file_path)
        rewards = {n: np.random.randint(5, 20) for n in G.nodes()}

        #設定agent數量、總預算
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
