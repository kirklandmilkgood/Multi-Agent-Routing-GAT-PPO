import random
import networkx as nx
import pandas as pd
import numpy as np
import os
import time
import json
import sys
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

class MATPGeneticSolver:
    def __init__(self, graph: nx.Graph, rewards: dict, total_budget: float, per_agent_budget: float,
                 num_agents: int, start_nodes: list,
                 population_size=30, generations=100, mutation_rate=0.5,
                 low_quantile=0.4, elite_k=2, seed=42): #設定GA參數

        self.graph = graph
        self.rewards = rewards
        self.total_budget = total_budget
        self.per_agent_budget = per_agent_budget
        self.num_agents = num_agents
        self.start_nodes = start_nodes
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.max_steps = 500

        self.low_quantile = low_quantile
        all_rewards = np.array([rewards[n] for n in graph.nodes], dtype=float)
        self.low_cut = float(np.quantile(all_rewards, self.low_quantile)) if len(all_rewards) else 0.0

        self.elite_k = max(0, int(elite_k))
        self.rng = random.Random(seed)
        np.random.seed(seed)

    
    def _choose_low_reward_neighbor(self, current):
        neighbors = list(self.graph.neighbors(current))
        if not neighbors:
            return None
        if self.rng.random() < 0.15:
            return self.rng.choice(neighbors)  
        best_val = float('inf')
        cand = []
        for v in neighbors:
            rv = float(self.rewards[v])
            if rv < best_val:
                best_val = rv
                cand = [v]
            elif rv == best_val:
                cand.append(v)
        return self.rng.choice(cand)

    
    def fitness(self, individual):
        total_reward = 0.0
        total_costs = [0.0 for _ in range(self.num_agents)]

        for i in range(self.num_agents):
            path = individual[i]
            cost = 0.0
            local_collected = set()

            for j in range(1, len(path)):
                u, v = path[j - 1], path[j]
                if not self.graph.has_edge(u, v):
                    return 0.0
                edge_cost = self.graph[u][v]['weight']
                if cost + edge_cost > self.per_agent_budget:
                    break
                if sum(total_costs) - total_costs[i] + cost + edge_cost > self.total_budget:
                    break

                cost += edge_cost

                
                if v not in local_collected:
                    rv = float(self.rewards[v])
                    if rv <= self.low_cut:
                        total_reward += 1.0  
                    local_collected.add(v)

            total_costs[i] = cost

        if sum(total_costs) > self.total_budget:
            return 0.0

        return total_reward

    
    def generate_individual(self):
        individual = []
        for i in range(self.num_agents):
            path = [self.start_nodes[i]]
            current = self.start_nodes[i]
            cost = 0.0
            steps = 0
            while steps < self.max_steps:
                nxt = self._choose_low_reward_neighbor(current)
                if nxt is None:
                    break
                edge_cost = self.graph[current][nxt]['weight']
                if cost + edge_cost > self.per_agent_budget:
                    break
                path.append(nxt)
                cost += edge_cost
                current = nxt
                steps += 1
            individual.append(path)
        return individual

    def crossover(self, p1, p2):
        return [deepcopy(self.rng.choice([p1[i], p2[i]])) for i in range(self.num_agents)]

    def mutate(self, individual):
        for i in range(self.num_agents):
            if self.rng.random() < self.mutation_rate:
                individual[i] = self.generate_individual()[i]

    def run(self):
        population = [self.generate_individual() for _ in range(self.population_size)]

        for _ in range(self.generations):
            population.sort(key=self.fitness, reverse=True)

            
            new_population = population[:self.elite_k] if self.elite_k > 0 else []

            
            while len(new_population) < self.population_size:
                parents = self.rng.sample(population, 2)
                child = self.crossover(parents[0], parents[1])
                self.mutate(child)
                new_population.append(child)

            population = new_population

        
        best = max(population, key=self.fitness)

        total_reward = 0.0
        total_costs = [0.0 for _ in range(self.num_agents)]
        valid_paths = [[] for _ in range(self.num_agents)]
        global_collected = set()

        for i in range(self.num_agents):
            path = best[i]
            if not path:
                valid_paths[i] = []
                continue
            valid_path = [path[0]]
            cost = 0.0
            for j in range(1, len(path)):
                u, v = path[j - 1], path[j]
                if not self.graph.has_edge(u, v):
                    break
                edge_cost = self.graph[u][v]['weight']
                if cost + edge_cost > self.per_agent_budget:
                    break
                if sum(total_costs) - total_costs[i] + cost + edge_cost > self.total_budget:
                    break

                cost += edge_cost
                valid_path.append(v)

                
                if v not in global_collected:
                    total_reward += self.rewards[v]
                    global_collected.add(v)

            total_costs[i] = cost
            valid_paths[i] = valid_path

        return valid_paths, total_reward, total_costs


def load_or_generate_graph(file_path="dataset/Large_network.xlsx"): #輸入要讀取圖的路徑
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

        G = load_or_generate_graph(eval_file_path)
        rewards = {n: int(np.random.randint(5, 20)) for n in G.nodes()}

        #設定agent數量、總預算、個人預算
        total_budget = t_budget
        per_agent_budget = i_budget
        start_node = 0
        start_nodes = [start_node] * num_agents

    
        solver = MATPGeneticSolver(
            G, rewards, total_budget, per_agent_budget, num_agents, start_nodes,
            population_size=30, generations=80, mutation_rate=0.6,
            low_quantile=0.5, elite_k=1, seed=42
        ) #設定GA參數
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
            dyn_reward, dyn_costs = evaluate_paths_in_dynamic_env(G, paths, rewards, total_budget, per_agent_budget, change_prob=0.10)
            print("\n--- 動態路網評估結果 (change_prob=0.10) ---")
            print(f"ACTUAL Total Reward Collected: {dyn_reward}")
            print(f"ACTUAL Costs per Agent: {dyn_costs}")
            print(f"ACTUAL Total Cost Used: {sum(dyn_costs)}")
