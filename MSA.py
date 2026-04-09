import numpy as np
import pandas as pd
import networkx as nx
import random
import os
import sys
import json
import time
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

class MATPInstance:
    def __init__(self, graph, rewards, num_agents, total_budget, per_agent_budget, start_node=0):
        self.graph = graph
        self.rewards = rewards
        self.num_agents = num_agents
        self.total_budget = total_budget
        self.per_agent_budget = per_agent_budget
        self.start_node = start_node

def load_graph_and_rewards(file_path):
    if os.path.exists(file_path):
        df = pd.read_excel(file_path)
        G = nx.Graph()
        for _, row in df.iterrows():
            G.add_edge(int(row['From']), int(row['To']), weight=int(row['Weight']))
        print(f"載入圖成功，共有 {G.number_of_nodes()} 節點與 {G.number_of_edges()} 邊")
    else:
        G = nx.erdos_renyi_graph(50, 0.6)
        while not nx.is_connected(G):
            G = nx.erdos_renyi_graph(50, 0.6)
        for (u, v) in G.edges():
            G[u][v]['weight'] = np.random.randint(1, 10)
        print("未找到檔案，已隨機生成連通圖")

    rewards_dict = {n: (np.random.randint(5, 20), 1) for n in G.nodes()}
    reward_list = [rewards_dict[n][0] for n in sorted(G.nodes())]
    return G, reward_list

class MATPSolverSA_MATP:
    def __init__(self, instance):
        self.instance = instance
        self.best_solution = None
        self.best_score = 0

    def total_cost(self, path):
        return sum(self.instance.graph[path[i]][path[i+1]]['weight']
                   for i in range(len(path)-1) if self.instance.graph.has_edge(path[i], path[i+1]))

    def total_reward(self, paths):
        visited = set()
        total = 0
        for path in paths:
            for node in path:
                if node not in visited:
                    visited.add(node)
                    total += self.instance.rewards[node]
        return total

    def is_valid_path(self, path):
        return all(self.instance.graph.has_edge(path[i], path[i+1]) for i in range(len(path)-1))

    def greedy_initial_solution(self):
        graph = self.instance.graph
        rewards = self.instance.rewards
        per_budget = self.instance.per_agent_budget
        total_budget = self.instance.total_budget
        num_agents = self.instance.num_agents
        start = self.instance.start_node

        solutions = []
        total_used = 0

        for _ in range(num_agents):
            current = start
            path = [current]
            cost = 0
            visited = set(path)

            while True:
                neighbors = [n for n in graph.neighbors(current) if n not in visited]
                if not neighbors:
                    break
                neighbors.sort(key=lambda n: -rewards[n])
                for next_node in neighbors:
                    edge_cost = graph[current][next_node]['weight']
                    if cost + edge_cost <= per_budget and total_used + edge_cost <= total_budget:
                        path.append(next_node)
                        cost += edge_cost
                        total_used += edge_cost
                        visited.add(next_node)
                        current = next_node
                        break
                else:
                    break
            solutions.append(path)
        return solutions

    def anneal(self, max_iter=1000, temp=1000, cooling_rate=0.995): #設定MSA參數
        current_solution = self.greedy_initial_solution()
        current_score = self.total_reward(current_solution)
        self.best_solution = current_solution
        self.best_score = current_score

        for _ in range(max_iter):
            neighbor = [path[:] for path in current_solution]

            if self.instance.num_agents >= 2:
                op = random.choice(['swap', 'insert'])
                a, b = random.sample(range(self.instance.num_agents), 2)
                if neighbor[a] and neighbor[b]:
                    if op == 'swap':
                        idx_a = random.randint(0, len(neighbor[a]) - 1)
                        idx_b = random.randint(0, len(neighbor[b]) - 1)
                        neighbor[a][idx_a], neighbor[b][idx_b] = neighbor[b][idx_b], neighbor[a][idx_a]
                    elif op == 'insert':
                        idx = random.randint(0, len(neighbor[a]) - 1)
                        node = neighbor[a].pop(idx)
                        neighbor[b].insert(random.randint(0, len(neighbor[b])), node)

            if not all(self.is_valid_path(path) for path in neighbor):
                continue

            total_cost = sum(self.total_cost(path) for path in neighbor)
            if total_cost > self.instance.total_budget:
                continue

            reward = self.total_reward(neighbor)
            acceptance_prob = np.exp((reward - current_score) / temp)
            if reward > current_score or random.random() < acceptance_prob:
                current_solution = neighbor
                current_score = reward
                if reward > self.best_score:
                    self.best_score = reward
                    self.best_solution = neighbor

            temp *= cooling_rate

        if self.best_solution is None:
            return [[] for _ in range(self.instance.num_agents)], 0
        return self.best_solution, self.best_score

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

        G, rewards = load_graph_and_rewards(eval_file_path)
        instance = MATPInstance(G, rewards, num_agents=num_agents, total_budget=t_budget, per_agent_budget=i_budget, start_node=0) 
        #設定agent數量、總預算、個人預算
        solver = MATPSolverSA_MATP(instance)

        start_time = time.time()
        solution, score = solver.anneal()
        end_time = time.time()

        if not any(solution):
            print("找不到合法解，請調整預算或圖的結構")
        else:
            print("\n最佳解（共同起點 0）：")
            for i, path in enumerate(solution):
                print(f"Agent {i+1}: {path}")
            total_cost = sum(solver.total_cost(path) for path in solution)
            print(f"\n收集報酬：{score}")
            print(f"總花費：{total_cost}")
            print(f"執行時間：{end_time - start_time:.2f} 秒")
        # 進行動態環境評估
            if dynamic_traffic:
                dyn_reward, dyn_costs = evaluate_paths_in_dynamic_env(G, solution, rewards, t_budget, i_budget, change_prob=0.10)
                print("\n--- 動態路網評估結果 (change_prob=0.10) ---")
                print(f"ACTUAL Total Reward Collected: {dyn_reward}")
                print(f"ACTUAL Costs per Agent: {dyn_costs}")
                print(f"ACTUAL Total Cost Used: {sum(dyn_costs)}")
