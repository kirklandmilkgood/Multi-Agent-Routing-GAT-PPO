import random
import networkx as nx
import pandas as pd
import numpy as np
import os
import time
from copy import deepcopy


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


def load_or_generate_graph(file_path="/home/lu/paper_code/dataset/ER_Graph_50Nodes.xlsx"): #輸入要讀取圖的路徑
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
    start_time = time.time()

    G = load_or_generate_graph()
    rewards = {n: int(np.random.randint(5, 20)) for n in G.nodes()}

    #設定agent數量、總預算、個人預算
    num_agents = 4
    total_budget = 100
    per_agent_budget = 50
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
