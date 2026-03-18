import random
import networkx as nx
import pandas as pd
import numpy as np
import os
import time


class MATPRandomWalkSolver:
    def __init__(self, graph: nx.Graph, rewards: dict, total_budget: float, per_agent_budget: float,
                 num_agents: int, start_nodes: list, max_steps=100):
        self.graph = graph
        self.rewards = rewards
        self.total_budget = total_budget
        self.per_agent_budget = per_agent_budget
        self.num_agents = num_agents
        self.start_nodes = start_nodes
        self.max_steps = max_steps
        self.collected = {v: False for v in graph.nodes}
        self.paths = [[start_nodes[i]] for i in range(num_agents)]
        self.travel_costs = [0.0 for _ in range(num_agents)]
        self.total_reward = 0.0

    def run(self):
        G = self.graph

        for i in range(self.num_agents):
            current = self.start_nodes[i]
            if not self.collected[current]:
                self.collected[current] = True
                self.total_reward += self.rewards[current]

            for _ in range(self.max_steps):
                neighbors = list(G.neighbors(current))
                if not neighbors:
                    break

                next_node = random.choice(neighbors)
                edge_cost = G[current][next_node]['weight']

                projected_cost = self.travel_costs[i] + edge_cost
                total_projected_cost = sum(self.travel_costs) - self.travel_costs[i] + projected_cost

                if projected_cost > self.per_agent_budget or total_projected_cost > self.total_budget:
                    break

                self.paths[i].append(next_node)
                self.travel_costs[i] = projected_cost

                if not self.collected[next_node]:
                    self.collected[next_node] = True
                    self.total_reward += self.rewards[next_node]

                current = next_node

        return self.paths, self.total_reward, self.travel_costs


def load_or_generate_graph(file_path="/home/lu/paper_code/dataset/ER_Graph_50Nodes.xlsx"):
    #輸入要讀取檔案路徑
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
    rewards = {n: np.random.randint(5, 20) for n in G.nodes()}

    #設定agent數量、總預算、個人預算限制
    num_agents = 4
    total_budget = 100
    per_agent_budget = 50
    start_node = 0
    start_nodes = [start_node] * num_agents

    solver = MATPRandomWalkSolver(G, rewards, total_budget, per_agent_budget, num_agents, start_nodes)
    paths, total_reward, costs = solver.run()

    print("Start Nodes:", start_nodes)
    print("Paths:", paths)
    print("Total Reward Collected:", total_reward)
    print("Costs per Agent:", costs)
    print("Total Cost Used:", sum(costs))

    end_time = time.time()
    print(f"Total Execution Time: {end_time - start_time:.2f} seconds")
