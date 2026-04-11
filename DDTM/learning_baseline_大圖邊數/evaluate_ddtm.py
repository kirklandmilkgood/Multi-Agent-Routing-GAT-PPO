import torch
import numpy as np
from model_ddtm import DDTM
from env_matp import MATPEnv
import networkx as nx
import pandas as pd
import os
import time

def load_input_graph(file_path="../../dataset/Large_network.xlsx"):
    if os.path.exists(file_path):
        df = pd.read_excel(file_path)
        G = nx.Graph()
        for _, row in df.iterrows():
            G.add_edge(int(row['From']), int(row['To']), weight=int(row['Weight']))
        print("已載入輸入圖")
    else:
        raise FileNotFoundError(f"找不到檔案: {file_path}")
    return G

def prepare_node_features(graph, rewards, visited):
    node_feats = []
    for node in sorted(graph.nodes()):
        r = rewards[node]
        v = visited[node]
        node_feats.append([r, v, 0])
    return torch.tensor(node_feats, dtype=torch.float)

def evaluate(num_agents, total_budget, per_agent_budget, dataset, is_dynamic, model_path="ddtm_memsafe_final.pt"):
    start_time = time.time()
    device = torch.device("cpu")
    max_step = 400

    G = load_input_graph(dataset)      
    n_nodes = G.number_of_nodes()
    # model = DDTM(num_nodes=n_nodes).to(device) 大圖會OOM
    model = DDTM(num_nodes=n_nodes, embed_dim=32, nhead=2, num_layers=1).to(device) #為了避免OOM
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    total_rewards = []

    for episode in range(10):
        rewards = {n: np.random.randint(5, 20) for n in G.nodes()}
        env = MATPEnv(graph=G, rewards=rewards, num_agents=num_agents, total_budget=total_budget, per_agent_budget=per_agent_budget, dynamic_traffic=is_dynamic)
        obs = env.reset()

        episode_reward = 0
        done = False
        step_count = 0
        global_budget_left = total_budget
        agent_paths = [[pos] for pos in obs["agent_pos"]]
        agent_costs = [0.0 for _ in range(num_agents)]
        agent_rewards = [0.0 for _ in range(num_agents)]

        while not done and step_count < max_step:
            node_feats = prepare_node_features(env.graph, obs["node_rewards"], obs["visited"]).unsqueeze(0).to(device)
            agent_pos = torch.tensor(obs["agent_pos"], dtype=torch.long).unsqueeze(0).to(device)
            budget_left = torch.tensor(obs["budget_left"], dtype=torch.float).unsqueeze(0).to(device)
            visited_mask = torch.tensor(obs["visited"], dtype=torch.bool).unsqueeze(0).to(device)

            with torch.no_grad():
                logits = model(node_feats, agent_pos, budget_left, visited_mask)

                temperature = 5.0
                probs = torch.nn.functional.softmax(logits / temperature, dim=-1)
                if torch.isnan(probs).any() or (probs.sum(-1) == 0).any():
                    print("[探索 fallback] 使用 uniform 分布")
                    mask = (logits != -1e9).float()
                    probs = mask / mask.sum(-1, keepdim=True).clamp(min=1e-8)

                dist = torch.distributions.Categorical(probs)
                actions = dist.sample()

            next_positions = actions.squeeze(0).tolist()
            updated_positions = obs["agent_pos"].copy()

            for i in range(num_agents):
                cur = obs["agent_pos"][i]
                nxt = next_positions[i]

                
                valid_neighbors = [
                    n for n in G.neighbors(cur)
                    if obs["budget_left"][i] >= G[cur][n]["weight"] and global_budget_left >= G[cur][n]["weight"]
                ]
                if (cur == nxt or not G.has_edge(cur, nxt) or obs["budget_left"][i] < G[cur][nxt]["weight"]) and valid_neighbors:
                    nxt = np.random.choice(valid_neighbors)
                    print(f"[Agent {i}] fallback 隨機移動到 {nxt}")

                if cur != nxt and G.has_edge(cur, nxt):
                    cost = G[cur][nxt]['weight']
                    if obs["budget_left"][i] >= cost and global_budget_left >= cost:
                        agent_costs[i] += cost
                        global_budget_left -= cost
                        if obs["visited"][nxt] == 0:
                            agent_rewards[i] += rewards[nxt]
                        updated_positions[i] = nxt
                        agent_paths[i].append(nxt)

            obs, reward, done, _ = env.step(updated_positions)
            episode_reward += reward
            step_count += 1

        print(f"Episode {episode} Total Reward: {episode_reward}")
        for i in range(num_agents):
            print(f"  Agent {i} path: {agent_paths[i]}")
            print(f"    Cost: {agent_costs[i]:.2f}, Collected Reward: {agent_rewards[i]:.2f}")
        total_rewards.append(episode_reward)

    elapsed = time.time() - start_time
    print(f"Average Reward over 10 episodes: {np.mean(total_rewards):.2f}")
    print(f"Total Evaluation Time: {elapsed:.2f} seconds")
