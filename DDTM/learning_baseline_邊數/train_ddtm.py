import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from model_ddtm import DDTM
from env_matp import MATPEnv
import networkx as nx
import random
import gc
import time

#設定訓練圖
def build_er_graph(n_nodes=2643, n_edges=6000):
    max_attempts = 1000
    for _ in range(max_attempts):
        G = nx.gnm_random_graph(n=n_nodes, m=n_edges)
        if nx.is_connected(G):
            break
    else:
        raise ValueError(f"無法在 {max_attempts} 次內產生連通圖 (n={n_nodes}, m={n_edges})")

    for (u, v) in G.edges():
        G[u][v]['weight'] = np.random.randint(1, 10)
    return G

def prepare_node_features(graph, rewards, visited):
    node_feats = []
    for node in sorted(graph.nodes()):
        r = rewards[node]
        v = visited[node]
        node_feats.append([r, v, 0])
    return torch.tensor(node_feats, dtype=torch.float)

def train(num_agents, total_budget, per_agent_budget, n_nodes, n_edges, n_episodes):
    start_time = time.time()
    device = torch.device("cpu")
    max_step = 200
    entropy_weight = 0.01

    # model = DDTM(num_nodes=n_nodes).to(device) 大圖會OOM
    model = DDTM(num_nodes=n_nodes, embed_dim=32, nhead=2, num_layers=1).to(device) #為了避免OOM

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    rewards_history = []

    for episode in range(n_episodes):
        graph = build_er_graph(n_nodes=n_nodes, n_edges=n_edges)
        print(f"Episode {episode} | Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

        rewards = {n: np.random.randint(5, 20) for n in graph.nodes()}
        env = MATPEnv(graph, rewards, num_agents=num_agents, total_budget=total_budget, per_agent_budget=per_agent_budget)

        obs = env.reset()
        log_probs = []
        entropies = []
        total_reward = 0.0
        step_count = 0
        done = False

        while not done and step_count < max_step:
            node_feats = prepare_node_features(env.graph, obs["node_rewards"], obs["visited"]).unsqueeze(0).to(device)
            agent_pos = torch.tensor(obs["agent_pos"], dtype=torch.long).unsqueeze(0).to(device)
            budget_left = torch.tensor(obs["budget_left"], dtype=torch.float).unsqueeze(0).to(device)
            visited_mask = torch.tensor(obs["visited"], dtype=torch.bool).unsqueeze(0).to(device)

            logits = model(node_feats, agent_pos, budget_left, visited_mask)
            actions, log_prob, entropy = model.select_action(logits)

            obs, reward, done, _ = env.step(actions.squeeze(0).tolist())
            log_probs.append(log_prob.sum())
            entropies.append(entropy.sum())
            total_reward += reward
            step_count += 1

        policy_loss = -torch.stack(log_probs).sum() * total_reward
        entropy_bonus = -entropy_weight * torch.stack(entropies).sum()
        loss = policy_loss + entropy_bonus

        # 記錄該 episode 的總分
        rewards_history.append(total_reward)

        if not torch.isfinite(loss):
            print(f"Episode {episode}: Loss is NaN or Inf, skipping update.")
            continue

        optimizer.zero_grad()
        try:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        except RuntimeError as e:
            print(f"RuntimeError during backward pass: {e}")
            continue

        print(f"Episode {episode}, Total Reward: {total_reward}, Loss: {loss.item():.4f}")

        if episode % 10 == 0:
            torch.save(model.state_dict(), f"ddtm_memsafe_checkpoint_ep{episode}.pt")

        
        del env, graph, rewards, obs, node_feats, agent_pos, budget_left, visited_mask
        gc.collect()

    torch.save(model.state_dict(), "ddtm_memsafe_final.pt")
    total_time = time.time() - start_time
    print(f"Total training time: {total_time:.2f} seconds")
    return rewards_history

if __name__ == "__main__":
    train()
