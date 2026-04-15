import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import GATConv
import numpy as np
import pandas as pd
import time

class GATActorCritic(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_agents, num_nodes):
        super().__init__()
        self.num_agents = num_agents
        self.gat1 = GATConv(input_dim, hidden_dim, heads=4, concat=True)
        self.gat2 = GATConv(hidden_dim*4, hidden_dim, heads=1)
        self.actor = nn.Linear(hidden_dim, num_agents)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.elu(self.gat1(x, edge_index))
        x = F.elu(self.gat2(x, edge_index))
        logits = self.actor(x).t()
        return logits, self.critic(x).mean(dim=0)

class PPOGATAgent:
    def __init__(self, env, lr=0.002, gamma=0.99, clip_epsilon=0.2, epochs=4,
                 hidden_dim=64, temperature=0.5):
        self.env = env
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.epochs = epochs
        self.temperature = temperature
        self.model = GATActorCritic(env.input_dim if hasattr(env, 'input_dim') else 4,
                                    hidden_dim, env.num_agents, env.num_nodes)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def get_action(self, state, graph_data):
        with torch.no_grad():
            logits, _ = self.model(graph_data)
            logits = logits / self.temperature
            action_probs = F.softmax(logits, dim=1)

        actions, log_probs = [], []
        for i in range(self.env.num_agents):
            probs = action_probs[i].cpu().numpy()

            
            if np.any(np.isnan(probs)) or np.sum(probs) == 0:
                
                curr_node = self.env.agent_positions[i]
                neighbors = list(self.env.graph.neighbors(curr_node))
                fallback_probs = np.zeros_like(probs)
                if neighbors:
                    fallback_probs[neighbors] = 1.0 / len(neighbors)
                else:
                    fallback_probs[:] = 1.0 / self.env.num_nodes
                probs = fallback_probs

            action = np.random.choice(self.env.num_nodes, p=probs)
            actions.append(action)
            log_probs.append(torch.log(action_probs[i][action] + 1e-8))

        return actions, torch.stack(log_probs).mean()

    def train(self, num_episodes=100):
        logs = []
        start_time = time.time()

        for episode in range(num_episodes):
            state, graph_data = self.env.reset()
            buffer = []
            done = False

            while not done:
                actions, log_prob = self.get_action(state, graph_data)
                next_state, reward, done, info, next_graph_data = self.env.step(actions)

                _, value = self.model(graph_data)
                buffer.append((state, graph_data, actions, reward, log_prob.detach(), value.detach().squeeze()))

                state = next_state
                graph_data = next_graph_data

            states, graph_data_list, actions_list, rewards, old_log_probs, values = zip(*buffer)
            returns = []
            G = 0
            for r in reversed(rewards):
                G = r + self.gamma * G
                returns.insert(0, G)
            returns = torch.tensor(returns, dtype=torch.float32)
            values = torch.stack(values)
            advantages = returns - values

            for _ in range(self.epochs):
                for idx in range(len(buffer)):
                    _, new_value = self.model(graph_data_list[idx])
                    new_action_probs, _ = self.model(graph_data_list[idx])
                    a = actions_list[idx]
                    new_log_prob = torch.stack([
                        torch.log(new_action_probs[i][a[i]] + 1e-8)
                        for i in range(self.env.num_agents)
                    ]).mean()
                    ratio = torch.exp(new_log_prob - old_log_probs[idx])
                    surr1 = ratio * advantages[idx]
                    surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages[idx]
                    actor_loss = -torch.min(surr1, surr2)
                    critic_loss = (returns[idx] - new_value.squeeze()) ** 2
                    loss = actor_loss + critic_loss

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

            logs.append([episode, sum(rewards), loss.item()])
            if episode % 10 == 0:
                print(f"[Episode {episode}] Total Reward: {sum(rewards):.2f} | Loss: {loss.item():.4f}")

        print(f"訓練時間：{time.time() - start_time:.2f} 秒")
        return pd.DataFrame(logs, columns=["Episode", "Total Reward", "Loss"])

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()
        print(f"成功載入模型參數：{path}")

    def evaluate(self, num_runs=10):
        all_rewards = []
        all_costs = []
        start_time = time.time()

        print(f"\n=== 評估結果（執行 {num_runs} 次） ===")
        for run in range(num_runs):
            state, graph_data = self.env.reset()
            done = False
            total_cost = 0
            agent_costs = [0] * self.env.num_agents
            agent_paths = [[p] for p in self.env.agent_positions]
            agent_rewards = [0] * self.env.num_agents
            collected_nodes = set([0])

            while not done:
                action, _ = self.get_action(state, graph_data)
                next_state, _, done, info, graph_data = self.env.step(action)

                executed_moves = info.get("executed_moves", [])
                for (agent_idx, from_node, to_node) in executed_moves:
                    if self.env.graph.has_edge(from_node, to_node):
                        cost = self.env.graph[from_node][to_node]['weight']
                        agent_costs[agent_idx] += cost
                        total_cost += cost
                    agent_paths[agent_idx].append(to_node)
                    if to_node not in collected_nodes and self.env.node_rewards[to_node][0] > 0:
                        agent_rewards[agent_idx] += self.env.node_rewards[to_node][0]
                        collected_nodes.add(to_node)

                state = next_state

            total_reward = sum(agent_rewards)
            all_rewards.append(total_reward)
            all_costs.append(total_cost)

            print(f"[Run {run}] Total Reward: {total_reward} | Total Cost: {total_cost}")
            for i in range(self.env.num_agents):
                print(f"  Agent {i} - Reward: {agent_rewards[i]:.2f} | Cost: {agent_costs[i]} | Path: {agent_paths[i]}")

        elapsed_time = time.time() - start_time
        print(f"=== 平均結果 ===")
        print(f"Average Reward: {np.mean(all_rewards):.2f} | Average Cost: {np.mean(all_costs):.2f}")
        print(f"總評估時間：{elapsed_time:.2f} 秒")
        print(f"平均每次評估耗時：{elapsed_time / num_runs:.2f} 秒")

        return pd.DataFrame({
            "Total Reward": all_rewards,
            "Total Cost": all_costs
        })
