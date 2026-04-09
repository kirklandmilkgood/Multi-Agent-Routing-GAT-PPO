import gym
import numpy as np
import networkx as nx
from gym import spaces

class MATPEnv(gym.Env):
    def __init__(self, graph: nx.Graph, rewards: dict, num_agents: int, total_budget: float, per_agent_budget: float, start_node: int = 0, dynamic_traffic: bool = False, change_prob: float = 0.10):
        super(MATPEnv, self).__init__()
        self.graph = graph
        self.raw_rewards = rewards.copy()
        self.num_agents = num_agents
        self.total_budget = total_budget
        self.per_agent_budget = per_agent_budget
        self.start_node = start_node

        # dynamic 開關與機率設定
        self.dynamic_traffic = dynamic_traffic
        self.change_prob = change_prob

        self.reset()

        self.action_space = spaces.Discrete(len(self.graph.nodes))
        self.observation_space = spaces.Dict({
            "agent_pos": spaces.Box(low=0, high=len(self.graph.nodes), shape=(self.num_agents,), dtype=np.int32),
            "budget_left": spaces.Box(low=0, high=total_budget, shape=(self.num_agents,), dtype=np.float32),
            "node_rewards": spaces.Box(low=0, high=np.max(list(self.raw_rewards.values())), shape=(len(self.graph.nodes),), dtype=np.float32),
            "visited": spaces.MultiBinary(len(self.graph.nodes))
        })

    # dynamic 機制函數
    def _update_traffic_step_by_step(self):
        for u, v, data in self.graph.edges(data=True):
            # 確保已記錄該條路的初始靜態權重 (t=0)
            if 'base_weight' not in data:
                data['base_weight'] = data['weight']
                
            # 以 change_prob 的機率觸發路況變化
            if np.random.rand() < self.change_prob:
                delta = int(np.random.choice([-2, -1, 1, 2]))
                new_weight = data['weight'] + delta
                # bounded limits: 1 to 10
                self.graph[u][v]['weight'] = max(1, min(10, new_weight))

    def reset(self):
        # 確保新的 episode 開始時，恢復基準狀態，並記錄 base_weight
        for u, v, data in self.graph.edges(data=True):
            if 'base_weight' in data:
                self.graph[u][v]['weight'] = data['base_weight']
            else:
                data['base_weight'] = data['weight']
                
        self.agent_positions = [self.start_node] * self.num_agents
        self.budget_left = [self.per_agent_budget] * self.num_agents
        self.visited = np.zeros(len(self.graph.nodes), dtype=np.int32)
        self.rewards = self.raw_rewards.copy()
        self.rewards[self.start_node] = 0
        self.total_collected_reward = 0
        return self._get_obs()

    def _get_obs(self):
        return {
            "agent_pos": np.array(self.agent_positions, dtype=np.int32),
            "budget_left": np.array(self.budget_left, dtype=np.float32),
            "node_rewards": np.array([self.rewards[n] for n in sorted(self.graph.nodes)], dtype=np.float32),
            "visited": self.visited.copy()
        }

    def step(self, actions):
        rewards_gained = [0.0] * self.num_agents
        done = True

        for i, action in enumerate(actions):
            current = self.agent_positions[i]
            if action == current or self.visited[action]:
                continue
            if not self.graph.has_edge(current, action):
                continue

            cost = self.graph[current][action]["weight"]
            if self.budget_left[i] < cost:
                continue

            self.agent_positions[i] = action
            self.budget_left[i] -= cost

            if not self.visited[action]:
                rewards_gained[i] = self.rewards[action]
                self.total_collected_reward += self.rewards[action]
                self.rewards[action] = 0
                self.visited[action] = 1

        # agent 結算移動後，整個路網的交通狀況隨機變化
        if self.dynamic_traffic:
            self._update_traffic_step_by_step()

        done = all(b <= 0 for b in self.budget_left)
        obs = self._get_obs()
        return obs, sum(rewards_gained), done, {}

    def render(self, mode='human'):
        for i in range(self.num_agents):
            print(f"Agent {i}: Pos {self.agent_positions[i]}, Budget {self.budget_left[i]:.2f}")
        print(f"Visited: {np.nonzero(self.visited)[0]}")
        print(f"Total reward collected: {self.total_collected_reward}")
