import gym
import numpy as np
import networkx as nx
from gym import spaces

class MATPEnv(gym.Env):
    def __init__(self, graph: nx.Graph, rewards: dict, num_agents: int, total_budget: float, per_agent_budget: float, start_node: int = 0):
        super(MATPEnv, self).__init__()
        self.graph = graph
        self.raw_rewards = rewards.copy()
        self.num_agents = num_agents
        self.total_budget = total_budget
        self.per_agent_budget = per_agent_budget
        self.start_node = start_node

        self.reset()

        self.action_space = spaces.Discrete(len(self.graph.nodes))
        self.observation_space = spaces.Dict({
            "agent_pos": spaces.Box(low=0, high=len(self.graph.nodes), shape=(self.num_agents,), dtype=np.int32),
            "budget_left": spaces.Box(low=0, high=total_budget, shape=(self.num_agents,), dtype=np.float32),
            "node_rewards": spaces.Box(low=0, high=np.max(list(self.raw_rewards.values())), shape=(len(self.graph.nodes),), dtype=np.float32),
            "visited": spaces.MultiBinary(len(self.graph.nodes))
        })

    def reset(self):
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

        done = all(b <= 0 for b in self.budget_left)
        obs = self._get_obs()
        return obs, sum(rewards_gained), done, {}

    def render(self, mode='human'):
        for i in range(self.num_agents):
            print(f"Agent {i}: Pos {self.agent_positions[i]}, Budget {self.budget_left[i]:.2f}")
        print(f"Visited: {np.nonzero(self.visited)[0]}")
        print(f"Total reward collected: {self.total_collected_reward}")
