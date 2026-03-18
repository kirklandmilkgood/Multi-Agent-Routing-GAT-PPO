import os
import torch
import pandas as pd
from mappo_agent import MAPPOAgent  # MAPPO adapter
from gat_env import MultiAgentTSPEnv, load_graph_from_excel

train_num_nodes = 2636
train_num_edges = 8500
eval_file_path = "../../dataset/road-minnesota.xlsx"
# 啟動訓練環境

train_env = MultiAgentTSPEnv(train_num_nodes, train_num_edges, num_agents=4, total_budget=200, individual_budget=150)

agent = MAPPOAgent(train_env)

# 訓練模型
train_log = agent.train(num_episodes=100)

# 執行評估
eval_env = MultiAgentTSPEnv(num_nodes=train_num_nodes, num_agents=4, total_budget=200, individual_budget=150)
eval_env.fixed_graph = load_graph_from_excel(eval_file_path)
agent.env = eval_env

eval_df = agent.evaluate(num_runs=10)
eval_df.to_excel(f"mappo_output_logs.xlsx", index=False)