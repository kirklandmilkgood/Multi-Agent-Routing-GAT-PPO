import os
import torch
import pandas as pd
from dgn_agent import DGNAgent  # dgn adapter
from gat_env import MultiAgentTSPEnv, load_graph_from_excel

train_num_nodes = 2636
train_num_edges = 8500
eval_file_path = "../../dataset/road-minnesota.xlsx"

# 啟動訓練環境
train_env = MultiAgentTSPEnv(train_num_nodes, train_num_edges, num_agents=4, total_budget=150, individual_budget=100)

agent = DGNAgent(train_env, hidden_dim=64, buffer_capacity=10000, batch_size=32)

# 訓練模型
train_log = agent.train(num_episodes=100)

# 執行評估
eval_env = MultiAgentTSPEnv(num_nodes=train_num_nodes, num_agents=4, total_budget=150, individual_budget=100)
eval_env.fixed_graph = load_graph_from_excel(eval_file_path)
agent.env = eval_env 

eval_df = agent.evaluate(num_runs=10)
eval_df.to_excel(f"vdn_output_logs.xlsx", index=False)