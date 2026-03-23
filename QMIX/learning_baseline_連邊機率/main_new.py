import os
import torch
import pandas as pd
from qmix_agent import QMIXAgent  # qmix adapter
from gat_env import MultiAgentTSPEnv, load_graph_from_excel

# 啟動訓練環境
train_env = MultiAgentTSPEnv(num_nodes=50, num_agents=4, total_budget=150, individual_budget=100)

agent = QMIXAgent(train_env, hidden_dim=64, buffer_capacity=10000, batch_size=32)

# 訓練模型
train_log = agent.train(num_episodes=50)

# 執行評估
eval_file_path = "../../dataset/ER_Graph_50Nodes.xlsx"
eval_env = MultiAgentTSPEnv(num_nodes=50, num_agents=4, total_budget=150, individual_budget=100)
eval_env.fixed_graph = load_graph_from_excel(eval_file_path)
agent.env = eval_env 

eval_df = agent.evaluate(num_runs=10)
eval_df.to_excel(f"qmix_output_logs.xlsx", index=False)