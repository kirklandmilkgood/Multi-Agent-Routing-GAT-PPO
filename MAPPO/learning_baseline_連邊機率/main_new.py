import os
import torch
import pandas as pd
from mappo_agent import MAPPOAgent  # MAPPO adapter
from gat_env import MultiAgentTSPEnv, load_graph_from_excel

# 啟動訓練環境
train_env = MultiAgentTSPEnv(num_nodes=1175, num_agents=4, total_budget=150, individual_budget=100)

agent = MAPPOAgent(train_env)

# 訓練模型
train_log = agent.train(num_episodes=100)

# 執行評估
eval_file_path = "../../dataset/road-euroroad.xlsx"
eval_env = MultiAgentTSPEnv(num_nodes=1175, num_agents=4, total_budget=150, individual_budget=100)
eval_env.fixed_graph = load_graph_from_excel(eval_file_path)
agent.env = eval_env 

eval_df = agent.evaluate(num_runs=10)
eval_df.to_excel(f"mappo_output_logs.xlsx", index=False)