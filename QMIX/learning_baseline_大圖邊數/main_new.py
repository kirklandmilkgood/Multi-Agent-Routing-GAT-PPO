import os
import sys
import json
import torch
import pandas as pd
from qmix_agent import QMIXAgent  # qmix adapter
from gat_env import MultiAgentTSPEnv, load_graph_from_excel

# 接收總指揮腳本傳來的 config 路徑，若無則預設抓取上層目錄的 config.json
config_path = sys.argv[1] if len(sys.argv) > 1 else "../../large_network_config.json"
    
if not os.path.exists(config_path):
        print(f"找不到設定檔: {config_path}")
        sys.exit(1)

with open(config_path, 'r', encoding='utf-8') as f:
        configs = json.load(f)["experiments"]

total_exps = len(configs)
for i in range(total_exps):
    num_nodes = configs[i]["num_nodes"]
    num_edges = configs[i]["num_edges"]
    num_agents = configs[i]["num_agents"]
    t_budget = configs[i]["total_budget"]
    i_budget = configs[i]["individual_budget"]
    dataset_path = configs[i]["dataset"]
    num_episodes = configs[i]["episodes"]
    print(f"experiment setting: num nodes: {num_nodes}, num edges: {num_edges}, num agents: {num_agents}, total budget: {t_budget}, individual budget: {i_budget}...")
    eval_file_path = dataset_path

    # 啟動訓練環境
    train_env = MultiAgentTSPEnv(num_nodes=num_nodes, num_agents=num_agents, total_budget=t_budget, individual_budget=i_budget)

    agent = QMIXAgent(train_env, hidden_dim=64, buffer_capacity=10000, batch_size=32)

    # 訓練模型
    train_log = agent.train(num_episodes=num_episodes)

    # 執行評估
    eval_env = MultiAgentTSPEnv(num_nodes=num_nodes, num_agents=num_agents, total_budget=t_budget, individual_budget=i_budget)
    eval_env.fixed_graph = load_graph_from_excel(eval_file_path)
    agent.env = eval_env 

    eval_df = agent.evaluate(num_runs=10)
    eval_df.to_excel(f"qmix_output_logs_{num_agents}.xlsx", index=False)