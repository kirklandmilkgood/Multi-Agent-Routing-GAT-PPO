import os
import sys
import json
import torch
import pandas as pd
from train_ddtm import train
from evaluate_ddtm import evaluate

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
    is_dynamic = True if configs[i]["dynamic"] else False
    print(f"experiment setting: num nodes: {num_nodes}, num edges: {num_edges}, num agents: {num_agents}, total budget: {t_budget}, individual budget: {i_budget}...")
    eval_file_path = dataset_path
    train(num_agents=num_agents, total_budget=t_budget, per_agent_budget=i_budget, n_nodes=num_nodes, n_edges=num_edges, n_episodes=num_episodes, is_dynamic=is_dynamic)
    evaluate(num_agents=num_agents, total_budget=t_budget, per_agent_budget=i_budget, dataset=dataset_path, is_dynamic=is_dynamic)