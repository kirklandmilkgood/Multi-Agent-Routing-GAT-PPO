import sys
import os
import torch
import json
import pandas as pd
from Ppo_With_Gat import PPOGATAgent
from gat_env import MultiAgentTSPEnv


# 接收總指揮腳本傳來的 config 路徑，若無則預設抓取上層目錄的 config.json
config_path = sys.argv[1] if len(sys.argv) > 1 else "../../large_network_config.json"
    
if not os.path.exists(config_path):
        print(f"找不到設定檔: {config_path}")
        sys.exit(1)

with open(config_path, 'r', encoding='utf-8') as f:
        configs = json.load(f)["experiments"]

total_exps = len(configs)
# === 訓練與評估參數設定 ===
checkpoint_dir = "ppo_checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)
checkpoint_path = os.path.join(checkpoint_dir, "gat_ppo_checkpoint_ep100.pt")
num_evaluation_runs = 10

# === 圖參數設定 ===
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
    output_log_path = f"ppo_output_gat_logs_{num_agents}.xlsx"
    # === 建立訓練環境 ===
    train_env = MultiAgentTSPEnv(
        num_nodes=num_nodes,
        num_edges=num_edges,
        num_agents=num_agents,
        total_budget=t_budget,
        individual_budget=i_budget,
        dynamic_traffic=is_dynamic
    )
    agent = PPOGATAgent(train_env)

    # === 訓練 PPO GAT 模型 ===
    print("開始訓練 PPO GAT 模型...")
    train_log = agent.train(num_episodes=num_episodes)
    torch.save(agent.model.state_dict(), checkpoint_path)
    print(f"訓練完成，模型已儲存至 {checkpoint_path}")

    # === 載入模型並建立評估環境（使用外部圖） ===
    agent.load_model(checkpoint_path)
    eval_env = MultiAgentTSPEnv(
        num_agents=num_agents,
        total_budget=t_budget,
        individual_budget=i_budget,
        eval_file_path=eval_file_path,
        dynamic_traffic=is_dynamic
    )
    agent.env = eval_env  # 替換 agent 的環境為評估環境

    # === 執行評估 ===
    eval_df = agent.evaluate(num_runs=num_evaluation_runs)
    eval_df.to_excel(output_log_path, index=False)
    print(f"訓練與評估記錄已儲存至 {output_log_path}")
