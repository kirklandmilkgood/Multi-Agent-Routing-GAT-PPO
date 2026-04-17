import os
import torch
import pandas as pd
from Ppo_With_Gat import PPOGATAgent
from gat_env import MultiAgentTSPEnv, load_graph_from_excel

# === 參數設定 ===
checkpoint_dir = "ppo_checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)
checkpoint_path = os.path.join(checkpoint_dir, "gat_ppo_checkpoint_ep50.pt")
output_log_path = "ppo_output_gat_logs.xlsx"
num_training_episodes = 50
num_evaluation_runs = 10
eval_file_path = "../../dataset/ER_Graph_200Nodes.xlsx"  # 評估圖檔路徑

# === 啟動訓練環境（使用隨機 ER 圖） ===
train_env = MultiAgentTSPEnv(num_nodes=200, num_agents=4, total_budget=300, individual_budget=200)
agent = PPOGATAgent(train_env)

# === 訓練 PPO 模型 ===
print("開始訓練 PPO GAT 模型...")
train_log = agent.train(num_episodes=num_training_episodes)
torch.save(agent.model.state_dict(), checkpoint_path)
print(f"訓練完成，模型已儲存至 {checkpoint_path}")

# === 載入模型並建立評估環境 ===
agent.load_model(checkpoint_path)

# 使用相同設定建立一個空白環境
eval_env = MultiAgentTSPEnv(num_nodes=200, num_agents=4, total_budget=300, individual_budget=200)
# 載入固定評估圖
eval_env.fixed_graph = load_graph_from_excel(eval_file_path)
agent.env = eval_env  # 替換為評估環境

# === 執行評估 ===
eval_df = agent.evaluate(num_runs=num_evaluation_runs)
eval_df.to_excel(output_log_path, index=False)
print(f"訓練與評估記錄已儲存至 {output_log_path}")
