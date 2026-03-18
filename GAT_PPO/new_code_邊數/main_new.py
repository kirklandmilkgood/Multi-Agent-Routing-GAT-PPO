
import os
import torch
import pandas as pd
from Ppo_With_Gat import PPOGATAgent
from gat_env import MultiAgentTSPEnv

# === 訓練與評估參數設定 ===
checkpoint_dir = "ppo_checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)
checkpoint_path = os.path.join(checkpoint_dir, "gat_ppo_checkpoint_ep100.pt")
output_log_path = "ppo_output_gat_logs.xlsx"
num_training_episodes = 100
num_evaluation_runs = 10

# === 圖參數設定 ===
train_num_nodes = 2636
train_num_edges = 8500
eval_file_path = "../../dataset/road-minnesota.xlsx"

# === 建立訓練環境（使用 ER 圖 with edge count） ===
train_env = MultiAgentTSPEnv(
    num_nodes=train_num_nodes,
    num_edges=train_num_edges,
    num_agents=4,
    total_budget=200,
    individual_budget=150
)
agent = PPOGATAgent(train_env)

# === 訓練 PPO GAT 模型 ===
print("開始訓練 PPO GAT 模型...")
train_log = agent.train(num_episodes=num_training_episodes)
torch.save(agent.model.state_dict(), checkpoint_path)
print(f"訓練完成，模型已儲存至 {checkpoint_path}")

# === 載入模型並建立評估環境（使用外部圖） ===
agent.load_model(checkpoint_path)
eval_env = MultiAgentTSPEnv(
    num_agents=4,
    total_budget=200,
    individual_budget=150,
    eval_file_path=eval_file_path
)
agent.env = eval_env  # 替換 agent 的環境為評估環境

# === 執行評估 ===
eval_df = agent.evaluate(num_runs=num_evaluation_runs)
eval_df.to_excel(output_log_path, index=False)
print(f"訓練與評估記錄已儲存至 {output_log_path}")
