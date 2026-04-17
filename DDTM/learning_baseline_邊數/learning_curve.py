import os
import sys
import json
import argparse
import torch
import numpy as np
import random
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns 
from train_ddtm import train
from evaluate_ddtm import evaluate

def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def main():
    parser = argparse.ArgumentParser(description="Run Learning Curve Experiment")
    parser.add_argument("--config", type=str, required=True, help="Path to the JSON config file")
    parser.add_argument("--seeds", type=int, nargs='+', required=True, help="List of random seeds")
    parser.add_argument("--algo", type=str, default="GAT-PPO", help="Algorithm name for logging")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"找不到設定檔: {args.config}")
        sys.exit(1)

    print(f"[{args.algo}] 啟動學習曲線實驗 | Seeds: {args.seeds}")

    with open(args.config, 'r', encoding='utf-8') as f:
        config_data = json.load(f)
        configs = config_data["experiments"]

    output_dir = "../../learning_curve"
    os.makedirs(output_dir, exist_ok=True)
    
    # global json 檔案讀寫機制
    # 設定唯一的 global json 檔名
    global_json_path = os.path.join(output_dir, "learning_curves.json")
    
    # 若檔案已經存在，先讀取進來；否則建立一個新的空結構
    if os.path.exists(global_json_path):
        with open(global_json_path, "r", encoding="utf-8") as f:
            try:
                global_results = json.load(f)
            except json.JSONDecodeError:
                global_results = {"figures": {}}
    else:
        global_results = {"figures": {}}
        
    # 確保 figures key 存在
    if "figures" not in global_results:
        global_results["figures"] = {}

    for i in range(len(configs)):
        num_nodes = configs[i]["num_nodes"]
        num_edges = configs[i]["num_edges"]
        num_agents = configs[i]["num_agents"]
        t_budget = configs[i]["total_budget"]
        i_budget = configs[i]["individual_budget"]
        num_episodes = configs[i]["episodes"]
        x_title = configs[i].get("horizontal_axis_title", "Episode")
        y_title = configs[i].get("vertical_axis_title", "Objective Value")
        
        map_name = "Taichung" if num_nodes > 10000 else "Minnesota"
        print(f"\n========== 正在執行 {map_name} 網路 | Episodes: {num_episodes} ==========")
        
        seaborn_data_records = []
        seed_raw_data = {} # 存放這個 map 下所有 seed 的純數據

        for seed in args.seeds:
            print(f"\n--- 啟動 Seed: {seed} ---")
            set_global_seed(seed)
            train_df = train(num_agents=num_agents, total_budget=t_budget, per_agent_budget=i_budget, n_nodes=num_nodes, n_edges=num_edges, n_episodes=num_episodes)           
            # 從 DataFrame 提取數值轉成 list
            if isinstance(train_df, pd.DataFrame):
                rewards_list = train_df["Total Reward"].tolist()
            else:
                rewards_list = train_df 
            
            # 存入字典供 json 使用
            seed_raw_data[str(seed)] = rewards_list

            # 給 seaborn 畫圖用的格式
            for ep, val in enumerate(rewards_list):
                seaborn_data_records.append({
                    x_title: ep,
                    y_title: val,
                    "Algorithm": args.algo,
                    "Seed": seed
                })

        # 將該 (algo + map) 的資訊打包並存入 dict
        dict_key = f"{args.algo}_{map_name}"
        global_results["figures"][dict_key] = {
            "title": f"{args.algo} Learning Curve ({map_name})",
            "x_name": x_title,
            "y_name": y_title,
            "data": seed_raw_data
        }

        # 即時畫圖輸出
        df_map = pd.DataFrame(seaborn_data_records)
        plt.figure(figsize=(10, 6))
        sns.set_theme(style="whitegrid")
        
        sns.lineplot(
            data=df_map, 
            x=x_title, 
            y=y_title, 
            hue="Algorithm", 
            errorbar='sd', 
            linewidth=2
        )
        
        plt.title(global_results["figures"][dict_key]["title"], fontsize=14, fontweight='bold')
        plt.xlabel(x_title, fontsize=12)
        plt.ylabel(y_title, fontsize=12)
        plt.tight_layout()
        
        img_filename = os.path.join(output_dir, f"{dict_key}_shaded.png")
        plt.savefig(img_filename, dpi=300)
        plt.close()
        print(f"[{args.algo}] {map_name} 的陰影圖已儲存至: {img_filename}")

    # 每個演算法執行完，就把最新的 dict 存回同一個檔案
    with open(global_json_path, "w", encoding="utf-8") as f:
        json.dump(global_results, f, indent=4, ensure_ascii=False)
        
    print(f"\n[{args.algo}] 訓練結束。數據已成功更新至總檔: {global_json_path}")

if __name__ == "__main__":
    main()