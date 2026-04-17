import os
import json
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

def plot_learning_curves(json_path="../learning_curve/learning_curves.json", output_dir="../test"):
    if not os.path.exists(json_path):
        print(f"找不到 JSON 檔案: {json_path}")
        return

    os.makedirs(output_dir, exist_ok=True)

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if "figures" not in data:
        print("JSON 格式錯誤，找不到 'figures' 鍵值。")
        return

    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.titleweight'] = 'bold'

    print("開始解析 JSON 數據並重新繪製超清晰陰影圖...")

    for dict_key, fig_data in data["figures"].items():
        title = fig_data["title"]
        x_title = fig_data["x_name"]
        y_title = fig_data["y_name"]
        seed_data = fig_data["data"]

        algo_name = title.split(" Learning Curve")[0]
        seaborn_data_records = []

        for seed_str, rewards_list in seed_data.items():
            seed_val = int(seed_str)
            for ep, val in enumerate(rewards_list):
                seaborn_data_records.append({
                    x_title: ep,
                    y_title: val,
                    "Algorithm": algo_name,
                    "Seed": seed_val
                })

        df_map = pd.DataFrame(seaborn_data_records)

        plt.figure(figsize=(10, 6))
        sns.set_theme(style="whitegrid")
        
        ax = sns.lineplot(
            data=df_map, 
            x=x_title, 
            y=y_title, 
            hue="Algorithm", 
            errorbar='sd', 
            linewidth=3
        )
        
        # 座標軸文字
        plt.xlabel(x_title, fontsize=20, fontweight='black', labelpad=10)
        plt.ylabel(y_title, fontsize=20, fontweight='black', labelpad=10)
        
        # 刻度數字 (ticks)
        plt.xticks(fontsize=16, fontweight='bold')
        plt.yticks(fontsize=16, fontweight='bold')
        
        # 圖例 (legend)
        plt.setp(ax.get_legend().get_texts(), fontsize='16', fontweight='bold')
        plt.setp(ax.get_legend().get_title(), fontsize='18', fontweight='black')
        
        plt.tight_layout()
        
        img_filename = os.path.join(output_dir, f"{dict_key}_shaded.png")
        plt.savefig(img_filename, dpi=300)
        plt.close()
        print(f"  -> 已還原圖表並儲存至: {img_filename}")

    print("\n所有圖表皆已成功還原！")

if __name__ == "__main__":
    plot_learning_curves()