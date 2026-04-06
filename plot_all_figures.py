import json
import os
import matplotlib.pyplot as plt
import numpy as np

def generate_academic_plots(json_filepath, output_dir="plots"):
    # 確保輸出資料夾存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 全局字體與參數設定
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    plt.rcParams['axes.linewidth'] = 1.0
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['grid.color'] = '#cccccc'

    # 定義 11 種演算法的配色盤
    color_map = {
        'Greedy': '#b3241b',   # 深紅
        'RW': '#db7b48',       # 橘色
        'GA': '#f2c85b',       # 黃色
        'MSA': '#3b7336',      # 墨綠
        'model P': '#6eb3df',  # 淺藍
        'DDTM': '#2a5b7d',     # 深藍
        'GAT-PPO': '#963592',  # 紫色 (主角標誌色)
        'MAPPO': '#e88bc1',    # 粉紅
        'VDN': '#8c564b',      # 棕色
        'QMIX': '#7f7f7f',     # 灰色
        'DGN': '#bcbd22'       # 橄欖綠
    }

    # 讀取實驗數據 JSON 檔案
    with open(json_filepath, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # 迴圈處理每一張圖
    for fig_data in config['figures']:
        saved_file_name = fig_data['saved_file_name']
        x_label = fig_data['horizontal_axis_title']
        y_label = fig_data['vertical_axis_title']
        x_ticks = np.array(fig_data['group_name'])
        data_dict = fig_data['data']

        print(f"正在繪製圖表: {saved_file_name} ...")

        # 建立畫布
        fig, ax = plt.subplots(figsize=(10, 5), dpi=300)

        # 動態計算柱子寬度與偏移量
        total_algs = len(data_dict)
        # 總共佔用空間的 85%，剩下的 15% 作為 group 之間間距
        total_width = 0.85 
        bar_width = total_width / total_algs
        offsets = np.linspace(-total_width/2 + bar_width/2, total_width/2 - bar_width/2, total_algs)

        # 依序畫出每個演算法的柱狀圖
        for i, (alg_name, alg_values) in enumerate(data_dict.items()):
            color = color_map.get(alg_name, '#333333') # 若找不到顏色預設用深灰
            ax.bar(x_ticks + offsets[i], alg_values, width=bar_width*0.9, 
                   label=alg_name, color=color, edgecolor='none', zorder=3)

        # 移除外框
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

        # 設定網格與坐標軸
        ax.xaxis.grid(False) 
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([str(n) for n in x_ticks], fontsize=14, fontweight='bold')
        ax.set_xlabel(x_label, fontsize=16, fontweight='bold')
        
        ax.set_ylabel(y_label, fontsize=16, fontweight='bold')
        ax.tick_params(axis='y', labelsize=14, length=0)
        ax.tick_params(axis='x', length=0, pad=10)

        # 設定圖例
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.05), ncol=6, 
                  frameon=False, fontsize=11, handlelength=1.2, handleheight=1.2)

        # 儲存與清理
        plt.tight_layout()
        output_path = os.path.join(output_dir, f"{saved_file_name}.png")
        plt.savefig(output_path, bbox_inches='tight')
        
        # 每畫完一張圖就 close
        plt.close(fig) 

    print(f"\n 所有圖表皆已匯出至 '{output_dir}/' 資料夾中。")

if __name__ == "__main__":
    generate_academic_plots("experiments_data.json", output_dir="figures")