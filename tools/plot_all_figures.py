import json
import os
import matplotlib.pyplot as plt
import numpy as np
import math

def generate_academic_plots(json_filepath, output_dir="../figures"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 全局字體與參數設定
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.titleweight'] = 'bold'
    plt.rcParams['axes.linewidth'] = 1.5  # 外框線條稍微加粗
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['grid.color'] = '#cccccc'

    # 定義演算法的配色盤
    color_map = {
        'Greedy': '#b3241b',   
        'RW': '#db7b48',       
        'GA': '#f2c85b',       
        'MSA': '#3b7336',      
        'model P': '#6eb3df',
        'Model P': '#6eb3df',
        'DDTM': '#2a5b7d',     
        'GAT-PPO': '#963592',  
        'MAPPO': '#e88bc1',    
        'VDN': '#8c564b',      
        'QMIX': '#7f7f7f',     
        'DGN': '#bcbd22'       
    }

    with open(json_filepath, 'r', encoding='utf-8') as f:
        config = json.load(f)

    for fig_data in config['figures']:
        saved_file_name = fig_data['saved_file_name']
        x_label = fig_data['horizontal_axis_title']
        y_label = fig_data['vertical_axis_title']
        x_ticks = np.array(fig_data['group_name'])
        data_dict = fig_data['data']

        print(f"正在繪製圖表: {saved_file_name} ...")

        fig, ax = plt.subplots(figsize=(12, 6), dpi=300)

        # 強制把 GAT-PPO 排到陣列最後面
        alg_names = list(data_dict.keys())
        target_alg = None
        
        # 尋找 GAT-PPO
        for name in alg_names:
            if name.lower() == 'gat-ppo':
                target_alg = name
                break
        
        # 如果有找到，就先移除它，再把它加到最後面 (確保柱子畫在最右邊)
        if target_alg:
            alg_names.remove(target_alg)
            alg_names.append(target_alg)

        # 動態計算柱子寬度與偏移量
        total_algs = len(alg_names)
        total_width = 0.92 
        bar_width = total_width / total_algs
        offsets = np.linspace(-total_width/2 + bar_width/2, total_width/2 - bar_width/2, total_algs)

        # 依照排好的 alg_names 依序畫圖
        for i, alg_name in enumerate(alg_names):
            alg_values = data_dict[alg_name]
            color = color_map.get(alg_name, '#333333') 
            
            ax.bar(x_ticks + offsets[i], alg_values, width=bar_width*0.98, 
                   label=alg_name, color=color, edgecolor='none', zorder=3)

        # 移除外框
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

        # 設定網格與坐標軸
        ax.xaxis.grid(False) 
        ax.set_xticks(x_ticks)
        
        # 放大的字體設定
        ax.set_xticklabels([str(n) for n in x_ticks], fontsize=20, fontweight='black')
        ax.set_xlabel(x_label, fontsize=24, fontweight='black', labelpad=10)
        ax.set_ylabel(y_label, fontsize=24, fontweight='black', labelpad=10)
        
        ax.tick_params(axis='y', labelsize=20, length=0)
        ax.tick_params(axis='x', length=0, pad=10)
        
        # 強制 Y 軸刻度文字加粗
        plt.setp(ax.get_yticklabels(), fontweight='bold')

        # 將 matplotlib 的「直向填充」改為「橫向填充」
        handles, labels = ax.get_legend_handles_labels()
        ncol = 6
        
        # 計算總共需要幾列 (如 11 個演算法 / 6 = 2 列)
        nrow = math.ceil(len(handles) / ncol)
        
        handles_reordered = []
        labels_reordered = []
        
        # 透過矩陣轉置的數學邏輯，預先將陣列洗牌
        for c in range(ncol):
            for r in range(nrow):
                idx = r * ncol + c
                if idx < len(handles):
                    handles_reordered.append(handles[idx])
                    labels_reordered.append(labels[idx])

        # 傳入洗牌後的結果
        ax.legend(handles_reordered, labels_reordered, 
                  loc='lower center', bbox_to_anchor=(0.5, 1.05), ncol=ncol, 
                  frameon=False, handlelength=1.2, handleheight=1.2,
                  prop={'size': 18, 'weight': 'bold'})

        # 儲存與清理
        plt.tight_layout()
        output_path = os.path.join(output_dir, f"{saved_file_name}.png")
        plt.savefig(output_path, bbox_inches='tight')
        
        plt.close(fig) 

    print(f"\n所有圖表皆已匯出至 '{output_dir}/' 資料夾中。")

if __name__ == "__main__":
    generate_academic_plots("../experiments_data.json", output_dir="../figures")