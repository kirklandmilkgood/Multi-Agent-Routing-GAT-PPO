import json
import os
import matplotlib.pyplot as plt
import numpy as np
import math

def generate_academic_plots(json_filepath, output_dir="../figures"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 優先使用 Cambria 或 Georgia 字體，退而求其次才是 Times New Roman
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Cambria', 'Georgia', 'Times New Roman'] + plt.rcParams['font.serif']
    plt.rcParams['axes.linewidth'] = 1.5
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

        # 畫布比例
        fig, ax = plt.subplots(figsize=(9, 4.5), dpi=300)

        # 強制把 GAT-PPO 排到陣列最後面
        alg_names = list(data_dict.keys())
        target_alg = None
        for name in alg_names:
            if name.lower() == 'gat-ppo':
                target_alg = name
                break
        
        if target_alg:
            alg_names.remove(target_alg)
            alg_names.append(target_alg)

        # 柱子設定
        total_algs = len(alg_names)
        total_width = 0.85
        bar_width = total_width / total_algs
        offsets = np.linspace(-total_width/2 + bar_width/2, total_width/2 - bar_width/2, total_algs)

        for i, alg_name in enumerate(alg_names):
            alg_values = data_dict[alg_name]
            color = color_map.get(alg_name, '#333333') 
            ax.bar(x_ticks + offsets[i], alg_values, width=bar_width*0.95, 
                   label=alg_name, color=color, edgecolor='none', zorder=3)

        # 移除外框
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

        ax.xaxis.grid(False) 
        ax.set_xticks(x_ticks)
        ax.set_xlim(min(x_ticks) - 0.6, max(x_ticks) + 0.6)
        
        ax.set_xticklabels([str(n) for n in x_ticks], fontsize=22)
        ax.set_xlabel(x_label, fontsize=30, fontweight='bold', labelpad=8)
        ax.set_ylabel(y_label, fontsize=30, fontweight='bold', labelpad=8)
        
        ax.tick_params(axis='y', labelsize=22, length=4, width=1.5)
        ax.tick_params(axis='x', length=0, pad=8)

        # 圖例排版邏輯
        handles, labels = ax.get_legend_handles_labels()
        ncol = 6
        nrow = math.ceil(len(handles) / ncol)
        
        handles_reordered = []
        labels_reordered = []
        for c in range(ncol):
            for r in range(nrow):
                idx = r * ncol + c
                if idx < len(handles):
                    handles_reordered.append(handles[idx])
                    labels_reordered.append(labels[idx])

        ax.legend(handles_reordered, labels_reordered, 
                  loc='lower center', bbox_to_anchor=(0.5, 1.05), ncol=ncol, 
                  frameon=False, 
                  handlelength=0.3,
                  handleheight=0.3,    
                  handletextpad=0.2,
                  columnspacing=0.4,   
                  prop={'size': 24})
        
        output_path = os.path.join(output_dir, f"{saved_file_name}.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.05)
        plt.close(fig) 

    print(f"\n 所有圖表皆已匯出至 '{output_dir}/' 資料夾中。")

if __name__ == "__main__":
    generate_academic_plots("../experiments_data.json", output_dir="../figures")