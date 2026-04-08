import json
import os
import matplotlib.pyplot as plt
import numpy as np

def generate_academic_plots(json_filepath, output_dir="plots"):
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
        'Greedy': '#b3241b',   
        'RW': '#db7b48',       
        'GA': '#f2c85b',       
        'MSA': '#3b7336',      
        'model P': '#6eb3df',  
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

        # 動態計算柱子寬度與偏移量
        total_algs = len(data_dict)
        
        total_width = 0.92 
        bar_width = total_width / total_algs
        offsets = np.linspace(-total_width/2 + bar_width/2, total_width/2 - bar_width/2, total_algs)

        # 依序畫出每個演算法的柱狀圖
        for i, (alg_name, alg_values) in enumerate(data_dict.items()):
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
        
        # 保持放大的字體設定
        ax.set_xticklabels([str(n) for n in x_ticks], fontsize=18, fontweight='bold')
        ax.set_xlabel(x_label, fontsize=22, fontweight='bold')
        ax.set_ylabel(y_label, fontsize=22, fontweight='bold')
        
        ax.tick_params(axis='y', labelsize=18, length=0)
        ax.tick_params(axis='x', length=0, pad=10)

        # 圖例位置設定
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.05), ncol=6, 
                  frameon=False, handlelength=1.2, handleheight=1.2,
                  prop={'size': 16, 'weight': 'bold'})

        # 儲存與清理
        plt.tight_layout()
        output_path = os.path.join(output_dir, f"{saved_file_name}.png")
        plt.savefig(output_path, bbox_inches='tight')
        
        plt.close(fig) 

    print(f"\n所有圖表皆已匯出至 '{output_dir}/' 資料夾中。")

if __name__ == "__main__":
    generate_academic_plots("experiments_data.json", output_dir="figures")