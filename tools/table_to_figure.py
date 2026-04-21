import json
import os
import matplotlib.pyplot as plt
import numpy as np
import math
from matplotlib.ticker import FuncFormatter

def draw_bar_chart(saved_file_name, x_label, y_label, x_ticks, data_dict, output_dir, color_map, use_log_scale=False):

    print(f"  -> 輸出圖表: {saved_file_name}.png ...")
    
    # 畫布設定
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
        raw_values = data_dict[alg_name]
        alg_values = [max(val, 1e0) if use_log_scale else val for val in raw_values]
        color = color_map.get(alg_name, '#333333') 
        ax.bar(x_ticks + offsets[i], alg_values, width=bar_width*0.95, 
               label=alg_name, color=color, edgecolor='none', zorder=3)

    if use_log_scale:
        ax.set_yscale('log')
        ax.set_ylim(1e0, 1e6)
        ax.set_yticks([1e0, 1e2, 1e4, 1e6])
        
        def custom_sci_fmt(x, pos):
            exp = int(math.log10(x))
            return f"1.E{exp:+03d}"
        
        ax.yaxis.set_major_formatter(FuncFormatter(custom_sci_fmt))
        ax.yaxis.set_minor_formatter(plt.NullFormatter())
        ax.yaxis.grid(True, which='minor', linestyle=':', alpha=0.4)

    # 移除外框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    ax.xaxis.grid(False) 
    ax.set_xticks(x_ticks)
    ax.set_xlim(min(x_ticks) - 0.6, max(x_ticks) + 0.6)
    
    # 高清晰度設定
    ax.set_xticklabels([str(n) for n in x_ticks], fontsize=22)
    ax.set_xlabel(x_label, fontsize=30, fontweight='bold', labelpad=8)
    ax.set_ylabel(y_label, fontsize=30, fontweight='bold', labelpad=8)
    
    # Y軸刻度數字
    ax.tick_params(axis='y', labelsize=22, length=4, width=1.5)
    ax.tick_params(axis='x', length=0, pad=8)
    plt.setp(ax.get_yticklabels())

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

    # 儲存與清理
    output_path = os.path.join(output_dir, f"{saved_file_name}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig) 


def convert_tables_to_figures(json_filepath, output_dir="../figures"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 字體優先使用 Cambria / Georgia
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Cambria', 'Georgia', 'Times New Roman'] + plt.rcParams['font.serif']
    plt.rcParams['axes.linewidth'] = 1.5
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['grid.color'] = '#cccccc'

    color_map = {
        'Greedy': '#b3241b', 'RW': '#db7b48', 'GA': '#f2c85b', 'MSA': '#3b7336',      
        'model P': '#6eb3df', 'Model P': '#6eb3df', 'DDTM': '#2a5b7d',     
        'GAT-PPO': '#963592', 'MAPPO': '#e88bc1', 'VDN': '#8c564b',      
        'QMIX': '#7f7f7f', 'DGN': '#bcbd22'       
    }

    with open(json_filepath, 'r', encoding='utf-8') as f:
        config = json.load(f)

    print("\n 開始將 Tables 資料轉換為長條圖 (針對數據優化 Log Scale)")
    
    if 'tables' not in config:
        print("錯誤：JSON 檔案中找不到 'tables' 標籤。")
        return

    for table_data in config['tables']:
        base_filename = table_data['saved_file_name']
        col_names = table_data['column_name']
        x_ticks = np.array(table_data['row_name'])
        
        raw_title = table_data['row_title']
        x_label = raw_title.replace('$', '').replace('\\#', 'Number').strip()
        y_label = "Time (s)" 
        table_dict = table_data['data']
        
        print(f"\n正在處理表格來源: {base_filename}")

        for col_idx, col_name in enumerate(col_names):
            chart_data_dict = {}
            for alg_name, matrix in table_dict.items():
                try:
                    chart_data_dict[alg_name] = [row[col_idx] for row in matrix]
                except IndexError:
                    chart_data_dict[alg_name] = [1e0] * len(x_ticks)

            safe_col_name = col_name.replace(" ", "_").replace("(", "").replace(")", "")
            
            if len(col_names) > 1:
                new_filename = f"{base_filename}_{safe_col_name}"
            else:
                new_filename = base_filename

            draw_bar_chart(new_filename, x_label, y_label, x_ticks, chart_data_dict, output_dir, color_map, use_log_scale=True)

    print(f"\n 所有 Table 轉換圖表皆已匯出至 '{output_dir}/' 資料夾中")

if __name__ == "__main__":
    convert_tables_to_figures("../experiments_data.json", output_dir="../figures")