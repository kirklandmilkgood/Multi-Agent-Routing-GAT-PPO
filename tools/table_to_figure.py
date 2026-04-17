import json
import os
import matplotlib.pyplot as plt
import numpy as np
import math

def draw_bar_chart(saved_file_name, x_label, y_label, x_ticks, data_dict, output_dir, color_map, use_log_scale=False):
    """
    核心長條圖繪製函數
    加入 use_log_scale 參數以支援對數座標軸
    """
    print(f"  -> 輸出圖表: {saved_file_name}.png ...")
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)

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

    # 動態計算柱子寬度與偏移量
    total_algs = len(alg_names)
    total_width = 0.92 
    bar_width = total_width / total_algs
    offsets = np.linspace(-total_width/2 + bar_width/2, total_width/2 - bar_width/2, total_algs)

    # 依照排好的 alg_names 依序畫圖
    for i, alg_name in enumerate(alg_names):
        alg_values = data_dict[alg_name]
        color = color_map.get(alg_name, '#333333') 
        
        # 為了避免 log scale 對 0 值報錯，如果數值為 0 或小於 0，給一個極小值 (如 0.01)
        # 不過訓練時間通常大於 0，這裡直接畫即可
        ax.bar(x_ticks + offsets[i], alg_values, width=bar_width*0.98, 
               label=alg_name, color=color, edgecolor='none', zorder=3)

    # 設定 Y 軸為對數座標 (log scale)
    if use_log_scale:
        ax.set_yscale('log')
        ax.yaxis.grid(True, which='minor', linestyle=':', alpha=0.4)

    # 移除外框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    # 設定網格與坐標軸
    ax.xaxis.grid(False) 
    ax.set_xticks(x_ticks)
    
    # 字體設定
    ax.set_xticklabels([str(n) for n in x_ticks], fontsize=18, fontweight='bold')
    ax.set_xlabel(x_label, fontsize=22, fontweight='bold')
    ax.set_ylabel(y_label, fontsize=22, fontweight='bold')
    
    ax.tick_params(axis='y', labelsize=18, length=0)
    ax.tick_params(axis='x', length=0, pad=10)

    # 將 matplotlib 的「直向填充」改為「橫向填充」
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
              frameon=False, handlelength=1.2, handleheight=1.2,
              prop={'size': 16, 'weight': 'bold'})

    # 儲存與清理
    plt.tight_layout()
    output_path = os.path.join(output_dir, f"{saved_file_name}.png")
    plt.savefig(output_path, bbox_inches='tight')
    plt.close(fig) 


def convert_tables_to_figures(json_filepath, output_dir="../figures"):
    """
    獨立讀取 Tables 資料並轉換為 Figures 的主程式
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 全局字體與參數設定
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    plt.rcParams['axes.linewidth'] = 1.0
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
        'QMIX': "#5c997f",     
        'DGN': '#bcbd22'       
    }

    with open(json_filepath, 'r', encoding='utf-8') as f:
        config = json.load(f)

    print("\n=== 開始將 Tables 資料轉換為長條圖 (使用 Log Scale) ===")
    
    if 'tables' not in config:
        print("錯誤：JSON 檔案中找不到 'tables' 標籤。")
        return

    for table_data in config['tables']:
        base_filename = table_data['saved_file_name']
        col_names = table_data['column_name']
        x_ticks = np.array(table_data['row_name'])
        
        # 處理 X 軸與 Y 軸標籤
        raw_title = table_data['row_title']
        x_label = raw_title.replace('$', '').replace('\\#', 'Number').strip()
        
        y_label = "Training Time (s) (Log Scale)" 
        
        table_dict = table_data['data']
        
        print(f"\n正在處理表格來源: {base_filename}")

        # 遍歷每一個 column
        for col_idx, col_name in enumerate(col_names):
            chart_data_dict = {}
            
            # 抽取該 column 的資料
            for alg_name, matrix in table_dict.items():
                try:
                    chart_data_dict[alg_name] = [row[col_idx] for row in matrix]
                except IndexError:
                    # log scale 下如果數值為 0 畫圖會消失，給一個 0.1 讓圖表不會 collapse
                    chart_data_dict[alg_name] = [0.1] * len(x_ticks)

            safe_col_name = col_name.replace(" ", "_").replace("(", "").replace(")", "")
            
            if len(col_names) > 1:
                new_filename = f"{base_filename}_{safe_col_name}"
            else:
                new_filename = base_filename

            # 核心繪圖函數，開啟 use_log_scale=True
            draw_bar_chart(new_filename, x_label, y_label, x_ticks, chart_data_dict, output_dir, color_map, use_log_scale=True)

    print(f"\n=== 所有 Table 轉換圖表皆已匯出至 '{output_dir}/' 資料夾中 ===")


if __name__ == "__main__":
    # 執行轉換程式
    convert_tables_to_figures("../experiments_data.json", output_dir="../figures")