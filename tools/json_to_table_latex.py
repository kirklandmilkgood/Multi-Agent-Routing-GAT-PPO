import json
import os

def generate_latex_from_json(json_filepath, output_dir="../tables"):
    # 確認輸出資料夾是否存在，若無則自動建立
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"已建立資料夾: {output_dir}/")

    # 讀取 json 檔案
    with open(json_filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    for table in data["tables"]:
        filename = table["saved_file_name"]
        caption_text = table["caption"]
        col_groups = table["column_name"]
        rows = table["row_name"]
        row_title = table["row_title"]
        data_dict = table["data"]
        
        # 動態獲取演算法名單與維度
        algos = list(data_dict.keys())
        num_col_groups = len(col_groups)
        num_algos = len(algos)
        total_data_cols = num_col_groups * num_algos
        
        # 建立欄位格式 (例如: |c|cccccc|cccccc|...)
        col_format = "|c|" + "|".join(["c" * num_algos for _ in range(num_col_groups)]) + "|"
        
        latex = []
        # table* 讓表格跨越雙欄
        latex.append("\\begin{table}[H]")
        latex.append("\\centering")
        
        # 縮小欄位間的空白 (預設是 6pt，改成 3pt 或 4pt)
        # 把空間還給文字，讓 \resizebox 縮放時文字會更大
        latex.append("\\setlength{\\tabcolsep}{4pt}")
        
        # 處理多欄位超寬表格，自動等比例縮放至頁面寬度
        latex.append("\\resizebox{\\textwidth}{!}{")
        latex.append(f"\\begin{{tabular}}{{{col_format}}}")
        latex.append("\\hline")
        
        # 建立第一層標題 (Dataset 名稱)
        header1 = f"\\multirow{{2}}{{*}}{{\\textbf{{{row_title}}}}}"
        for cg in col_groups:
            safe_cg = cg.replace("_", "\\_")
            header1 += f" & \\multicolumn{{{num_algos}}}{{c|}}{{\\textbf{{{safe_cg}}}}}"
        header1 += f" \\\\ \\cline{{2-{total_data_cols + 1}}}"
        latex.append(header1)
        
        # 建立第二層標題 (演算法名稱)
        header2 = ""
        for _ in range(num_col_groups):
            for algo in algos:
                safe_algo = algo.replace("_", "\\_")
                header2 += f" & \\textbf{{{safe_algo}}}"
        header2 += " \\\\ \\hline"
        latex.append(header2)
        
        # 動態填入數據
        for i, row_val in enumerate(rows):
            row_str = f"{row_val}"
            for j in range(num_col_groups):
                for algo in algos:
                    try:
                        val = data_dict[algo][i][j]
                        row_str += f" & {val:.2f}"
                    except IndexError:
                        row_str += " & -"
            row_str += " \\\\"
            latex.append(row_str)
            
        latex.append("\\hline")
        latex.append("\\end{tabular}")
        latex.append("}")
        
        # 加入 json 中指定的 caption 與自動 label
        latex.append(f"\\caption*{{{caption_text}}}")
        latex.append(f"\\label{{tab:{filename}}}")
        
        latex.append("\\end{table}")
        
        # 將結果寫入 .tex 檔案
        output_filepath = os.path.join(output_dir, f"{filename}.tex")
        with open(output_filepath, 'w', encoding='utf-8') as out_f:
            out_f.write("\n".join(latex) + "\n")
            
        print(f"生成 LaTeX 表格並儲存至: {output_filepath}")

if __name__ == "__main__":
    generate_latex_from_json("../experiments_data.json")