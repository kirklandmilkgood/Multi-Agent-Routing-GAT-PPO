import subprocess
import os
import sys
import datetime

# 取得絕對路徑的 config 檔案
root_dir = os.path.dirname(os.path.abspath(__file__))
er_config_path =os.path.join(root_dir, "er_graph_config.json")
if not os.path.exists(er_config_path):
    print(f"找不到全域設定檔: {er_config_path}")
    sys.exit(1)
euro_config_path = os.path.join(root_dir, "euroroad_config.json")
if not os.path.exists(euro_config_path):
    print(f"找不到全域設定檔: {euro_config_path}")
    sys.exit(1)
minnesota_config_path = os.path.join(root_dir, "minnesota_config.json")
if not os.path.exists(minnesota_config_path):
    print(f"找不到全域設定檔: {minnesota_config_path}")
    sys.exit(1)
large_network_config_path = os.path.join(root_dir, "large_network_config.json")
if not os.path.exists(large_network_config_path):
    print(f"找不到全域設定檔: {large_network_config_path}")
    sys.exit(1)
large_network_dynamic_config_path = os.path.join(root_dir, "large_network_dynamic_config.json")
if not os.path.exists(large_network_config_path):
    print(f"找不到全域設定檔: {large_network_dynamic_config_path}")
    sys.exit(1)
taichung_lc_config_path = os.path.join(root_dir, "learning_curve_taichung_config.json")
if not os.path.exists(taichung_lc_config_path):
    print(f"找不到全域設定檔: {taichung_lc_config_path}")
    sys.exit(1)
minnesota_lc_config_path = os.path.join(root_dir, "learning_curve_minnesota_config.json")
if not os.path.exists(minnesota_lc_config_path):
    print(f"找不到全域設定檔: {minnesota_lc_config_path}")
    sys.exit(1)

# 定義 learning curve 隨機種子
learning_curve_seeds = [42, 100, 123]
# 定義想要依序跑的演算法資料夾名稱
# 可隨時註解掉不想跑的演算法
nn_algorithms = [
    "VDN",
    "DGN",
    "MAPPO",
    "GAT_PPO",
    # "DDTM",
    "QMIX"
]
algorithms = [
    # "greedy",
    # "RW",
    # "GA",
    # "MSA",
    # "Model_P"
]

branch_map = [
    ("learning_baseline_邊數", minnesota_lc_config_path),
    ("learning_baseline_大圖邊數", taichung_lc_config_path),
    # ("learning_baseline_連邊機率", er_config_path),
    # ("learning_baseline_連邊機率", euro_config_path),
    # ("learning_baseline_邊數", minnesota_config_path),
    # ("learning_baseline_大圖邊數", large_network_config_path),
    # ("learning_baseline_大圖邊數", large_network_dynamic_config_path)
]

# 建立 timestamp，用來命名 Log 檔，避免每次執行覆蓋掉舊紀錄
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_file_path = os.path.join(root_dir, f"experiment_{timestamp}.log")

print("開始執行全自動化多模型實驗評估\n")
print(f"所有終端機輸出將同步記錄於: {log_file_path}\n")

# 開啟 Log 檔案準備寫入 (使用 "a" 附加模式)
with open(log_file_path, "a", encoding="utf-8") as log_file:
    
    # 定義一個工具函數，同時印在螢幕並寫入檔案
    def log_and_print(message):
        print(message, end="")
        log_file.write(message)
        log_file.flush() # 強制立刻寫入硬碟，防止當機遺失
    
    def algo_workflow(is_nn_network=False, branch="", config_path=""):
        algorithms_arr = nn_algorithms if is_nn_network else algorithms 
        for algo in algorithms_arr:
            algo_dir = os.path.join(root_dir, algo) if is_nn_network else root_dir
            branch_dir = os.path.join(algo_dir, branch) if is_nn_network else ""
            script_path = os.path.join(branch_dir, "main_new.py") if is_nn_network else os.path.join(root_dir, f"{algo}.py")

            if not os.path.exists(script_path):
                log_and_print(f"\n找不到腳本 {script_path}，跳過 {algo.upper()} 演算法...\n")
                continue

            log_and_print(f"\n[{algo.upper()}] --------------------------------------------------\n")
            log_and_print(f"\n正在啟動 {algo.upper()} 的訓練與評估任務...\n")
            
            try:
                # 環境變數字典，強制設定 Python 的 I/O 編碼為 utf-8
                my_env = os.environ.copy()
                my_env["PYTHONIOENCODING"] = "utf-8"
                # 使用 Popen 取代 run，這樣能即時攔截輸出
                # stderr=subprocess.STDOUT 代表把錯誤也混入標準輸出，一起存進 Log
                workplace = branch_dir if is_nn_network else root_dir
                target_file = "main_new.py" if is_nn_network else f"{algo}.py"
                process = subprocess.Popen(
                    [sys.executable,"-X", "utf8", "-u", target_file, config_path],
                    cwd=workplace,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace", # 遇到亂碼自動替換，防止腳本當機
                    env=my_env  # 將設定好的環境變數傳給子程式
                )

                # 即時逐行讀取子程式的輸出
                for line in process.stdout:
                    log_and_print(line)

                # 等待該演算法執行完畢，並取得回傳碼
                process.wait()

                # 檢查是否發生 Error
                if process.returncode != 0:
                    log_and_print(f"\n{algo.upper()} 執行過程中發生錯誤！(Return Code: {process.returncode})\n")
                else:
                    log_and_print(f"\n{algo.upper()} 任務完成！\n")
                
            except Exception as e:
                log_and_print(f"啟動 {algo.upper()} 時發生系統錯誤: {e}\n")

    def learning_curve_workflow(branch="", config_path=""):
        log_and_print("\n==================================================\n")
        log_and_print("啟動 Learning Curve 實驗階段 (多 Seed 陰影繪圖模式)...\n")
        log_and_print("==================================================\n")
        
        # 將 seed 陣列轉成字串陣列，方便 subprocess 傳遞
        seed_strs = [str(s) for s in learning_curve_seeds]
        seeds_display = " ".join(seed_strs)
        
        for algo in nn_algorithms:
            algo_dir = os.path.join(root_dir, algo)
            branch_dir = os.path.join(algo_dir, branch)
            
            script_path = os.path.join(branch_dir, "learning_curve.py")
            if not os.path.exists(script_path):
                log_and_print(f"\n找不到學習曲線腳本 {script_path}，跳過 {algo.upper()}...\n")
                continue

            # 一次傳入所有 seed
            log_and_print(f"\n[{algo.upper()} | Seeds: {seeds_display}] 正在收集與繪製學習曲線數據...\n")
            
            try:
                my_env = os.environ.copy()
                my_env["PYTHONIOENCODING"] = "utf-8"
                
                # 將多個 seed 與演算法名稱一起傳進去
                command = [
                    sys.executable, "-X", "utf8", "-u", "learning_curve.py", 
                    "--config", config_path,
                    "--algo", algo.upper(), # 傳入實際演算法名稱供畫圖與存檔使用
                    "--seeds"
                ] + seed_strs # 展開 seed list 接在後面

                process = subprocess.Popen(
                    command,
                    cwd=branch_dir,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    env=my_env
                )

                for line in process.stdout:
                    log_and_print(line)

                process.wait()

                if process.returncode != 0:
                    log_and_print(f"\n{algo.upper()} 學習曲線發生錯誤！(Return Code: {process.returncode})\n")
                else:
                    log_and_print(f"\n{algo.upper()} 學習曲線數據收集與陰影繪圖完成！\n")
                    
            except Exception as e:
                log_and_print(f"啟動 {algo.upper()} Learning Curve 時發生系統錯誤: {e}\n")

    for branch, config_path in branch_map:
        # 依序遍歷演算法資料並執行訓練與評估 workflow
        if "learning_curve" in config_path:
            learning_curve_workflow(branch=branch, config_path=config_path)
        else:
            algo_workflow(is_nn_network=False, config_path=config_path)
            algo_workflow(is_nn_network=True, branch=branch, config_path=config_path)

    finish_msg = "\n 所有排定的演算法與參數組合已全部執行完畢！\n"
    log_and_print(finish_msg)