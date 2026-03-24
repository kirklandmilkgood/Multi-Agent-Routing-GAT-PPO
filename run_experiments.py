import subprocess
import os
import sys
import datetime

# 取得絕對路徑的 config 檔案
root_dir = os.path.dirname(os.path.abspath(__file__))
euro_config_path = os.path.join(root_dir, "euroroad_config.json")
if not os.path.exists(euro_config_path):
    print(f"找不到全域設定檔: {euro_config_path}")
    sys.exit(1)
minnesota_config_path = os.path.join(root_dir, "minnesota_config.json")
if not os.path.exists(minnesota_config_path):
    print(f"找不到全域設定檔: {minnesota_config_path}")
    sys.exit(1)

# 定義想要依序跑的演算法資料夾名稱
# 可隨時註解掉不想跑的演算法
algorithms = [
    "VDN",
    "QMIX",
    "DGN",
    #"MAPPO",
    #"GAT_PPO"
]
branch_map = {
    "learning_baseline_連邊機率" : euro_config_path,
    "learning_baseline_邊數" : minnesota_config_path
}

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

    for branch, config_path in branch_map.items():
        # 依序遍歷演算法資料夾並執行
        for algo in algorithms:
            algo_dir = os.path.join(root_dir, algo)
            branch_dir = os.path.join(algo_dir, branch)
            script_path = os.path.join(branch_dir, "main_new.py")

            if not os.path.exists(script_path):
                log_and_print(f"\n找不到腳本 {script_path}，跳過 {algo.upper()} 演算法...\n")
                continue

            log_and_print(f"\n[{algo.upper()}] --------------------------------------------------\n")
            log_and_print(f"\n正在啟動 {algo.upper()} 的訓練與評估任務...\n")
            
            try:
                # 使用 Popen 取代 run，這樣能即時攔截輸出
                # stderr=subprocess.STDOUT 代表把錯誤也混入標準輸出，一起存進 Log
                process = subprocess.Popen(
                    [sys.executable, "-u", "main_new.py", config_path],
                    cwd=branch_dir,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace" # 遇到亂碼自動替換，防止腳本當機
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

    finish_msg = "\n 所有排定的演算法與參數組合已全部執行完畢！\n"
    log_and_print(finish_msg)