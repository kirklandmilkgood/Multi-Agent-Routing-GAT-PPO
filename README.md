# Multi-Agent Tour Planning (MATP) Experiment Framework

This repository contains the official implementation and experimental framework for the paper:

**Multi-Agent Tour Planning for Rural Mobile Clinics via Reinforcement Learning** Bay-Yuan Hsu, Li-Hung Lu  
Department of Industrial Engineering and Engineering Management, National Tsing Hua University, Taiwan  
Email: byhsu@ie.nthu.edu.tw, lihung@gapp.nthu.edu.tw  

## Overview
This repository provides a comprehensive and automated experimental framework for evaluating Multi-Agent Tour Planning (MATP) algorithms. It integrates state-of-the-art Multi-Agent Reinforcement Learning (MARL) baselines alongside traditional heuristic search algorithms. The framework is designed to conduct batch training and evaluation across various network topologies, ranging from synthetic Erdos-Renyi (ER) graphs to macro-city scale real-world road networks (e.g., Euroroad, Minnesota, Taichung).

## Directory Structure
The repository is organized into the following primary components:

- /dataset: Contains the graph data files, including Excel (.xlsx), Matrix Market (.mtx) formats, and visual representations (.png) of the road networks.
- /figures: The output directory for generated experimental result plots (e.g., solution quality and runtime comparisons).
- /tables: The output directory for generated data tables.
- /tools: Auxiliary scripts and utilities for data processing and graph generation.
- MARL Algorithm Directories (/DDTM, /DGN, /GAT_PPO, /MAPPO, /QMIX, /VDN): Each directory contains the implementation of a specific learning-based baseline. Subdirectories within these folders (e.g., learning_baseline_連邊機率) manage different experimental branches and contain the respective main_new.py execution scripts.
- Traditional Algorithm Scripts (GA.py, greedy.py, Model_P.py, MSA.py, RW.py): Standalone Python scripts implementing heuristic and exact search methods.
- Configuration Files (*_config.json): JSON files governing the hyperparameters, budget constraints, and graph settings for different experimental scenarios.
- run_experiments.py: The central automation script that orchestrates the execution of all algorithms and logs the results.
- requirements.txt: The list of Python dependencies required to execute the framework.

## Prerequisites and Installation
Ensure that Python 3.8 or higher is installed. Install the required dependencies using pip:

    pip install -r requirements.txt

## Configuration
Experimental parameters are controlled via JSON configuration files located in the root directory. Before running an experiment, ensure the desired configuration file is correctly mapped in the branch_map variable within run_experiments.py. 

Available configuration files include:
- er_graph_config.json
- euroroad_config.json
- minnesota_config.json
- large_network_config.json
- large_network_dynamic_config.json

## Execution
To initiate the automated experimental pipeline, execute the main runner script from the root directory:

    python run_experiments.py

The script operates in two phases:
1. Traditional Heuristics: Executes algorithms sequentially (Greedy, RW, GA, MSA, Model_P) based on the root-level Python scripts.
2. MARL Baselines: Navigates into the respective algorithm directories (VDN, DGN, MAPPO, GAT_PPO, DDTM, QMIX), targets the specified branch, and executes the main_new.py script.

To disable specific algorithms, comment them out in the algorithms or nn_algorithms lists within run_experiments.py.

## Logging and Output
The execution pipeline automatically captures standard output and standard error. Logs are written simultaneously to the console and to a timestamped log file in the root directory (format: experiment_YYYYMMDD_HHMMSS.log). This ensures zero data loss during prolonged execution and facilitates reproducible research.

---

# 多代理人路徑規劃 (MATP) 實驗框架

本 repo 為以下學術論文的官方實作與實驗框架：

**Multi-Agent Tour Planning for Rural Mobile Clinics via Reinforcement Learning** Bay-Yuan Hsu, Li-Hung Lu  
Department of Industrial Engineering and Engineering Management, National Tsing Hua University, Taiwan  
Email: byhsu@ie.nthu.edu.tw, lihung@gapp.nthu.edu.tw  

## 專案概述
本 repo 提供了一個完整且高度自動化的實驗框架，專門用於評估多代理人路徑規劃 (Multi-Agent Tour Planning, MATP) 演算法。本框架整合了目前最先進的多代理人強化學習 (MARL) 基準模型以及傳統啟發式搜尋演算法，旨在支援跨越多種網路拓撲的批次訓練與推論評估，涵蓋範圍從合成的 Erdos-Renyi (ER) 圖形到宏觀城市級別的真實路網 (如 Euroroad, Minnesota, Taichung)。

## 目錄結構
本專案的主要結構配置如下：

- /dataset: 存放圖形資料集檔案，包含 Excel (.xlsx)、Matrix Market (.mtx) 格式，以及路網的視覺化圖片 (.png)。
- /figures: 存放實驗結果生成的視覺化圖表 (例如：解的品質與執行時間比較圖)。
- /tables: 存放實驗結果產出的數據表格。
- /tools: 包含資料前處理與圖形生成所需的輔助腳本與工具。
- MARL 演算法目錄 (/DDTM, /DGN, /GAT_PPO, /MAPPO, /QMIX, /VDN): 各目錄包含特定學習型基準演算法的實作。這些資料夾內的子目錄 (如 learning_baseline_連邊機率) 用於管理不同的實驗分支，並包含對應的 main_new.py 執行腳本。
- 傳統演算法腳本 (GA.py, greedy.py, Model_P.py, MSA.py, RW.py): 實作傳統啟發式與精確搜尋方法的獨立 Python 腳本。
- 設定檔 (*_config.json): 用於控制不同實驗場景之超參數、預算限制與圖形設定的 JSON 檔案。
- run_experiments.py: 核心自動化腳本，負責協排所有演算法的執行與日誌記錄。
- requirements.txt: 執行本框架所需的 Python 依賴套件清單。

## 環境要求與安裝
請確保系統已安裝 Python 3.8 或以上版本。透過 pip 安裝所需的依賴套件：

    pip install -r requirements.txt

## 實驗設定
實驗參數統一由根目錄下的 JSON 設定檔進行控制。在執行實驗前，請確保 run_experiments.py 內的 branch_map 變數已正確映射至您欲執行的目標設定檔。

可用的設定檔包含：
- er_graph_config.json
- euroroad_config.json
- minnesota_config.json
- large_network_config.json
- large_network_dynamic_config.json

## 程式執行
請在專案根目錄下執行主控腳本以啟動自動化實驗管線：

    python run_experiments.py

該腳本的執行流程分為兩個階段：
1. 傳統啟發式演算法: 依序執行位於根目錄下的 Python 腳本 (Greedy, RW, GA, MSA, Model_P)。
2. MARL 基準演算法: 進入對應的演算法目錄 (VDN, DGN, MAPPO, GAT_PPO, DDTM, QMIX)，切換至指定的分支目錄，並執行 main_new.py 腳本。

若需停用特定演算法，請直接在 run_experiments.py 中的 algorithms 或 nn_algorithms 陣列裡將其註解。

## log 檔與輸出紀錄
執行管線會自動擷取標準輸出 (stdout) 與標準錯誤 (stderr)。所有的輸出資訊將會同步顯示於終端機，並寫入根目錄下帶有時間戳記的日誌檔案 (命名格式：experiment_YYYYMMDD_HHMMSS.log)。此機制可確保在長時間執行過程中不會發生資料遺失，並確保研究結果具備高度的可復現性。