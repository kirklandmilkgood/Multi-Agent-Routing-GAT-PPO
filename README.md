# Multi-Agent Tour Planning (MATP) Experiment Framework

This repository contains the official implementation and experimental framework for the paper:

**Multi-Agent Tour Planning for Rural Mobile Clinics via Reinforcement Learning** Bay-Yuan Hsu, Li-Hung Lu  
Department of Industrial Engineering and Engineering Management, National Tsing Hua University, Taiwan  
Email: byhsu@ie.nthu.edu.tw, lihung@gapp.nthu.edu.tw  

## Overview
This repository provides a comprehensive and highly automated experimental framework for evaluating Multi-Agent Tour Planning (MATP) algorithms. It integrates our proposed **GAT-PPO** architecture alongside state-of-the-art Multi-Agent Reinforcement Learning (MARL) baselines and traditional heuristic search algorithms. The framework is designed to conduct batch training, robust evaluation, and automated visualization across various network topologies, ranging from synthetic Erdos-Renyi (ER) graphs to macro-city scale real-world road networks (e.g., Minnesota, Taichung) under both static and dynamic traffic conditions.

## Key Features
* **Proposed GAT-PPO Architecture:** Integrates Graph Attention Networks (GAT) with Proximal Policy Optimization (PPO) to achieve topological awareness and monotonic policy improvements in sparse-reward MATP domains.
* **Comprehensive SOTA Baselines:** Includes fully adapted implementations of value-based (VDN, QMIX, DGN), policy-based (MAPPO), and Transformer-based (DDTM) MARL algorithms, alongside exact solvers (Model P) and heuristics (GA, Greedy, RW, MSA).
* **Macro-Scale & Dynamic Environments:** Features the massive Taichung dataset (~10k nodes, 16k edges) and a Bounded Random Walk Congestion Model to simulate non-stationary, real-world traffic dynamics.
* **End-to-End Automated Pipeline:** A centralized runner script (`run_experiments.py`) seamlessly orchestrates model training, multi-seed evaluation, log tracking, JSON data aggregation, and academic-grade plot generation (e.g., shaded learning curves).

## Directory Structure
The repository is systematically organized into the following primary components:

- `/dataset/`: Contains graph data files, including Excel (`.xlsx`) formats and visual representations (`.png`) of the road networks.
- `/figures/` & `/tables/`: Output directories for generated experimental result plots (e.g., solution quality and runtime comparisons) and data tables.
- `/learning_curve/`: Output directory specifically for storing aggregated multi-seed training data (`learning_curves.json`) and generated academic shaded learning curve plots (`.png`).
- `/tools/`: Auxiliary scripts and utilities for data processing, JSON parsing, and automated Seaborn/Matplotlib plotting.
- **MARL Algorithm Directories** (`/DDTM`, `/DGN`, `/GAT_PPO`, `/MAPPO`, `/QMIX`, `/VDN`): Each contains the implementation of a specific learning-based algorithm. Subdirectories manage different experimental branches (e.g., dynamic vs. static) and contain the respective execution scripts (`main_new.py` or `learning_curve.py`).
- **Traditional Algorithm Scripts** (`GA.py`, `greedy.py`, `Model_P.py`, etc.): Standalone Python scripts implementing heuristic and exact search methods.
- **Configuration Files** (`*_config.json`): JSON files governing hyperparameters, budget constraints, and graph settings for different experimental scenarios.
- `run_experiments.py`: The central automation script that orchestrates the execution of all algorithms and manages logging.
- `requirements.txt`: The list of Python dependencies required to execute the framework.

## Prerequisites and Installation
Ensure that Python 3.8 or higher is installed. Install the required dependencies (including PyTorch, PyTorch Geometric, NetworkX, Pandas, Matplotlib, and Seaborn) using pip:

```bash
pip install -r requirements.txt
```

## Configuration & Datasets
Experimental parameters are strictly controlled via JSON configuration files located in the root directory.

* **Standard Evaluation Configs:** `er_graph_config.json`, `euroroad_config.json`, `minnesota_config.json`, `large_network_config.json` (Taichung static), `large_network_dynamic_config.json` (Taichung dynamic).
* **Learning Curve Configs:** `learning_curve_minnesota_config.json`, `learning_curve_taichung_config.json`.

Before running an experiment, ensure the desired configuration file is correctly mapped in the `branch_map` variable within `run_experiments.py`.

## Execution and Usage
To initiate the automated experimental pipeline, execute the main runner script from the root directory:

```bash
python run_experiments.py
```

The script intelligently routes execution based on the configuration file name:

1. **Standard Evaluation Mode:** (Triggered by standard configs)
   * Executes traditional heuristics sequentially.
   * Navigates into the respective MARL directories, targets the specified branch, and executes `main_new.py` to evaluate solution quality and inference runtime.
2. **Learning Curve Generation Mode:** (Triggered if the config name contains `learning_curve`)
   * Bypasses traditional heuristics and only executes MARL baselines.
   * Automatically iterates through predefined random seeds (e.g., 42, 100, 123) by invoking `learning_curve.py`.
   * Aggregates the step-by-step training rewards into a global JSON file and generates highly readable, academic-grade plots with standard deviation shading.

*Note: To disable specific algorithms, simply comment them out in the `algorithms` or `nn_algorithms` lists within `run_experiments.py`.*

## Logging and Output
The execution pipeline automatically captures standard output (`stdout`) and standard error (`stderr`). Logs are written simultaneously to the console and to a timestamped log file in the root directory (format: `experiment_YYYYMMDD_HHMMSS.log`). This ensures zero data loss during prolonged execution, facilitates debugging, and guarantees reproducible research.

---

# 多代理人路徑規劃 (MATP) 實驗框架

本 Repository 為以下學術論文的官方實作與實驗框架：

**Multi-Agent Tour Planning for Rural Mobile Clinics via Reinforcement Learning** Bay-Yuan Hsu, Li-Hung Lu  
Department of Industrial Engineering and Engineering Management, National Tsing Hua University, Taiwan  
Email: byhsu@ie.nthu.edu.tw, lihung@gapp.nthu.edu.tw  

## 專案概述
本專案提供了一個完整且高度自動化的實驗框架，專門用於評估多代理人路徑規劃 (Multi-Agent Tour Planning, MATP) 演算法。本框架整合了我們提出的 **GAT-PPO** 架構，以及目前最先進的多代理人強化學習 (MARL) 基準模型與傳統啟發式搜尋演算法。本框架專為批次訓練、強健性評估與自動化視覺化而設計，支援跨越多種網路拓撲的實驗，涵蓋範圍從合成的 Erdos-Renyi (ER) 圖形到宏觀城市級別的真實路網 (如 Minnesota, Taichung)，並同時支援靜態與動態交通壅塞環境。

## 核心特色
* **GAT-PPO 架構：** 結合圖注意力網路 (GAT) 與近端策略最佳化 (PPO)，在稀疏獎勵的 MATP 領域中實現拓撲感知與策略的單調改進。
* **完整的 SOTA 基準模型：** 包含全面適配 MATP 限制的基於價值 (VDN, QMIX, DGN)、基於策略 (MAPPO) 與基於 Transformer (DDTM) 的 MARL 演算法，以及精確求解器 (Model P) 和啟發式演算法 (GA, Greedy, RW, MSA)。
* **宏觀與動態環境：** 引入龐大的台中真實路網資料集 (~10,000 節點，16,000 邊)，並內建有界隨機漫步壅塞模型以模擬非平穩的真實動態交通。
* **端到端自動化管線：** 透過單一主控腳本 (`run_experiments.py`) 無縫協調模型訓練、多種子 (Multi-seed) 評估、日誌追蹤、JSON 數據聚合，以及學術級圖表 (如帶有標準差陰影的學習曲線) 的自動生成。

## 目錄結構
本專案的主要結構配置如下：

- `/dataset/`: 存放圖形資料集檔案，包含 Excel (`.xlsx`) 格式以及路網的視覺化圖片 (`.png`)。
- `/figures/` 與 `/tables/`: 存放實驗結果生成的視覺化圖表 (例如：解的品質與執行時間比較圖) 及數據表格。
- `/learning_curve/`: 專門存放多種子訓練過程的聚合數據 (`learning_curves.json`) 以及自動生成的帶陰影學習曲線圖 (`.png`)。
- `/tools/`: 包含資料前處理、JSON 解析與自動化 Seaborn/Matplotlib 繪圖的輔助腳本。
- **MARL 演算法目錄** (`/DDTM`, `/DGN`, `/GAT_PPO`, `/MAPPO`, `/QMIX`, `/VDN`): 各目錄包含特定學習型基準演算法的實作。資料夾內的子目錄用於管理不同的實驗分支 (如靜態與動態)，並包含對應的執行腳本 (`main_new.py` 或 `learning_curve.py`)。
- **傳統演算法腳本** (`GA.py`, `greedy.py`, `Model_P.py` 等): 實作傳統啟發式與精確搜尋方法的獨立 Python 腳本。
- **設定檔** (`*_config.json`): 用於控制不同實驗場景之超參數、預算限制與圖形設定的 JSON 檔案。
- `run_experiments.py`: 核心自動化腳本，負責排程所有演算法的執行與日誌記錄。
- `requirements.txt`: 執行本框架所需的 Python 依賴套件清單。

## 環境要求與安裝
請確保系統已安裝 Python 3.8 或以上版本。透過 pip 安裝所需的依賴套件 (包含 PyTorch, PyTorch Geometric, NetworkX, Pandas, Matplotlib, Seaborn 等)：

```bash
pip install -r requirements.txt
```

## 實驗設定與資料集
實驗參數統一由根目錄下的 JSON 設定檔進行嚴格控制。

* **標準效能評估設定檔：** `er_graph_config.json`, `euroroad_config.json`, `minnesota_config.json`, `large_network_config.json` (台中靜態), `large_network_dynamic_config.json` (台中動態)。
* **學習曲線繪製設定檔：** `learning_curve_minnesota_config.json`, `learning_curve_taichung_config.json`。

在執行實驗前，請確保 `run_experiments.py` 內的 `branch_map` 變數已正確映射至您欲執行的目標設定檔。

## 程式執行與使用方法
請在專案根目錄下執行主控腳本以啟動自動化實驗管線：

```bash
python run_experiments.py
```

該腳本會根據設定檔的名稱自動切換執行模式：

1. **標準效能評估模式：** (由標準設定檔觸發)
   * 依序執行傳統啟發式演算法。
   * 進入對應的 MARL 目錄與分支，執行 `main_new.py` 以評估解的品質與推論執行時間。
2. **學習曲線繪製模式：** (若設定檔名稱包含 `learning_curve` 則觸發)
   * 略過傳統演算法，僅執行 MARL 基準模型。
   * 透過呼叫 `learning_curve.py` 自動遍歷預設的隨機種子 (如 42, 100, 123)。
   * 將每一步的訓練獎勵聚合至全域 JSON 檔案中，並自動繪製出帶有標準差陰影的圖表。

*註：若需停用特定演算法，請直接在 `run_experiments.py` 中的 `algorithms` 或 `nn_algorithms` 陣列裡將其註解即可。*

## log 檔與輸出紀錄
執行管線會自動擷取標準輸出 (`stdout`) 與標準錯誤 (`stderr`)。所有的輸出資訊將會同步顯示於終端機，並即時寫入根目錄下帶有時間戳記的日誌檔案 (命名格式：`experiment_YYYYMMDD_HHMMSS.log`)。此機制可確保在長時間訓練過程中不會發生資料遺失，方便除錯，並保障研究結果的完全可復現性。