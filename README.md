# LSTM 台股價格預測系統

## 專案描述

本專案使用 LSTM 深度學習模型預測台股未來 20 個交易日的漲跌幅區間（5 類分類）與收盤價。系統支援自動超參數調整（含特徵集選擇），並可根據使用者指定日期進行預測。

## 功能特色

- ✅ **基準模型訓練**: 使用歷史台股資料訓練 LSTM 模型
- ✅ **自動超參數調整**: 使用 Keras Tuner 探索最佳模型配置
- ✅ **特徵集選擇**: 支援三種特徵集（動能型、震盪型、全特徵集）
- ✅ **GPU 加速**: 支援 NVIDIA GPU 訓練，自動降級至 CPU
- ✅ **日期預測**: 根據指定日期預測未來收盤價區間
- ✅ **自動配置管理**: 訓練時自動保存模型配置，預測時自動載入正確參數
- ✅ **時間戳記版本管理**: 每次訓練自動加上時間戳記，永不覆蓋舊模型（v2.1 新增）
- ✅ **基準模型比較**: 超參數調整時可與基準模型比較，自動生成改善報告（v2.1 新增）

## 環境需求

- **Python**: 3.12 (推薦，Ubuntu 24.04 預設版本)
- **TensorFlow**: 2.16.1 (含自動 CUDA 支援)
- **GPU**: NVIDIA GPU (選用，TensorFlow 會自動安裝 CUDA/cuDNN)
- **作業系統**: WSL2 (Ubuntu 22.04/24.04) 或 Linux
- **記憶體**: 建議 16GB+ RAM, 6GB+ VRAM (使用 GPU 時)

## 安裝步驟

### WSL2 環境設置（推薦）

本專案主要在 WSL2 (Ubuntu) 環境開發與測試，以下為詳細設置步驟：

#### 步驟 1: 啟動 WSL2 並切換到專案目錄

```bash
# 從 Windows 啟動 WSL2 (Ubuntu)
wsl -d Ubuntu_D

# 切換到專案目錄（Windows 磁碟機映射到 /mnt/）
cd /mnt/d/000-github-repositories/stock-price-prediction-v04
```

#### 步驟 2: 確認 Python 版本

```bash
# 檢查 Python 版本（需要 3.12 推薦）
python3 --version
```

**Ubuntu 24.04 (Noble) 使用者**：
- ✅ 預設 Python 3.12（完全相容，直接使用）
- ✅ 不需要額外安裝

**Ubuntu 22.04 使用者**：
```bash
# 需要安裝 Python 3.12
sudo apt update
sudo apt install software-properties-common -y
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update
sudo apt install python3.12 python3.12-venv python3.12-dev
```

**為什麼選擇 Python 3.12？**
- Ubuntu 24.04 預設版本，無需額外安裝
- TensorFlow 2.16.1 完全支援
- 與 NumPy 1.26.2、Pandas 2.1.4 相容性佳
- 效能比 3.11 更好

**安裝必要套件**：
```bash
# 安裝 Python 開發套件
sudo apt update
sudo apt install python3-pip python3-venv python3-dev build-essential
```

#### 步驟 3: 建立 Python 虛擬環境

```bash
# 建立虛擬環境（使用 Python 3.12）
python3.12 -m venv venv

# 啟動虛擬環境
source venv/bin/activate

# 確認虛擬環境已啟動（命令列前會顯示 (venv)）
which python  # 應顯示: /mnt/d/.../venv/bin/python
python --version  # 應顯示: Python 3.12.x
```

#### 步驟 4: 升級 pip 並安裝依賴套件

```bash
# 升級 pip
pip install --upgrade pip

# 安裝依賴套件（包含 TensorFlow 和自動 CUDA 支援）
pip install -r requirements.txt
```

**重要提示**：
- `tensorflow[and-cuda]==2.16.1` 會**自動安裝** CUDA 12.3 和 cuDNN 8.9
- **無需手動安裝** CUDA Toolkit 或 cuDNN
- 安裝時間約 5-10 分鐘（首次安裝會下載 CUDA 相關套件）
- 總下載大小約 2-3 GB

#### 步驟 5: 驗證系統環境

```bash
# 運行系統驗證腳本（檢查所有配置）
python validate_pipeline.py
```

驗證腳本會自動檢查：
- ✅ Python 版本（應為 3.12.x）
- ✅ TensorFlow 2.16.1 安裝
- ✅ GPU 可用性（自動偵測 NVIDIA GPU）
- ✅ CUDA 環境（TensorFlow 自動配置）
- ✅ 必要套件版本
- ✅ 專案結構完整性

**關於 GPU 支援**：

```bash
# 檢查 NVIDIA GPU 驅動（需要在 Windows 安裝 NVIDIA CUDA on WSL 驅動）
nvidia-smi

# 驗證 TensorFlow GPU 支援
python -c "import tensorflow as tf; print('TensorFlow 版本:', tf.__version__)"
python -c "import tensorflow as tf; print('GPU 裝置:', tf.config.list_physical_devices('GPU'))"
```

**預期結果**：
- 如果有 NVIDIA GPU 且驅動正確：`GPU 裝置: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]`
- 如果沒有 GPU 或驅動未安裝：`GPU 裝置: []`（系統會自動使用 CPU）

**GPU 驅動安裝**：
- 僅需在 **Windows** 安裝 NVIDIA CUDA on WSL 驅動程式
- 下載位置: https://developer.nvidia.com/cuda/wsl
- WSL2 內**不需要**手動安裝 CUDA（TensorFlow 已自動安裝）

#### 步驟 6: 快速測試

```bash
# 快速測試 GPU 設置
python -c "from src.utils.gpu_checker import setup_gpu; setup_gpu()"

# 執行單元測試（確認所有模組正常）
pytest tests/unit/test_data_loader.py -v

# 或執行完整系統驗證
python validate_pipeline.py
```

### Windows 環境（不推薦）

若必須在 Windows 環境執行：

```bash
# 建立虛擬環境
python -m venv venv

# 啟動虛擬環境
venv\Scripts\activate

# 安裝依賴套件
pip install -r requirements.txt
```

**注意**: Windows 環境可能遇到路徑與編碼問題，強烈建議使用 WSL2。

## 使用方式

### 在 WSL2 環境執行

**前置作業**：確保已啟動虛擬環境
```bash
# 切換到專案目錄
cd /mnt/d/000-github-repositories/stock-price-prediction-v04

# 啟動虛擬環境
source venv/bin/activate
```

### 1. 訓練基準模型

```bash
# 使用 Set A 特徵集訓練基準模型
python src/cli/train.py \
    --data-file 19980601-20251111-converted.csv \
    --feature-set "Set A" \
    --time-steps 60 \
    --epochs 100 \
    --batch-size 32 \
    --model-name baseline_model \
    --output-dir models/
```

**訓練參數說明**：
- `--data-file`: CSV 資料檔案路徑（可使用相對路徑）
- `--feature-set`: 特徵集選擇 (Set A/B/C)
- `--time-steps`: 時間窗口大小 (20/40/60/80)
- `--epochs`: 訓練週期數（建議 100）
- `--batch-size`: 批次大小（GPU: 32-64, CPU: 16-32）
- `--model-name`: 模型名稱（預設: baseline_model，會自動加上時間戳記）
- `--output-dir`: 模型儲存目錄

**執行時間參考**（基於 100 epochs）：
- GPU 環境: 約 1-2 小時
- CPU 環境: 約 8-12 小時

**輸出檔案**（自動加上時間戳記）：
```
models/baseline_model_20251116_144357.h5              ← 模型檔案
models/baseline_model_20251116_144357_config.json     ← 配置檔案（含訓練參數）
models/baseline_model_20251116_144357_scaler.pkl      ← 特徵縮放器
logs/training_logs/baseline_model_20251116_144357_training_log.csv  ← 訓練日誌
```

**重要說明**：
- ✅ 每次訓練會自動加上時間戳記（格式: YYYYMMDD_HHMMSS）
- ✅ 不會覆蓋舊模型，所有歷史版本都保留
- ✅ 模型和 scaler 使用相同的時間戳記，方便配對
- 📝 記下模型路徑以便後續超參數調整時使用

### 2. 執行超參數調整（進階）

#### 方法 1: 與基準模型比較（推薦）

```bash
# 使用基準模型進行超參數調整並比較效能
python src/cli/tune.py \
    --data-file 19980601-20251111-converted.csv \
    --max-trials 50 \
    --epochs-per-trial 50 \
    --baseline-model models/baseline_model_20251116_151143.h5 \
    --tuned-model-name best_tuned_model \
    --tuner-type bayesian \
    --project-name lstm_stock_tuning_20251116_151143 \
    --overwrite \
    --output-dir models/
```

**調參參數說明**：
- `--max-trials`: 最大試驗次數（建議 50-100）
- `--epochs-per-trial`: 每次試驗的訓練週期（建議 50）
- `--baseline-model`: 基準模型路徑（用於比較，v2.1 新增）
- `--tuned-model-name`: 調整模型名稱（預設: best_tuned_model，會自動加時間戳記）
- `--tuner-type`: Tuner 類型
  * `random`: 隨機搜尋（快速）
  * `bayesian`: 貝葉斯優化（推薦）
  * `hyperband`: Hyperband 演算法
- `--overwrite`: 覆寫先前的試驗紀錄（⚠️ 重要參數，見下方說明）
- `--project-name`: Tuner 專案名稱（預設: lstm_stock_tuning）

**⚠️ 關於試驗數據緩存機制**：

Keras Tuner 會自動保存所有試驗結果到 `logs/tuning_logs/<project_name>/`。當你再次執行調整時：

- **預設行為（不加 `--overwrite`）**：
  - ✅ 重複使用之前的試驗結果（不重新訓練）
  - ✅ 如果 `--max-trials` 更大，只訓練額外的試驗
  - ⚠️ 這會導致「瞬間完成」的情況
  - 💡 適用於：想要繼續之前的調整，增加試驗次數

- **加上 `--overwrite` 參數**：
  - 🔄 完全刪除舊的試驗數據
  - 🔄 重新開始所有試驗
  - 💡 適用於：資料更新、參數改變、需要重新訓練

**何時需要使用 `--overwrite`**：
1. ✅ CSV 資料檔案更新時
2. ✅ 想要完全重新訓練時
3. ✅ 修改了模型架構或特徵工程時
4. ✅ 發現之前的試驗結果有問題時

**範例**：
```bash
# 完全重新訓練（清除舊數據）
python src/cli/tune.py \
    --data-file updated_data.csv \
    --max-trials 50 \
    --baseline-model models/baseline_model_20251116_144357.h5 \
    --overwrite  # 加上這個參數

# 或使用不同的專案名稱（保留舊數據）
python src/cli/tune.py \
    --data-file updated_data.csv \
    --max-trials 50 \
    --project-name lstm_stock_tuning_v2  # 使用新名稱
```

**輸出檔案**（自動加上時間戳記）：
```
models/best_tuned_model_20251116_153022.h5                    ← 調整後模型
models/best_tuned_model_20251116_153022_config.json           ← 配置檔案（含基準模型路徑）
logs/tuning_logs/tuning_results_20251116_153022.txt           ← 詳細調整結果（前10名）
logs/tuning_logs/comparison_report_20251116_153022.txt        ← 與基準模型比較報告
logs/tuning_logs/lstm_stock_tuning/                           ← Keras Tuner 試驗紀錄
```

**配置檔案範例**（`best_tuned_model_20251116_153022_config.json`）：
```json
{
  "model_file": "models/best_tuned_model_20251116_153022.h5",
  "feature_set_id": "Set B",
  "time_steps": 60,
  "created_at": "2025-11-16 15:30:22",
  "num_layers": 3,
  "dropout_rate": 0.2,
  "learning_rate": 0.0005,
  "batch_size": 64,
  "epochs_trained": 50,
  "tuner_type": "bayesian",
  "max_trials": 50,
  "baseline_model": "models/baseline_model_20251116_144357.h5",  ← 記錄基準模型
  "tuning_timestamp": "20251116_153022",
  "model_type": "tuned"
}
```

**比較報告範例**：
```
================================================================================
模型比較報告
================================================================================

調整時間: 20251116_153022
基準模型: models/baseline_model_20251116_144357.h5
調整模型: models/best_tuned_model_20251116_153022.h5

效能比較（測試集）:
  基準模型 - 測試準確度: 45.67%
  調整模型 - 測試準確度: 48.92%
  準確度改善: +7.12%

✅ 調整模型優於基準模型
```

#### 方法 2: 僅調整（不比較）

```bash
# 不指定基準模型，僅進行超參數調整
python src/cli/tune.py \
    --data-file 19980601-20251111-converted.csv \
    --max-trials 50 \
    --epochs-per-trial 50 \
    --tuner-type bayesian \
    --output-dir models/
```

**執行時間參考**（50 trials, 50 epochs/trial）：
- GPU 環境: 約 6-12 小時
- CPU 環境: 約 2-3 天

**重要說明**：
- ✅ 調參會自動搜尋最佳特徵集（Set A/B/C）
- ✅ 訓練完成後自動保存模型配置（含基準模型路徑）
- ✅ 自動生成詳細的調整結果和比較報告
- ✅ 前 10 名試驗結果會完整記錄在日誌中
- 📊 可隨時查看比較報告了解改善幅度
- 建議使用 `--overwrite` 參數重新開始，或省略以繼續先前的調參
- 可使用 `nohup` 在背景執行：
  ```bash
  nohup python src/cli/tune.py \
      --data-file 19980601-20251111-converted.csv \
      --max-trials 50 \
      --epochs-per-trial 50 \
      --baseline-model models/baseline_model_20251116_144357.h5 > tune.log 2>&1 &
  ```

### 3. 執行預測

#### 方法 1: 自動載入配置（推薦）

系統會自動從配置文件（`{model_name}_config.json`）載入正確的參數：

```bash
# 不需要指定 --feature-set 和 --time-steps
python src/cli/predict.py \
    --model-file models/best_tuned_model.h5 \
    --data-file 19980601-20251111-converted.csv \
    --input-date "2024-01-15"
```

系統會自動：
- 尋找 `models/best_tuned_model_config.json`
- 讀取正確的 `feature_set_id` 和 `time_steps`
- 在日誌中顯示使用的配置

#### 方法 2: 手動指定參數

如果需要手動指定參數（例如測試不同配置）：

```bash
python src/cli/predict.py \
    --model-file models/best_tuned_model.h5 \
    --data-file 19980601-20251111-converted.csv \
    --input-date "2024-01-15" \
    --feature-set "Set B" \
    --time-steps 60
```

**預測參數說明**：
- `--model-file`: 訓練好的模型檔案路徑
- `--input-date`: 輸入日期（格式: YYYY-MM-DD 或 YYYY/M/D）
- `--feature-set`: （選用）特徵集，未指定時自動從配置載入
- `--time-steps`: （選用）時間窗口，未指定時自動從配置載入

**重要提示**：
- 使用 `tune.py` 或 `train.py` 訓練的模型會自動保存配置文件
- 舊模型如需配置文件，請參考 `MODEL_CONFIG_GUIDE.md` 手動創建
- 配置文件命名規則：`{model_name}.h5` → `{model_name}_config.json`

**批次預測多個日期**：
```bash
# 自動載入配置（推薦）
python src/cli/predict.py \
    --model-file models/best_tuned_model.h5 \
    --data-file 19980601-20251111-converted.csv \
    --input-date "2024-01-15" "2024-02-20" "2024-03-10"

# 或手動指定參數
python src/cli/predict.py \
    --model-file models/best_tuned_model.h5 \
    --data-file 19980601-20251111-converted.csv \
    --input-date "2024-01-15" "2024-02-20" "2024-03-10" \
    --feature-set "Set B" \
    --time-steps 60
```

**儲存預測結果至檔案**：
```bash
python src/cli/predict.py \
    --model-file models/best_tuned_model.h5 \
    --data-file 19980601-20251111-converted.csv \
    --input-date "2024-01-15" \
    --output-file predictions/result_20240115.txt
```

### 完整工作流程範例

```bash
# 1. 切換到專案目錄並啟動虛擬環境
cd /mnt/d/000-github-repositories/stock-price-prediction-v04
source venv/bin/activate

# 2. 訓練基準模型
python src/cli/train.py \
    --data-file 19980601-20251111-converted.csv \
    --feature-set "Set A" \
    --time-steps 60 \
    --epochs 100 \
    --batch-size 32 \
    --model-name baseline_model

# 輸出範例：models/baseline_model_20251116_144357.h5
# 📝 記下這個路徑！

# 3. 執行超參數調整（與基準模型比較）
python src/cli/tune.py \
    --data-file 19980601-20251111-converted.csv \
    --max-trials 50 \
    --epochs-per-trial 50 \
    --baseline-model models/baseline_model_20251116_144357.h5 \
    --tuned-model-name best_tuned_model \
    --tuner-type bayesian

# 輸出範例：
# - models/best_tuned_model_20251116_153022.h5
# - logs/tuning_logs/comparison_report_20251116_153022.txt

# 4. 查看比較報告
cat logs/tuning_logs/comparison_report_20251116_153022.txt

# 5. 使用最佳模型進行預測（自動載入配置）
python src/cli/predict.py \
    --model-file models/best_tuned_model_20251116_153022.h5 \
    --data-file 19980601-20251111-converted.csv \
    --input-date "2024-06-01"
```

**或使用自動化腳本**：

```bash
# Linux/Mac
bash example_workflow.sh

# Windows
example_workflow.bat
```

這會自動執行：訓練基準模型 → 超參數調整 → 生成比較報告

## 模型配置自動管理

### 📋 功能說明

從 v2.0 開始，系統支援**自動保存和載入模型配置**，解決了「不知道模型訓練時使用哪些參數」的問題。

### ✨ 主要優點

1. **防止配置錯誤**: 自動使用訓練時的正確參數
2. **簡化預測命令**: 不需要記住每個模型的 `--feature-set` 和 `--time-steps`
3. **可追溯性**: 配置文件記錄了模型的所有訓練參數

### 🎯 如何使用

#### 訓練時自動保存配置

當您使用 `tune.py` 或 `train.py` 訓練模型時，系統會自動創建兩個文件：
- `models/best_tuned_model.h5` - 模型文件
- `models/best_tuned_model_config.json` - **配置文件（自動創建）**

配置文件包含：
```json
{
  "model_file": "models/best_tuned_model.h5",
  "feature_set_id": "Set B",
  "time_steps": 60,
  "created_at": "2025-11-16 11:45:32",
  "num_layers": 3,
  "dropout_rate": 0.2,
  "learning_rate": 0.0005,
  "batch_size": 64
}
```

#### 預測時自動載入配置

執行預測時，系統會自動讀取配置文件：

```bash
# 新方式：不需要指定 --feature-set 和 --time-steps
python src/cli/predict.py \
    --model-file models/best_tuned_model.h5 \
    --data-file data.csv \
    --input-date "2024-01-15"

# 系統會自動：
# 1. 尋找 models/best_tuned_model_config.json
# 2. 讀取 feature_set_id 和 time_steps
# 3. 在日誌中顯示：
#    📋 使用模型配置中的特徵集: Set B
#    📋 使用模型配置中的時間窗口: 60
```

### 📚 詳細文檔

完整的使用指南請參考：[MODEL_CONFIG_GUIDE.md](MODEL_CONFIG_GUIDE.md)

包含：
- 配置文件格式說明
- 查看和管理配置的方法
- 舊模型的配置文件創建
- 手動覆蓋配置的方式
- 技術實作細節

## 專案結構

```
.
├── src/
│   ├── data/              # 資料載入與分割
│   ├── features/          # 特徵工程
│   ├── models/            # LSTM 模型定義
│   ├── tuning/            # 超參數調整
│   ├── prediction/        # 預測邏輯
│   ├── utils/             # 工具函式（含模型配置管理）
│   └── cli/               # CLI 指令
├── tests/
│   ├── unit/              # 單元測試
│   └── integration/       # 整合測試
├── models/                # 訓練好的模型與配置文件
├── logs/                  # 訓練日誌
├── notebooks/             # Jupyter Notebook（探索性分析）
└── requirements.txt       # Python 依賴套件
```

## 測試

```bash
# 執行所有測試
pytest tests/ -v

# 執行單元測試
pytest tests/unit/ -v

# 執行整合測試
pytest tests/integration/ -v
```

## 程式碼品質檢查

```bash
# 格式化程式碼
black src/ tests/

# 檢查程式碼風格
flake8 src/ tests/

# 型別檢查
mypy src/
```

## 資料格式

輸入 CSV 檔案需包含以下欄位：
- `date`: 交易日期 (YYYY/M/D)
- `open`, `high`, `low`, `close`: 開盤價、最高價、最低價、收盤價
- `volume`: 成交量（億）
- `SMA5`, `SMA10`, `SMA20`, `SMA60`, `SMA120`, `SMA240`: 簡單移動平均線
- `MA5`, `MA10`: 移動平均線
- `DIF12-26`, `MACD9`, `OSC`: MACD 指標
- `K(9,3)`, `D(9,3)`: KD 指標
- `net buy sell`, `cumulative net buy sell`, `buy`, `sell`: 法人籌碼資料

## 常見問題

### WSL2 相關問題

#### Q1: 如何在 WSL2 中存取 Windows 檔案？

Windows 磁碟機自動掛載在 `/mnt/` 下：
```bash
# D: 磁碟機
cd /mnt/d/your-path/

# C: 磁碟機
cd /mnt/c/Users/YourName/
```

#### Q2: WSL2 中如何啟用 GPU？

1. **在 Windows 安裝 NVIDIA CUDA on WSL 驅動**：
   - 下載：https://developer.nvidia.com/cuda/wsl
   - 安裝後重啟 WSL2

2. **在 WSL2 中驗證**：
   ```bash
   nvidia-smi  # 應顯示 GPU 資訊
   ```

3. **不要在 WSL2 內安裝 CUDA Toolkit**：
   - TensorFlow-GPU 會自動使用 Windows 的 CUDA

#### Q3: 虛擬環境在 WSL2 中建立失敗？

```bash
# 確保已安裝 python3-venv
sudo apt update
sudo apt install python3.9-venv

# 若還是失敗，檢查磁碟權限
ls -la /mnt/d/your-project/  # 確認有寫入權限
```

#### Q4: WSL2 執行速度很慢？

建議將專案移到 WSL2 檔案系統（非 `/mnt/` 下）：
```bash
# 複製專案到 WSL2 home 目錄
cp -r /mnt/d/stock-price-prediction-v04 ~/

# 在 home 目錄執行會更快
cd ~/stock-price-prediction-v04
```

### GPU 相關問題

#### GPU 不可用怎麼辦？

系統會自動降級使用 CPU 訓練。若需啟用 GPU，請確認：
1. **WSL2**: 安裝 NVIDIA CUDA on WSL 驅動（從 Windows 安裝）
2. **Linux**: CUDA 與 cuDNN 已正確安裝
3. **驗證**: `nvidia-smi` 能正常顯示 GPU 資訊
4. **TensorFlow**: 版本與 CUDA 版本相容（建議使用 requirements.txt）

#### 訓練時 GPU 記憶體不足？

調整以下參數：
```bash
# 減少批次大小
--batch-size 16  # 從 32 降至 16

# 減少時間窗口
--time-steps 40  # 從 60 降至 40

# 使用較少 LSTM 單元數（在調參時）
# 系統會自動嘗試較小的配置
```

或在程式碼中啟用記憶體動態增長（已內建）：
```python
# src/utils/gpu_checker.py 已實作
tf.config.experimental.set_memory_growth(gpu, True)
```

### 訓練與預測問題

#### 驗證準確度低於預期？

檢查以下項目：
1. **資料品質**: 確認 CSV 檔案包含所有必要欄位
2. **特徵工程**: 使用 `--feature-set "Set C"` 嘗試全特徵集
3. **訓練週期**: 增加 `--epochs` 至 150-200
4. **超參數調整**: 執行 `tune.py` 尋找最佳配置
5. **資料量**: 確保至少有 200+ 個交易日的資料

#### 預測時報錯「資料不足」？

```bash
# 錯誤訊息: ValueError: 輸入日期之前的資料不足 60 天

# 解決方法:
# 1. 選擇較晚的日期（確保前面有足夠歷史資料）
--input-date "2024-06-01"  # 而非 "2024-01-01"

# 2. 或減少時間窗口
--time-steps 20  # 而非 60
```

#### 如何查看訓練進度？

訓練日誌會儲存在 `logs/` 目錄：
```bash
# 查看訓練日誌
tail -f logs/training_logs/baseline_model_training_log.csv

# 查看調參進度
tail -f logs/tuning_logs/lstm_stock_tuning/*/trial_*/trial.json
```

#### 如何中斷並恢復調參？

```bash
# 中斷: Ctrl+C

# 恢復: 省略 --overwrite 參數，系統會自動繼續
python src/cli/tune.py \
    --data-file data.csv \
    --max-trials 50 \
    --epochs-per-trial 50
    # 注意：不要加 --overwrite
```

### 效能優化建議

#### 如何加快訓練速度？

1. **使用 GPU**: 速度可提升 5-10 倍
2. **調整批次大小**: GPU 可用 64-128，CPU 使用 16-32
3. **減少訓練週期**: 開發時使用 10-20 epochs 快速驗證
4. **使用較小的時間窗口**: 40 而非 80

#### 如何節省磁碟空間？

```bash
# 清理舊的模型檔案
rm models/backup_*.h5

# 清理 Keras Tuner 試驗紀錄（保留最佳模型）
rm -rf logs/tuning_logs/*/trial_*

# 清理 __pycache__
find . -type d -name __pycache__ -exec rm -rf {} +
```

## 相關文件

### 核心功能文件
- [模型配置自動載入指南](MODEL_CONFIG_GUIDE.md) - **重要！如何使用自動配置功能**
- [時間戳記功能說明](TIMESTAMP_FEATURE.md) - **v2.1 新增！自動版本管理**
- [超參數調整完整指南](HYPERPARAMETER_TUNING_GUIDE.md) - **v2.1 新增！基準模型比較**
- [訓練系統改進總結](TRAINING_SYSTEM_IMPROVEMENTS.md) - **v2.1 新增！系統更新說明**

### 環境設置文件
- [已驗證的環境配置](VERIFIED_ENVIRONMENT.md) - WSL2 GPU 環境設置
- [WSL2 GPU 快速驗證](WSL2_GPU_QUICK_VERIFY.md) - GPU 設置驗證腳本

### 規格文件
- [功能規格](specs/001-lstm-stock-prediction/spec.md)
- [實作計畫](specs/001-lstm-stock-prediction/plan.md)
- [快速開始指南](specs/001-lstm-stock-prediction/quickstart.md)
- [資料模型](specs/001-lstm-stock-prediction/data-model.md)

## 版本更新

### v2.1 (2025-11-16) - 時間戳記與基準模型比較

**新增功能**：
- ✅ **自動時間戳記管理**：訓練和調整時自動加上時間戳記，永不覆蓋舊模型
- ✅ **基準模型比較**：超參數調整時可指定基準模型進行比較
- ✅ **詳細日誌記錄**：完整記錄調整過程和前 10 名試驗結果
- ✅ **自動比較報告**：自動生成模型效能比較報告
- ✅ **基準模型追蹤**：調整模型配置中記錄基於哪個基準模型
- ✅ **自動化工作流程腳本**：提供一鍵執行的範例腳本

**修改的檔案**：
- `src/models/model_builder.py` - 加入時間戳記功能
- `src/cli/train.py` - 更新模型儲存邏輯
- `src/cli/tune.py` - 加入基準模型比較功能
- `src/tuning/hyperparameter_tuner.py` - 加入比較和詳細日誌

**新增文件**：
- `TIMESTAMP_FEATURE.md` - 時間戳記功能說明
- `HYPERPARAMETER_TUNING_GUIDE.md` - 超參數調整完整指南
- `TRAINING_SYSTEM_IMPROVEMENTS.md` - 系統改進總結
- `example_workflow.sh` / `.bat` - 自動化工作流程腳本
- `test_timestamp.py` - 功能測試腳本

**破壞性變更**：
- 無。所有新功能預設啟用，但向後相容

**遷移指南**：
- 舊模型仍可正常使用
- 新訓練會自動使用時間戳記
- 建議查看 `HYPERPARAMETER_TUNING_GUIDE.md` 了解新工作流程

### v2.0 - 模型配置自動管理

**新增功能**：
- ✅ 訓練時自動保存模型配置
- ✅ 預測時自動載入正確參數
- ✅ 配置文件格式化和管理工具

**新增文件**：
- `MODEL_CONFIG_GUIDE.md` - 配置管理指南

## 授權

本專案僅供教育與研究用途。

## 作者

LSTM 台股價格預測系統開發團隊
