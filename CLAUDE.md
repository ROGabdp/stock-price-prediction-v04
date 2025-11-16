# CLAUDE.md

此檔案為 Claude Code (claude.ai/code) 在此儲存庫中工作時提供指引。

## 專案概述

基於 LSTM 的台股價格預測系統，具備自動超參數調整與版本管理功能。預測未來 20 個交易日的價格走勢（5 類分類），支援可配置的特徵集。

## 環境設置

**關鍵**：必須在 WSL2 Ubuntu 環境中使用 Python 3.12 工作。

```bash
# 從 Windows 啟動
wsl -d Ubuntu_D
cd /mnt/d/000-github-repositories/stock-price-prediction-v04

# 啟動虛擬環境
source venv/bin/activate

# 驗證環境
python --version  # 必須是 3.12.x
```

## 常用指令

### 訓練指令

```bash
# 訓練基準模型（產生帶時間戳記的輸出檔案）
python src/cli/train.py \
    --data-file 19980601-20251111-converted.csv \
    --feature-set "Set A" \
    --time-steps 60 \
    --epochs 100 \
    --batch-size 32 \
    --model-name baseline_model

# 輸出：models/baseline_model_YYYYMMDD_HHMMSS.h5 (+ _config.json, _scaler.pkl)
```

### 超參數調整指令

**重要**：Keras Tuner 會將試驗結果緩存在 `logs/tuning_logs/<project_name>/`。當資料更新時必須使用 `--overwrite`，或使用不同的 `--project-name`。

```bash
# 完整調整並與基準模型比較（資料更新時務必使用 --overwrite）
python src/cli/tune.py \
    --data-file 19980601-20251111-converted.csv \
    --max-trials 50 \
    --epochs-per-trial 50 \
    --baseline-model models/baseline_model_20251116_144357.h5 \
    --tuned-model-name best_tuned_model \
    --overwrite  # 資料更新時必須加上！

# 快速測試（5 次試驗）
python src/cli/tune.py \
    --data-file 19980601-20251111-converted.csv \
    --max-trials 5 \
    --epochs-per-trial 10 \
    --overwrite
```

### 預測指令

```bash
# 使用訓練好的模型預測（自動載入配置）
python src/cli/predict.py \
    --model-file models/best_tuned_model_20251116_153022.h5 \
    --data-file 19980601-20251111-converted.csv \
    --input-date "2024-01-15"
```

### 測試指令

```bash
# 執行所有測試
pytest tests/ -v

# 執行特定測試
pytest tests/test_data_loader.py -v

# 測試含覆蓋率報告
pytest tests/ --cov=src --cov-report=html

# 驗證完整流程
python validate_pipeline.py

# 測試時間戳記功能
python test_timestamp.py

# 測試檔案命名一致性
python test_file_naming.py
```

## 架構概述

### 三層式結構

1. **CLI 層** (`src/cli/`): 使用者介面指令
   - `train.py`: 基準模型訓練，自動加上時間戳記
   - `tune.py`: 超參數搜尋，含緩存警告系統
   - `predict.py`: 日期預測，自動載入配置

2. **核心邏輯層** (`src/`):
   - `data/`: CSV 載入、時間序列分割、序列建立
   - `features/`: 特徵工程，內建 3 種特徵集
   - `models/`: LSTM 架構（基準模型 + 動態建構器）
   - `tuning/`: Keras Tuner 整合與基準模型比較
   - `prediction/`: 未來價格預測邏輯
   - `utils/`: GPU 偵測、日誌記錄、配置管理

3. **配置層**: JSON 格式的模型配置
   - 訓練時自動儲存：`{model_name}_config.json`
   - 預測時自動載入
   - 追蹤：feature_set_id, time_steps, 超參數, baseline_model

### 關鍵設計模式

#### 1. 自動時間戳記 (v2.1)
所有模型/配置/縮放器都會加上 `YYYYMMDD_HHMMSS` 後綴以防止覆蓋：
- `src/models/model_builder.py` 中的 `train_model()` 加上時間戳記
- 在 `history["model_name"]` 中回傳實際的模型名稱
- `train.py` 使用此名稱來配對 config/scaler

#### 2. Keras Tuner 緩存管理
**位置**: `logs/tuning_logs/<project_name>/trial_*/`

**關鍵警告系統** (在 `tune.py` 第 233-262 行):
- 開始前偵測現有試驗
- 顯示 5 秒倒數警告
- 防止因緩存結果導致的「瞬間完成」混淆

**何時必須使用 `--overwrite`**:
- CSV 資料更新（新增記錄）
- 特徵工程修改
- 模型架構變更
- 想要全新訓練（非續訓）

**何時不需要使用 `--overwrite`**:
- 增加 `--max-trials`（繼續之前的執行）
- 僅更改基準模型比較目標

#### 3. 特徵集系統
在 `src/features/feature_sets.py` 中預定義三種特徵集：
- **Set A** (12 個特徵): 動能型（SMA、MACD、法人買賣）
- **Set B** (12 個特徵): 震盪型（KD、法人買賣）
- **Set C** (15 個特徵): 全特徵組合

調整時，Keras Tuner 會自動搜尋所有三種特徵集。

#### 4. 模型配置配對
每個 `.h5` 模型都有相同時間戳記的配對檔案：
```
baseline_model_20251116_144357.h5         # 模型權重
baseline_model_20251116_144357_config.json # 超參數配置
baseline_model_20251116_144357_scaler.pkl  # 特徵縮放器
```

配置檔案包含 `baseline_model` 欄位（針對調整模型）以追蹤來源。

#### 5. 基準模型比較系統
使用 `--baseline-model` 進行調整時：
1. 載入基準模型進行評估
2. 使用 Keras Tuner 訓練調整模型
3. 在測試集上比較兩者
4. 生成 `comparison_report_<timestamp>.txt`
5. 在調整模型的配置中記錄基準模型路徑

### 資料流程

```
CSV → load_csv_data() → prepare_features_and_target() → split_time_series()
  → fit_scaler() → create_sequences() → train_model() → 儲存（帶時間戳記）
```

**關鍵細節**: Scaler 僅在訓練資料上 fit，然後套用到驗證/測試集。

### 時間序列分割
**關鍵**: 使用時間順序分割（非隨機）:
- 70% 訓練、15% 驗證、15% 測試
- 實作在 `src/data/data_splitter.py`
- 使用滑動窗口建立序列（可配置 time_steps: 20/40/60/80）

## 檔案命名規範

**模型**: `{name}_{timestamp}.h5`
**配置**: `{name}_{timestamp}_config.json`
**縮放器**: `{name}_{timestamp}_scaler.pkl`
**訓練日誌**: `{name}_{timestamp}_training_log.csv`
**調整結果**: `tuning_results_{timestamp}.txt`
**比較報告**: `comparison_report_{timestamp}.txt`

## 重要實作注意事項

### 新增特徵時
1. 更新 `src/features/feature_sets.py` 中的 `FEATURE_SETS`
2. 修改 `src/features/feature_engineer.py` 中的 `prepare_features_and_target()`
3. **必須使用 `--overwrite`** 執行 tune.py 以使用更新後的特徵
4. 在相關特徵集描述中記錄

### 修改模型架構時
1. 編輯 `src/models/` 中的 `build_lstm_model()` 或 `build_dynamic_lstm_model()`
2. 更新 `src/tuning/hyperparameter_tuner.py` 中的超參數搜尋空間
3. **必須使用 `--overwrite`** 重新執行調整
4. 若配置格式改變，更新 `MODEL_CONFIG_GUIDE.md`

### 進行預測時
- 預測會自動從配置載入 feature_set_id 和 time_steps
- 需要匹配的 scaler 檔案 (`{model_name}_scaler.pkl`)
- 輸入日期必須存在於 CSV 中以建立序列
- 回傳 5 類機率分佈 + 預測收盤價

### GPU 使用
- TensorFlow 2.16.1 包含自動 CUDA 支援
- GPU 偵測在 `src/utils/gpu_checker.py`（自動降級至 CPU）
- 批次大小：GPU 32-64、CPU 16-32
- 訓練時間：GPU ~1-2 小時、CPU ~8-12 小時（100 epochs 基準模型）

## 關鍵文件

- **README.md**: 使用者設置與使用指南
- **TIMESTAMP_FEATURE.md**: 時間戳記系統細節
- **HYPERPARAMETER_TUNING_GUIDE.md**: 完整調整工作流程與緩存說明
- **TRAINING_SYSTEM_IMPROVEMENTS.md**: v2.1 功能摘要
- **MODEL_CONFIG_GUIDE.md**: 配置檔案格式與自動載入

## 常見陷阱

1. **忘記 `--overwrite`**: 導致從緩存試驗「瞬間完成」
   - **修正**: 加上 `--overwrite` 或使用新的 `--project-name`
   - **偵測**: tune.py 會顯示自動警告並 5 秒倒數

2. **scaler/model 時間戳記不匹配**: 預測時無法載入 scaler
   - **修正**: 確保 scaler 和 model 共用相同時間戳記
   - **預防**: 使用 `history["model_name"]` 取得模型名稱

3. **預測時錯誤的特徵集**: 模型配置中有 feature_set_id
   - **修正**: 從配置自動載入，無需手動指定
   - **驗證**: 檢查 `{model}_config.json` 確認正確的 feature_set_id

4. **時間序列資料洩漏**: 不要隨機打亂資料
   - **修正**: 永遠使用 `split_time_series()`（時間順序分割）
   - **驗證**: 檢查日誌中的訓練/驗證/測試日期範圍

5. **Windows 中的 UTF-8 編碼問題**: 控制台無法顯示中文
   - **修正**: 測試腳本中已使用 `io.TextIOWrapper` 處理
   - **模式**: `sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')`

## 版本歷史

- **v2.1** (2025-11-16): 新增時間戳記版本管理、基準模型比較、緩存警告系統
- **v2.0**: 新增自動配置管理
- **v1.0**: 初始 LSTM 實作含 Keras Tuner
