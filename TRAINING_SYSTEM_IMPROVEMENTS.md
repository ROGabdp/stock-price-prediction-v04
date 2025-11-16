# 訓練系統改進總結

## 概述

本次更新為 LSTM 台股預測系統加入了完整的版本管理和模型追蹤功能，解決了模型覆蓋和追蹤困難的問題。

## 改進內容

### 1. 基準模型訓練（時間戳記功能）

#### 改進的檔案
- `src/models/model_builder.py`
- `src/cli/train.py`

#### 新增功能
- ✅ 自動時間戳記：每次訓練自動加上 `YYYYMMDD_HHMMSS` 格式的時間戳記
- ✅ 防止覆蓋：多次訓練產生不同檔案，所有歷史版本都保留
- ✅ 自動配對：模型和 scaler 使用相同的時間戳記
- ✅ 訓練記錄：在訓練歷史中保存實際模型名稱

#### 使用範例
```bash
# 訓練基準模型
python src/cli/train.py \
    --data-file data.csv \
    --feature-set "Set A" \
    --model-name baseline_model

# 輸出（自動加上時間戳記）
# models/baseline_model_20251116_144357.h5
# models/baseline_model_20251116_144357_scaler.pkl
```

#### 相關文件
- `TIMESTAMP_FEATURE.md` - 時間戳記功能詳細說明
- `test_timestamp.py` - 功能測試腳本

---

### 2. 超參數調整（基準模型比較功能）

#### 改進的檔案
- `src/cli/tune.py`
- `src/tuning/hyperparameter_tuner.py`

#### 新增功能
- ✅ 時間戳記管理：調整模型也自動加上時間戳記
- ✅ 基準模型比較：可指定基準模型進行效能比較
- ✅ 基準模型追蹤：記錄調整模型是基於哪個基準模型
- ✅ 詳細日誌：記錄調整過程、前 10 名試驗結果
- ✅ 比較報告：自動生成詳細的比較報告

#### 新增參數
```bash
--baseline-model    # 基準模型路徑（用於比較）
--tuned-model-name  # 調整模型名稱（會自動加時間戳記）
```

#### 使用範例
```bash
# 超參數調整並與基準模型比較
python src/cli/tune.py \
    --data-file data.csv \
    --max-trials 50 \
    --baseline-model models/baseline_model_20251116_144357.h5 \
    --tuned-model-name best_tuned_model

# 輸出
# models/best_tuned_model_20251116_153022.h5
# models/best_tuned_model_20251116_153022_config.json
# logs/tuning_logs/tuning_results_20251116_153022.txt
# logs/tuning_logs/comparison_report_20251116_153022.txt
```

#### 相關文件
- `HYPERPARAMETER_TUNING_GUIDE.md` - 超參數調整完整指南

---

### 3. 模型配置追蹤

#### 改進功能
模型配置檔案（`*_config.json`）現在包含：
- `baseline_model`: 基準模型路徑（調整模型專用）
- `tuning_timestamp`: 調整時間戳記
- `model_name`: 實際的模型名稱（含時間戳記）
- 完整的超參數配置

#### 配置範例
```json
{
  "model_path": "models/best_tuned_model_20251116_153022.h5",
  "feature_set_id": "Set B",
  "time_steps": 60,
  "created_at": "2025-11-16 15:30:22",
  "additional_params": {
    "num_layers": 3,
    "dropout_rate": 0.2,
    "learning_rate": 0.0005,
    "batch_size": 64,
    "baseline_model": "models/baseline_model_20251116_144357.h5",
    "tuning_timestamp": "20251116_153022"
  }
}
```

---

### 4. 日誌和報告系統

#### 訓練日誌
位置：`logs/training_logs/`
- 訓練過程 CSV 日誌
- 包含每個 epoch 的損失和準確度

#### 調整結果報告
位置：`logs/tuning_logs/tuning_results_<timestamp>.txt`

內容：
- 調整時間和基準模型資訊
- 總試驗次數和執行時間
- 最佳模型效能指標
- 最佳超參數配置
- 前 10 名試驗結果詳情

#### 比較報告
位置：`logs/tuning_logs/comparison_report_<timestamp>.txt`

內容：
- 基準模型和調整模型的效能對比
- 改善幅度（損失和準確度）
- 是否優於基準模型的判斷

---

## 完整工作流程

### 場景：資料更新後重新訓練和調整

```bash
# 1. 訓練新的基準模型（使用更新後的資料）
python src/cli/train.py \
    --data-file updated_data.csv \
    --feature-set "Set A" \
    --model-name baseline_model

# 記下輸出的模型路徑
# 例如：models/baseline_model_20251116_144357.h5

# 2. 執行超參數調整（與新基準模型比較）
python src/cli/tune.py \
    --data-file updated_data.csv \
    --max-trials 50 \
    --epochs-per-trial 50 \
    --baseline-model models/baseline_model_20251116_144357.h5 \
    --tuned-model-name best_tuned_model

# 3. 查看比較報告
cat logs/tuning_logs/comparison_report_20251116_153022.txt

# 4. 查看詳細調整結果
cat logs/tuning_logs/tuning_results_20251116_153022.txt

# 5. 查看模型配置（確認使用的基準模型）
cat models/best_tuned_model_20251116_153022_config.json
```

### 自動化腳本
提供了兩個自動化腳本來執行完整流程：
- `example_workflow.sh` - Linux/Mac 版本
- `example_workflow.bat` - Windows 版本

使用方式：
```bash
# Linux/Mac
bash example_workflow.sh

# Windows
example_workflow.bat
```

---

## 檔案結構

```
stock-price-prediction-v04/
├── models/
│   ├── baseline_model_20251116_144357.h5           # 基準模型
│   ├── baseline_model_20251116_144357_scaler.pkl
│   ├── baseline_model_20251116_144357_config.json
│   ├── best_tuned_model_20251116_153022.h5         # 調整模型
│   └── best_tuned_model_20251116_153022_config.json
│
├── logs/
│   ├── training_logs/
│   │   └── baseline_model_20251116_144357_training_log.csv
│   └── tuning_logs/
│       ├── tuning_results_20251116_153022.txt
│       ├── comparison_report_20251116_153022.txt
│       └── lstm_stock_tuning/                      # Keras Tuner 紀錄
│
├── src/
│   ├── cli/
│   │   ├── train.py                                # ✅ 已更新
│   │   └── tune.py                                 # ✅ 已更新
│   ├── models/
│   │   └── model_builder.py                        # ✅ 已更新
│   └── tuning/
│       └── hyperparameter_tuner.py                 # ✅ 已更新
│
├── TIMESTAMP_FEATURE.md                            # ✅ 新增
├── HYPERPARAMETER_TUNING_GUIDE.md                  # ✅ 新增
├── TRAINING_SYSTEM_IMPROVEMENTS.md                 # ✅ 新增（本文件）
├── test_timestamp.py                               # ✅ 新增
├── example_workflow.sh                             # ✅ 新增
└── example_workflow.bat                            # ✅ 新增
```

---

## 主要優勢

### 1. 完整的版本追蹤
- 每個模型都有唯一的時間戳記
- 永遠不會覆蓋舊模型
- 可以輕鬆回溯到任何歷史版本

### 2. 清晰的模型關係
- 調整模型明確記錄基於哪個基準模型
- 配置檔案包含完整的訓練資訊
- 比較報告提供詳細的效能對比

### 3. 便於實驗管理
- 可以同時維護多個實驗版本
- 所有日誌和報告都有時間戳記
- 方便比較不同時間點的訓練結果

### 4. 生產就緒
- 詳細的文件說明
- 自動化的工作流程腳本
- 完整的測試和驗證

---

## 測試驗證

### 時間戳記功能測試
```bash
python test_timestamp.py
```

測試項目：
- ✅ 時間戳記格式正確 (YYYYMMDD_HHMMSS)
- ✅ 模型和縮放器檔案路徑正確
- ✅ 多次訓練不會互相覆蓋
- ✅ add_timestamp 參數可正常控制

### 完整流程測試
```bash
# 快速測試（少量試驗）
bash example_workflow.sh
# 或
example_workflow.bat
```

---

## 向後相容性

### 保持相容的功能
1. **add_timestamp 參數**：`train_model()` 函數可設定 `add_timestamp=False` 使用舊行為
2. **tune.py 參數**：所有舊參數仍然有效
3. **不指定基準模型**：`tune.py` 可不使用 `--baseline-model` 參數

### 遷移建議
如果你有舊的訓練腳本：
1. 基本上不需要修改，新功能預設啟用
2. 如果需要舊行為，在程式碼中設定 `add_timestamp=False`
3. 建議更新到新的工作流程以獲得完整功能

---

## 常見問題解答

### Q: 時間戳記會影響效能嗎？
A: 不會。時間戳記只是字串操作，對訓練效能沒有影響。

### Q: 可以自訂時間戳記格式嗎？
A: 目前格式固定為 `YYYYMMDD_HHMMSS`。如需修改，請編輯相關程式碼中的 `strftime` 格式。

### Q: 舊模型會自動刪除嗎？
A: 不會。所有模型都會保留，需要手動管理磁碟空間。

### Q: 如何找到最新的模型？
A: 使用檔案系統的排序功能：
```bash
# Linux/Mac
ls -t models/baseline_model_*.h5 | head -1

# Windows PowerShell
Get-ChildItem models\baseline_model_*.h5 | Sort-Object LastWriteTime -Descending | Select-Object -First 1
```

### Q: 比較報告顯示調整模型未優於基準模型怎麼辦？
A: 這是正常的，建議：
1. 增加試驗次數（`--max-trials`）
2. 增加每次試驗的訓練週期（`--epochs-per-trial`）
3. 嘗試不同的 tuner 類型（`--tuner-type bayesian`）
4. 檢查資料品質和數量

---

## 未來改進方向

1. **模型管理 CLI**
   - 列出所有模型及其效能
   - 自動清理舊模型
   - 模型效能排行榜

2. **視覺化儀表板**
   - 訓練過程可視化
   - 模型比較圖表
   - 效能趨勢分析

3. **自動化實驗追蹤**
   - 整合 MLflow 或 Weights & Biases
   - 自動記錄實驗參數和結果
   - 實驗版本管理

4. **智能基準模型選擇**
   - 自動選擇最佳的基準模型
   - 多基準模型同時比較
   - 基準模型推薦系統

---

## 貢獻者

本次改進由 Claude Code (Anthropic) 協助完成。

---

## 授權

遵循專案主要授權協議。

---

## 更新日誌

### 2025-11-16
- ✅ 加入基準模型訓練時間戳記功能
- ✅ 加入超參數調整時間戳記功能
- ✅ 加入基準模型比較功能
- ✅ 加入詳細的訓練日誌記錄
- ✅ 建立完整的使用文件
- ✅ 建立自動化工作流程腳本
- ✅ 建立測試驗證腳本
