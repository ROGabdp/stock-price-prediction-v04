# 超參數調整與基準模型比較指南

## 概述

這份指南說明如何使用改進的超參數調整系統，包括：
- ✅ 自動時間戳記管理
- ✅ 基準模型比較功能
- ✅ 詳細的訓練日誌記錄
- ✅ 完整的比較報告生成

## 完整工作流程

### 步驟 1: 訓練基準模型

首先訓練一個基準模型作為比較基礎：

```bash
python src/cli/train.py \
    --data-file 19980601-20251111-converted.csv \
    --feature-set "Set A" \
    --time-steps 60 \
    --epochs 100 \
    --batch-size 32 \
    --model-name baseline_model
```

**輸出檔案**（自動加上時間戳記）：
```
models/baseline_model_20251116_144357.h5
models/baseline_model_20251116_144357_scaler.pkl
models/baseline_model_20251116_144357_config.json
logs/training_logs/baseline_model_20251116_144357_training_log.csv
```

記下你的基準模型路徑：
```
models/baseline_model_20251116_144357.h5
```

### 步驟 2: 執行超參數調整（與基準模型比較）

使用基準模型進行超參數調整：

```bash
python src/cli/tune.py \
    --data-file 19980601-20251111-converted.csv \
    --max-trials 50 \
    --epochs-per-trial 50 \
    --baseline-model models/baseline_model_20251116_144357.h5 \
    --tuned-model-name best_tuned_model \
    --output-dir models/
```

**重要參數說明**：
- `--baseline-model`: 指定要比較的基準模型路徑
- `--tuned-model-name`: 調整後模型的名稱（會自動加時間戳記）
- `--max-trials`: 超參數搜尋的試驗次數（建議 50-100）
- `--epochs-per-trial`: 每次試驗的訓練週期（建議 30-50）

**輸出檔案**（自動加上時間戳記）：
```
models/best_tuned_model_20251116_153022.h5
models/best_tuned_model_20251116_153022_config.json
logs/tuning_logs/tuning_results_20251116_153022.txt
logs/tuning_logs/comparison_report_20251116_153022.txt
logs/tuning_logs/lstm_stock_tuning/
```

### 步驟 3: 查看比較報告

調整完成後，查看比較報告：

```bash
cat logs/tuning_logs/comparison_report_20251116_153022.txt
```

**比較報告內容範例**：
```
================================================================================
模型比較報告
================================================================================

調整時間: 20251116_153022
基準模型: models/baseline_model_20251116_144357.h5
調整模型: models/best_tuned_model_20251116_153022.h5

================================================================================
效能比較（測試集）
================================================================================

基準模型:
  - 測試損失: 1.2345
  - 測試準確度: 45.67%

調整模型:
  - 測試損失: 1.1234
  - 測試準確度: 48.92%

================================================================================
改善幅度
================================================================================

損失改善: -0.1111 (-9.00%)
準確度改善: 0.0325 (+7.12%)

✅ 調整模型優於基準模型
```

### 步驟 4: 查看詳細調整結果

查看完整的調整過程和前 10 名試驗結果：

```bash
cat logs/tuning_logs/tuning_results_20251116_153022.txt
```

**調整結果內容範例**：
```
================================================================================
超參數調整結果
================================================================================

調整時間: 2025-11-16 15:30:22
基準模型: models/baseline_model_20251116_144357.h5

================================================================================
調整統計
================================================================================

總試驗次數: 50
總執行時間: 3.45 小時
平均每次試驗時間: 4.14 分鐘

================================================================================
最佳模型效能
================================================================================

最佳驗證損失: 1.1234
最佳驗證準確度: 48.92%

================================================================================
最佳超參數配置
================================================================================

  feature_set_id: Set B
  time_steps: 60
  num_layers: 3
  units_per_layer: [128, 96, 64]
  dropout_rate: 0.2
  learning_rate: 0.0005
  batch_size: 64

================================================================================
前 10 名試驗結果
================================================================================

Rank 1 - Trial abc123:
  驗證損失: 1.1234
  驗證準確度: 48.92%
  超參數:
    - feature_set_id: Set B
    - time_steps: 60
    - num_layers: 3
    ...

Rank 2 - Trial def456:
  ...
```

## 進階使用

### 不指定基準模型（僅調整）

如果只想進行超參數調整，不需要與基準模型比較：

```bash
python src/cli/tune.py \
    --data-file 19980601-20251111-converted.csv \
    --max-trials 50 \
    --epochs-per-trial 50
```

這樣不會生成比較報告，但仍會有時間戳記和完整日誌。

### 快速測試（少量試驗）

用少量試驗快速測試調整流程：

```bash
python src/cli/tune.py \
    --data-file 19980601-20251111-converted.csv \
    --max-trials 5 \
    --epochs-per-trial 10 \
    --baseline-model models/baseline_model_20251116_144357.h5
```

### 自訂調整模型名稱

```bash
python src/cli/tune.py \
    --data-file 19980601-20251111-converted.csv \
    --max-trials 50 \
    --epochs-per-trial 50 \
    --baseline-model models/baseline_model_20251116_144357.h5 \
    --tuned-model-name experiment_v2
```

輸出：`models/experiment_v2_20251116_153022.h5`

## ⚠️ 重要：Keras Tuner 緩存機制

### 緩存機制說明

Keras Tuner 會自動保存所有試驗結果到 `logs/tuning_logs/<project_name>/` 目錄。這個緩存機制可以：

✅ **優點**：
- 節省時間：避免重複訓練相同的超參數組合
- 支援續訓：可以中斷後繼續之前的調整
- 增量調整：增加 `--max-trials` 時只訓練新的試驗

⚠️ **注意事項**：
- 如果資料更新或參數改變，**必須清除緩存**
- 忘記清除緩存會導致**瞬間完成**的假象（實際上是讀取舊結果）
- 系統會在檢測到舊數據時自動警告

### 緩存行為

#### 預設行為（`--overwrite` 未設定）

```bash
python src/cli/tune.py \
    --data-file data.csv \
    --max-trials 50
```

**會發生什麼**：
1. 🔍 檢查 `logs/tuning_logs/lstm_stock_tuning/` 是否存在試驗數據
2. ✅ 如果存在，重複使用之前的結果（**不重新訓練**）
3. ⚠️ 系統會顯示警告並倒數 5 秒
4. 💡 如果 `--max-trials` 比之前大，只訓練額外的試驗

**範例情境**：
```
第一次執行: --max-trials 30  → 訓練 30 個試驗（耗時 2 小時）
第二次執行: --max-trials 30  → 瞬間完成（讀取緩存）⚠️
第三次執行: --max-trials 50  → 只訓練 20 個新試驗（耗時約 1.3 小時）
```

#### 清除緩存（`--overwrite` 設定）

```bash
python src/cli/tune.py \
    --data-file data.csv \
    --max-trials 50 \
    --overwrite  # 加上這個參數
```

**會發生什麼**：
1. 🗑️ 刪除 `logs/tuning_logs/lstm_stock_tuning/` 目錄
2. 🔄 重新開始所有試驗
3. ✅ 確保使用最新的資料和參數

### 何時需要使用 `--overwrite`

#### ✅ 必須使用的情況

1. **CSV 資料檔案更新**
   ```bash
   # 資料增加了新的 20 筆記錄
   python src/cli/tune.py \
       --data-file updated_data.csv \
       --max-trials 50 \
       --overwrite  # ← 必須加上
   ```

2. **修改了特徵工程或模型架構**
   ```bash
   # 修改了 src/features/ 或 src/models/ 中的程式碼
   python src/cli/tune.py \
       --data-file data.csv \
       --max-trials 50 \
       --overwrite  # ← 必須加上
   ```

3. **想要完全重新訓練**
   ```bash
   # 對之前的結果不滿意
   python src/cli/tune.py \
       --data-file data.csv \
       --max-trials 50 \
       --overwrite  # ← 必須加上
   ```

#### ❌ 不需要使用的情況

1. **增加試驗次數**（續訓）
   ```bash
   # 第一次: 30 個試驗
   python src/cli/tune.py --max-trials 30

   # 第二次: 增加到 50 個（只訓練額外的 20 個）
   python src/cli/tune.py --max-trials 50  # 不需要 --overwrite
   ```

2. **查看不同的基準模型比較**（已訓練好的模型）
   ```bash
   # 只是換一個基準模型來比較，試驗結果不變
   python src/cli/tune.py \
       --max-trials 50 \
       --baseline-model models/baseline_model_v2.h5  # 不需要 --overwrite
   ```

### 其他清除緩存的方法

#### 方法 1: 使用不同的專案名稱

```bash
# 保留舊的試驗數據，開始新的實驗
python src/cli/tune.py \
    --data-file updated_data.csv \
    --max-trials 50 \
    --project-name lstm_stock_tuning_v2  # 使用新名稱
```

這會在 `logs/tuning_logs/lstm_stock_tuning_v2/` 建立新的目錄。

**優點**：
- ✅ 保留所有歷史實驗
- ✅ 可以比較不同版本的調整結果
- ✅ 適合做實驗追蹤

**建議命名規則**：
```
lstm_stock_tuning          # 預設
lstm_stock_tuning_setB     # 測試 Set B 特徵集
lstm_stock_tuning_v2       # 版本 2
lstm_stock_tuning_20251116 # 依日期命名
```

#### 方法 2: 手動刪除目錄

```bash
# Windows PowerShell
Remove-Item -Recurse -Force logs\tuning_logs\lstm_stock_tuning

# Windows CMD
rmdir /s /q logs\tuning_logs\lstm_stock_tuning

# Linux/Mac
rm -rf logs/tuning_logs/lstm_stock_tuning
```

**優點**：完全控制刪除行為
**缺點**：需要手動操作

### 警告系統

當系統檢測到舊的試驗數據時，會顯示以下警告：

```
⚠️==============================================================================
⚠️ 偵測到先前的調整紀錄: logs\tuning_logs\lstm_stock_tuning
⚠️ 已有 33 個試驗數據
⚠️
⚠️ Keras Tuner 將會：
⚠️   1. 重複使用這些試驗結果（不會重新訓練）
⚠️   2. 如果 max_trials 更大，只訓練額外的試驗
⚠️   3. 這可能導致「瞬間完成」的情況
⚠️
⚠️ 如果要完全重新訓練，請使用以下任一方式：
⚠️   - 加上 --overwrite 參數
⚠️   - 使用不同的 --project-name
⚠️   - 手動刪除目錄: logs\tuning_logs\lstm_stock_tuning
⚠️==============================================================================

⚠️ 將在 5 秒後繼續使用現有數據...
⚠️ 將在 4 秒後繼續使用現有數據...
⚠️ 將在 3 秒後繼續使用現有數據...
⚠️ 將在 2 秒後繼續使用現有數據...
⚠️ 將在 1 秒後繼續使用現有數據...
```

這給你 5 秒鐘的時間來決定是否要中止（按 Ctrl+C）並重新執行加上 `--overwrite` 參數。

### 最佳實踐

1. **資料更新時**
   ```bash
   python src/cli/tune.py --data-file new_data.csv --overwrite
   ```

2. **實驗追蹤時**
   ```bash
   python src/cli/tune.py --project-name experiment_$(date +%Y%m%d)
   ```

3. **續訓時**
   ```bash
   # 第一次
   python src/cli/tune.py --max-trials 30

   # 增加到 50（自動續訓）
   python src/cli/tune.py --max-trials 50
   ```

4. **定期清理**
   ```bash
   # 刪除超過 30 天的舊試驗數據
   # (手動檢查 logs/tuning_logs/ 目錄)
   ```

## 追蹤訓練歷史

### 查看模型配置

每個模型都有對應的配置檔案，記錄了訓練資訊：

```bash
cat models/best_tuned_model_20251116_153022_config.json
```

**配置檔案內容**：
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
    "epochs_trained": 50,
    "tuner_type": "random",
    "max_trials": 50,
    "baseline_model": "models/baseline_model_20251116_144357.h5",
    "tuning_timestamp": "20251116_153022"
  }
}
```

**重要資訊**：
- `baseline_model`: 這個調整模型是與哪個基準模型比較的
- `tuning_timestamp`: 調整的時間戳記
- 所有超參數配置

### 比較多個調整結果

列出所有調整結果並比較：

```bash
# Windows
dir /B logs\tuning_logs\comparison_report_*.txt

# Linux/Mac
ls -1 logs/tuning_logs/comparison_report_*.txt
```

輸出：
```
comparison_report_20251116_153022.txt
comparison_report_20251116_164512.txt
comparison_report_20251117_091548.txt
```

查看每個報告，找出最佳的調整結果。

## 檔案結構說明

調整完成後，你會得到以下檔案結構：

```
stock-price-prediction-v04/
├── models/
│   ├── baseline_model_20251116_144357.h5           # 基準模型
│   ├── baseline_model_20251116_144357_scaler.pkl   # 基準模型縮放器
│   ├── baseline_model_20251116_144357_config.json  # 基準模型配置
│   ├── best_tuned_model_20251116_153022.h5         # 調整模型 #1
│   ├── best_tuned_model_20251116_153022_config.json
│   ├── best_tuned_model_20251116_164512.h5         # 調整模型 #2
│   └── best_tuned_model_20251116_164512_config.json
│
└── logs/
    ├── training_logs/
    │   ├── baseline_model_20251116_144357_training_log.csv
    │   └── ...
    │
    └── tuning_logs/
        ├── tuning_results_20251116_153022.txt      # 調整結果 #1
        ├── comparison_report_20251116_153022.txt   # 比較報告 #1
        ├── tuning_results_20251116_164512.txt      # 調整結果 #2
        ├── comparison_report_20251116_164512.txt   # 比較報告 #2
        └── lstm_stock_tuning/                      # Keras Tuner 試驗紀錄
            ├── trial_001/
            ├── trial_002/
            └── ...
```

## 最佳實踐

### 1. 命名規範

建議使用有意義的模型名稱：

- 基準模型：`baseline_model`、`baseline_v1`、`baseline_setA`
- 調整模型：`tuned_model`、`experiment_v1`、`tuned_setB`

### 2. 保留訓練記錄

定期備份重要的訓練記錄：

```bash
# 備份整個 logs 目錄
cp -r logs/ logs_backup_20251116/

# 或者只備份特定的調整結果
cp logs/tuning_logs/comparison_report_20251116_153022.txt backups/
```

### 3. 資料更新後的工作流程

當你更新 CSV 資料（例如新增 20 筆資料）後：

```bash
# 1. 訓練新的基準模型
python src/cli/train.py \
    --data-file updated_data.csv \
    --feature-set "Set A" \
    --model-name baseline_model_v2

# 2. 使用新基準模型進行調整
python src/cli/tune.py \
    --data-file updated_data.csv \
    --max-trials 50 \
    --baseline-model models/baseline_model_v2_20251117_091548.h5

# 3. 比較新舊模型效能
# 查看比較報告確認新資料是否帶來改善
cat logs/tuning_logs/comparison_report_<timestamp>.txt
```

### 4. 尋找最佳模型

建立一個簡單的腳本來比較所有模型：

```python
import json
from pathlib import Path

# 讀取所有模型配置
models = []
for config_file in Path("models").glob("*_config.json"):
    with open(config_file) as f:
        config = json.load(f)
        models.append(config)

# 按效能排序（需要手動添加測試準確度到配置中）
# 或者解析訓練日誌獲取效能指標
for model in sorted(models, key=lambda x: x.get('test_accuracy', 0), reverse=True):
    print(f"{model['model_path']}: {model.get('test_accuracy', 'N/A')}")
```

## 常見問題

### Q1: 如何知道調整模型是基於哪個基準模型？

**A**: 查看調整模型的配置檔案：

```bash
cat models/best_tuned_model_20251116_153022_config.json
```

在 `additional_params` 中的 `baseline_model` 欄位會顯示基準模型路徑。

### Q2: 調整結果沒有比基準模型好怎麼辦？

**A**: 這是正常的！可能的原因：
1. 試驗次數不夠（增加 `--max-trials`）
2. 每次試驗訓練週期不夠（增加 `--epochs-per-trial`）
3. 基準模型已經很好了
4. 資料量不足以支援更複雜的模型

建議：
- 增加試驗次數到 100-200
- 嘗試不同的 tuner 類型：`--tuner-type bayesian`

### Q3: 可以同時比較多個基準模型嗎？

**A**: 目前一次調整只能指定一個基準模型。如果要比較多個基準模型，需要：
1. 分別對每個基準模型執行調整
2. 手動比較生成的報告

### Q4: 時間戳記格式是什麼？

**A**: 格式為 `YYYYMMDD_HHMMSS`
- 年月日：20251116
- 時分秒：153022
- 完整：20251116_153022

這確保了檔案名稱的唯一性和可排序性。

## 總結

這個改進的超參數調整系統提供了：

1. ✅ **自動版本管理**：所有模型和報告都有時間戳記
2. ✅ **基準模型追蹤**：知道每個調整模型是基於哪個基準
3. ✅ **詳細日誌**：完整記錄訓練過程和比較結果
4. ✅ **易於查找**：所有資訊都有文件記錄，方便後續查找

現在你可以安心地進行多次實驗，所有歷史記錄都會被完整保留！
