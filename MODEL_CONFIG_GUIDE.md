# 模型配置自動載入指南

## 📋 功能說明

從現在開始,當您使用 `tune.py` 訓練模型時,系統會自動保存模型的配置參數到 JSON 文件。在使用 `predict.py` 進行預測時,系統會自動讀取這些配置,您**不再需要手動指定** `--feature-set` 和 `--time-steps` 參數!

## ✨ 主要優點

1. **防止配置錯誤**: 自動使用訓練時的正確參數
2. **簡化預測命令**: 不需要記住每個模型的配置
3. **可追溯性**: 配置文件記錄了模型的訓練參數

## 🎯 使用方式

### 1. 訓練模型時自動保存配置

當您執行超參數調整時:

```bash
python src/cli/tune.py \
    --data-file 19980601-20251111-converted.csv \
    --max-trials 50 \
    --epochs-per-trial 50 \
    --tuner-type bayesian \
    --output-dir models/
```

系統會自動創建兩個文件:
- `models/best_tuned_model.h5` - 模型文件
- `models/best_tuned_model_config.json` - **配置文件(新增)**

### 2. 預測時自動載入配置

**新方式(推薦)** - 不需要指定 feature-set 和 time-steps:

```bash
python src/cli/predict.py \
    --model-file models/best_tuned_model.h5 \
    --data-file 19980601-20251111-converted.csv \
    --input-date "2024-01-15"
```

系統會自動:
1. 尋找 `models/best_tuned_model_config.json`
2. 讀取配置中的 `feature_set_id` 和 `time_steps`
3. 使用這些參數進行預測

**舊方式(仍然支援)** - 手動指定參數:

```bash
python src/cli/predict.py \
    --model-file models/best_tuned_model.h5 \
    --data-file 19980601-20251111-converted.csv \
    --input-date "2024-01-15" \
    --feature-set "Set B" \
    --time-steps 60
```

如果您明確指定了參數,系統會使用您指定的值而不是配置文件中的值。

## 📄 配置文件格式

配置文件是 JSON 格式,包含模型的所有重要參數:

```json
{
  "model_file": "models/best_tuned_model.h5",
  "feature_set_id": "Set B",
  "time_steps": 60,
  "created_at": "2025-11-16 11:45:32",
  "num_layers": 3,
  "dropout_rate": 0.2,
  "learning_rate": 0.0005,
  "batch_size": 64,
  "epochs_trained": 50,
  "tuner_type": "bayesian",
  "max_trials": 50
}
```

### 關鍵參數說明

| 參數 | 說明 | 預測時使用 |
|------|------|-----------|
| `feature_set_id` | 特徵集 (Set A/B/C) | ✅ 是 |
| `time_steps` | 時間窗口大小 | ✅ 是 |
| `num_layers` | LSTM 層數 | ❌ 否(僅供參考) |
| `dropout_rate` | Dropout 比率 | ❌ 否(僅供參考) |
| `learning_rate` | 學習率 | ❌ 否(僅供參考) |
| `batch_size` | 批次大小 | ❌ 否(僅供參考) |
| `epochs_trained` | 訓練週期數 | ❌ 否(僅供參考) |
| `created_at` | 創建時間 | ❌ 否(僅供參考) |

## 🔍 查看模型配置

### 方法 1: 直接查看 JSON 文件

```bash
cat models/best_tuned_model_config.json
```

### 方法 2: 使用 Python

```python
import json

with open("models/best_tuned_model_config.json", "r") as f:
    config = json.load(f)

print(f"特徵集: {config['feature_set_id']}")
print(f"時間窗口: {config['time_steps']}")
```

### 方法 3: 使用工具模組

```python
from src.utils.model_config import load_model_config

config = load_model_config("models/best_tuned_model.h5")
if config:
    print(f"特徵集: {config['feature_set_id']}")
    print(f"時間窗口: {config['time_steps']}")
```

## ⚠️ 注意事項

### 1. 舊模型的配置文件

如果您有舊的模型文件(如 `baseline_model.h5`)沒有配置文件,預測時會出現警告:

```
⚠️ 未找到模型配置文件,使用命令行參數
   請確認 --feature-set 和 --time-steps 與模型訓練時一致
   當前使用: feature_set=Set A, time_steps=60
```

**解決方案**: 手動創建配置文件或使用命令行參數指定。

我們已經為 `baseline_model.h5` 創建了配置文件 `baseline_model_config.json`。

### 2. 配置文件命名規則

配置文件名稱必須遵循以下規則:
- 模型文件: `{name}.h5`
- 配置文件: `{name}_config.json`

例如:
- `best_tuned_model.h5` → `best_tuned_model_config.json`
- `baseline_model.h5` → `baseline_model_config.json`
- `my_model.h5` → `my_model_config.json`

### 3. 手動覆蓋配置

如果您需要用不同的參數測試模型(例如測試模型對不同特徵集的泛化能力),可以手動指定參數:

```bash
python src/cli/predict.py \
    --model-file models/best_tuned_model.h5 \
    --data-file data.csv \
    --input-date "2024-01-15" \
    --feature-set "Set C" \
    --time-steps 80
```

這種情況下,系統會使用您指定的值而不是配置文件中的值。

## 📚 實際範例

### 範例 1: 完整工作流程

```bash
# 1. 執行超參數調整(自動保存配置)
python src/cli/tune.py \
    --data-file 19980601-20251111-converted.csv \
    --max-trials 20 \
    --epochs-per-trial 50 \
    --tuner-type bayesian

# 2. 查看生成的配置
cat models/best_tuned_model_config.json

# 3. 使用模型預測(自動載入配置)
python src/cli/predict.py \
    --model-file models/best_tuned_model.h5 \
    --data-file 19980601-20251111-converted.csv \
    --input-date "2024-01-15"
```

### 範例 2: 批次預測多個日期

```bash
python src/cli/predict.py \
    --model-file models/best_tuned_model.h5 \
    --data-file 19980601-20251111-converted.csv \
    --input-date "2024-01-15" "2024-02-20" "2024-03-10"
```

所有預測都會使用配置文件中的參數。

### 範例 3: 檢查配置並預測

```bash
# 查看模型使用的配置
cat models/best_tuned_model_config.json

# 確認後執行預測
python src/cli/predict.py \
    --model-file models/best_tuned_model.h5 \
    --data-file 19980601-20251111-converted.csv \
    --input-date "2024-01-15"
```

## 🔧 技術細節

### 配置保存機制

在 `src/cli/tune.py` 中,當保存最佳模型時:

```python
from src.utils.model_config import save_model_config

# 保存模型
best_model.save(str(model_file))

# 保存配置
save_model_config(
    model_path=str(model_file),
    feature_set_id=best_hps.get("feature_set_id"),
    time_steps=best_hps.get("time_steps"),
    additional_params={...}
)
```

### 配置載入機制

在 `src/cli/predict.py` 中,載入模型後:

```python
from src.utils.model_config import load_model_config

# 載入配置
model_config = load_model_config(args.model_file)

# 如果配置存在且用戶未指定參數,使用配置中的值
if model_config and args.feature_set == default_value:
    args.feature_set = model_config.get("feature_set_id")
```

## 🎉 總結

這個新功能讓預測變得更簡單、更安全:

✅ **自動化**: 訓練時自動保存,預測時自動載入
✅ **防錯**: 避免使用錯誤的特徵集或時間窗口
✅ **靈活**: 仍然支援手動指定參數
✅ **可追溯**: 配置文件記錄了所有訓練參數

現在您只需要記住模型文件路徑,其他參數都會自動處理!
