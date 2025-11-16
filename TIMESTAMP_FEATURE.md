# 模型時間戳記功能說明

## 概述

為了避免重新訓練模型時覆蓋舊模型，現在訓練系統會自動在模型名稱中加入時間戳記。

## 功能特點

### 1. 自動時間戳記
- 每次訓練時，系統會自動在模型名稱後加入當前時間戳記
- 時間戳記格式：`YYYYMMDD_HHMMSS`（例如：`20251116_144357`）
- 預設行為：**啟用時間戳記**（可透過參數關閉）

### 2. 檔案命名規則

#### 訓練前
```
模型名稱: baseline_model
```

#### 訓練後（自動加入時間戳記）
```
模型檔案: models/baseline_model_20251116_144357.h5
縮放器檔案: models/baseline_model_20251116_144357_scaler.pkl
訓練日誌: logs/training_logs/baseline_model_20251116_144357_training_log.csv
```

### 3. 多次訓練範例

如果你在不同時間進行多次訓練，會產生不同的模型檔案：

```
第一次訓練 (2025-11-16 14:43:57):
├─ models/baseline_model_20251116_144357.h5
└─ models/baseline_model_20251116_144357_scaler.pkl

第二次訓練 (2025-11-16 15:30:22):
├─ models/baseline_model_20251116_153022.h5
└─ models/baseline_model_20251116_153022_scaler.pkl

第三次訓練 (2025-11-17 09:15:48):
├─ models/baseline_model_20251117_091548.h5
└─ models/baseline_model_20251117_091548_scaler.pkl
```

所有模型都會被保留，不會互相覆蓋！

## 使用方式

### 預設使用（推薦）

正常執行訓練指令，系統會自動加入時間戳記：

```bash
python src/cli/train.py \
    --data-file 19980601-20251111-converted.csv \
    --feature-set "Set A" \
    --time-steps 60 \
    --epochs 100 \
    --batch-size 32 \
    --model-name baseline_model
```

結果：
- 模型會儲存為 `models/baseline_model_20251116_144357.h5`
- 縮放器會儲存為 `models/baseline_model_20251116_144357_scaler.pkl`

### 進階使用：在程式碼中控制

如果你在寫自己的訓練腳本，可以透過 `add_timestamp` 參數控制：

```python
from src.models.model_builder import train_model

# 啟用時間戳記（預設）
model, history = train_model(
    model, X_train, y_train, X_val, y_val,
    epochs=100,
    batch_size=32,
    model_name="baseline_model",
    add_timestamp=True  # 預設為 True
)
# 結果: baseline_model_20251116_144357.h5

# 關閉時間戳記（使用原始名稱）
model, history = train_model(
    model, X_train, y_train, X_val, y_val,
    epochs=100,
    batch_size=32,
    model_name="baseline_model",
    add_timestamp=False  # 明確設為 False
)
# 結果: baseline_model.h5
```

## 取得實際的模型名稱

訓練完成後，可以從訓練歷史中取得實際使用的模型名稱：

```python
model, history = train_model(...)

# 從 history 取得帶時間戳記的模型名稱
model_name_with_timestamp = history["model_name"]
print(f"模型已儲存為: {model_name_with_timestamp}.h5")
```

## 修改的檔案

### 1. `src/models/model_builder.py`
- 新增 `from datetime import datetime`
- `train_model()` 函數新增 `add_timestamp` 參數（預設 `True`）
- 自動生成時間戳記並附加到模型名稱
- 將實際模型名稱儲存在 `history["model_name"]` 中

### 2. `src/cli/train.py`
- 訓練完成後從 `history["model_name"]` 取得實際模型名稱
- 使用相同的時間戳記儲存縮放器（`scaler`）
- 更新日誌輸出以顯示正確的檔案路徑

### 3. 新增測試檔案
- `test_timestamp.py` - 驗證時間戳記功能的測試腳本

## 測試

執行測試腳本來驗證功能：

```bash
python test_timestamp.py
```

測試項目：
1. ✅ 時間戳記格式正確 (YYYYMMDD_HHMMSS)
2. ✅ 模型和縮放器檔案路徑正確
3. ✅ 多次訓練不會互相覆蓋
4. ✅ add_timestamp 參數可正常控制

## 優點

1. **防止覆蓋**：每次訓練產生獨立的模型檔案
2. **版本追蹤**：可以輕鬆比較不同時間訓練的模型
3. **向後相容**：可透過 `add_timestamp=False` 使用舊行為
4. **自動配對**：模型和縮放器使用相同的時間戳記，易於管理
5. **時間可讀**：時間戳記格式清晰，可快速識別訓練時間

## 注意事項

1. 如果你在同一秒內執行多次訓練，可能會產生相同的時間戳記（機率很低）
2. 舊的模型檔案不會自動刪除，需要手動管理磁碟空間
3. 載入模型時，需要使用完整的檔案名稱（包含時間戳記）

## 範例：載入帶時間戳記的模型

```python
from src.models.model_builder import load_model
from src.features.scalers import load_scaler

# 指定完整的檔案名稱（包含時間戳記）
model_name = "baseline_model_20251116_144357"

# 載入模型和縮放器
model = load_model(f"models/{model_name}.h5")
scaler = load_scaler(f"models/{model_name}_scaler.pkl")
```

## 建議的工作流程

1. **訓練新模型**：使用預設設定，自動產生帶時間戳記的檔案
2. **評估效果**：比較不同訓練時間的模型表現
3. **選擇最佳模型**：找出表現最好的模型檔案
4. **部署使用**：使用完整檔名載入最佳模型進行預測

## 總結

現在你可以放心地更新 CSV 資料並重新訓練模型，不用擔心覆蓋舊模型！每次訓練都會產生新的模型檔案，方便你進行版本管理和效果比較。
