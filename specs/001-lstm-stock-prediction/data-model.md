# 資料模型：LSTM 台股價格預測系統

**Phase**: 1 (設計與合約)
**日期**: 2025-11-15
**目的**: 定義系統中的核心資料實體、欄位、關係與驗證規則

## 資料模型概覽

本系統包含以下核心資料實體：
1. **RawStockData** (原始股價資料)
2. **ProcessedFeatures** (處理後的特徵矩陣)
3. **FeatureSet** (特徵集配置)
4. **TrainingDataset** (訓練資料集)
5. **LSTMModel** (LSTM 模型)
6. **HyperparameterTrial** (超參數試驗紀錄)
7. **PredictionResult** (預測結果)

---

## 實體 1: RawStockData (原始股價資料)

**描述**: 從 CSV 檔案載入的原始台股歷史資料

### 欄位

| 欄位名稱 | 資料型別 | 必填 | 說明 | 驗證規則 |
|---------|---------|------|------|---------|
| date | datetime64 | ✅ | 交易日期 | 格式: YYYY/M/D, 不可有缺失值 |
| open | float64 | ✅ | 開盤價 | > 0 |
| high | float64 | ✅ | 最高價 | ≥ open, ≥ low, ≥ close |
| low | float64 | ✅ | 最低價 | > 0, ≤ open, ≤ close |
| close | float64 | ✅ | 收盤價 | > 0 |
| volume | float64 | ✅ | 成交量（億） | ≥ 0 |
| SMA5, SMA10, SMA20, SMA60, SMA120, SMA240 | float64 | ✅ | 簡單移動平均線 | > 0 |
| MA5, MA10 | float64 | ✅ | 移動平均線 | > 0 |
| DIF12-26 | float64 | ✅ | MACD DIF 值 | 可為負值 |
| MACD9 | float64 | ✅ | MACD 訊號線 | 可為負值 |
| OSC | float64 | ✅ | MACD 柱狀圖 | 可為負值 |
| K(9,3) | float64 | ✅ | KD 指標 K 值 | 0 ≤ K ≤ 1 |
| D(9,3) | float64 | ✅ | KD 指標 D 值 | 0 ≤ D ≤ 1 |
| net buy sell | float64 | ✅ | 法人淨買賣超（元） | 可為負值 |
| cumulative net buy sell | float64 | ✅ | 法人累積淨買賣超（元） | 可為負值 |
| buy | float64 | ✅ | 法人買超（元） | ≥ 0 |
| sell | float64 | ✅ | 法人賣超（元） | ≥ 0 |

### 關係
- **無直接關係**，為系統輸入資料源
- 經過特徵工程後轉換為 `ProcessedFeatures`

### 狀態轉換
```
CSV 檔案 → 載入驗證 → RawStockData (DataFrame)
           ↓ (失敗)
         錯誤訊息 (欄位缺失、資料型別錯誤、驗證失敗)
```

---

## 實體 2: ProcessedFeatures (處理後的特徵矩陣)

**描述**: 經過特徵工程與縮放後的特徵矩陣

### 欄位

| 欄位名稱 | 資料型別 | 必填 | 說明 | 驗證規則 |
|---------|---------|------|------|---------|
| date | datetime64 | ✅ | 交易日期（索引） | 與 RawStockData 對應 |
| feature_set_id | str | ✅ | 特徵集 ID (Set A/B/C) | 枚舉值: "Set A", "Set B", "Set C" |
| features | ndarray | ✅ | 特徵矩陣 | 形狀: (n_samples, n_features) |
| target | ndarray | ✅ | 目標變數 (One-Hot) | 形狀: (n_samples, 5), 值為 0 或 1 |
| scaler | StandardScaler | ✅ | 特徵縮放器 | 已 fit 的 scaler 物件 |

### 特徵集內容

#### Set A (動能型) - 約 12-15 個特徵
- 價格變化率: `open_return`, `high_return`, `low_return`, `close_return` (4 個)
- Volume 變化率: `volume_change` (1 個)
- SMA 比例: `sma20_sma60_ratio`, `sma20_slope` (2 個)
- MACD: `DIF12-26`, `MACD9` (2 個)
- 法人籌碼: `net_buy_sell`, `cumulative_net_buy_sell`, `net_buy_sell_change` (3 個)

#### Set B (震盪型) - 約 12-15 個特徵
- 價格變化率: `open_return`, `high_return`, `low_return`, `close_return` (4 個)
- Volume 變化率: `volume_change` (1 個)
- SMA 比例: `sma20_sma60_ratio`, `sma20_slope` (2 個)
- KD 指標: `K(9,3)`, `D(9,3)` (2 個)
- 法人籌碼: `net_buy_sell`, `cumulative_net_buy_sell`, `net_buy_sell_change` (3 個)

#### Set C (全特徵集) - 約 18-22 個特徵
- 所有 Set A + Set B 特徵（去重後）
- 額外包含: `OSC`, 其他可能的衍生特徵

### 關係
- 由 `RawStockData` 透過特徵工程產生
- 輸入至 `TrainingDataset` 進行時間窗口切割

### 狀態轉換
```
RawStockData → 特徵工程 → ProcessedFeatures (特定 feature_set_id)
                ↓
              時間窗口切割 → TrainingDataset
```

---

## 實體 3: FeatureSet (特徵集配置)

**描述**: 特徵集定義配置（Set A/B/C）

### 欄位

| 欄位名稱 | 資料型別 | 必填 | 說明 | 驗證規則 |
|---------|---------|------|------|---------|
| id | str | ✅ | 特徵集 ID | 枚舉值: "Set A", "Set B", "Set C" |
| name | str | ✅ | 特徵集名稱 | 如："動能型", "震盪型", "全特徵集" |
| features | list[str] | ✅ | 包含的特徵清單 | 非空清單 |
| exclude | list[str] | ✅ | 排除的特徵清單 | 可為空清單 |

### 關係
- 定義 `ProcessedFeatures` 的特徵選擇邏輯
- 用於超參數調整時的特徵集選擇

---

## 實體 4: TrainingDataset (訓練資料集)

**描述**: 經過時間窗口切割後的訓練/驗證/測試資料集

### 欄位

| 欄位名稱 | 資料型別 | 必填 | 說明 | 驗證規則 |
|---------|---------|------|------|---------|
| X_train | ndarray | ✅ | 訓練集輸入 | 形狀: (n_train, time_steps, n_features) |
| y_train | ndarray | ✅ | 訓練集目標 | 形狀: (n_train, 5) |
| X_val | ndarray | ✅ | 驗證集輸入 | 形狀: (n_val, time_steps, n_features) |
| y_val | ndarray | ✅ | 驗證集目標 | 形狀: (n_val, 5) |
| X_test | ndarray | ✅ | 測試集輸入 | 形狀: (n_test, time_steps, n_features) |
| y_test | ndarray | ✅ | 測試集目標 | 形狀: (n_test, 5) |
| time_steps | int | ✅ | 時間窗口大小 | 枚舉值: 20, 40, 60, 80 |
| feature_set_id | str | ✅ | 使用的特徵集 | 枚舉值: "Set A", "Set B", "Set C" |

### 驗證規則
- 時間序列分割: `train_end < val_start < val_end < test_start`
- 無資料洩漏: 訓練集不包含驗證/測試集的未來資訊
- 分割比例: 約 70% / 15% / 15%

### 關係
- 由 `ProcessedFeatures` 經過時間窗口切割產生
- 輸入至 `LSTMModel` 進行訓練

---

## 實體 5: LSTMModel (LSTM 模型)

**描述**: LSTM 深度學習模型實例

### 欄位

| 欄位名稱 | 資料型別 | 必填 | 說明 | 驗證規則 |
|---------|---------|------|------|---------|
| model | tf.keras.Model | ✅ | Keras 模型物件 | Sequential or Functional API |
| architecture | dict | ✅ | 模型架構配置 | 包含層數、單元數、dropout |
| time_steps | int | ✅ | 輸入時間窗口 | 20, 40, 60, 80 |
| n_features | int | ✅ | 輸入特徵數量 | 12-22 (依特徵集) |
| n_classes | int | ✅ | 輸出類別數 | 固定為 5 |
| optimizer | str | ✅ | 優化器 | "adam" |
| learning_rate | float | ✅ | 學習率 | 0.001, 0.0005, 0.0001 |
| loss | str | ✅ | 損失函數 | "categorical_crossentropy" |

### 架構配置範例 (基準模型)
```python
{
    "layers": [
        {"type": "LSTM", "units": 128, "return_sequences": True},
        {"type": "Dropout", "rate": 0.2},
        {"type": "LSTM", "units": 64, "return_sequences": True},
        {"type": "Dropout", "rate": 0.2},
        {"type": "LSTM", "units": 32, "return_sequences": False},
        {"type": "Dropout", "rate": 0.2},
        {"type": "Dense", "units": 5, "activation": "softmax"}
    ]
}
```

### 關係
- 使用 `TrainingDataset` 進行訓練
- 產生 `HyperparameterTrial` 紀錄
- 用於 `PredictionResult` 預測

### 狀態轉換
```
未訓練 → 訓練中 → 已訓練 (儲存為 .h5)
        ↓ (Early Stopping)
      訓練終止 (Patience 達到)
```

---

## 實體 6: HyperparameterTrial (超參數試驗紀錄)

**描述**: 每次超參數調整試驗的紀錄

### 欄位

| 欄位名稱 | 資料型別 | 必填 | 說明 | 驗證規則 |
|---------|---------|------|------|---------|
| trial_id | str | ✅ | 試驗唯一 ID | UUID 格式 |
| time_steps | int | ✅ | 時間窗口 | 20, 40, 60, 80 |
| num_layers | int | ✅ | LSTM 層數 | 2, 3, 4 |
| units_per_layer | list[int] | ✅ | 每層單元數 | 如 [128, 64, 32] |
| dropout_rate | float | ✅ | Dropout 比率 | 0.1 ≤ rate ≤ 0.4 |
| learning_rate | float | ✅ | 學習率 | 0.001, 0.0005, 0.0001 |
| batch_size | int | ✅ | 批次大小 | 32, 64, 128 |
| feature_set_id | str | ✅ | 特徵集 ID | "Set A", "Set B", "Set C" |
| val_loss | float | ✅ | 驗證損失 | > 0 |
| val_accuracy | float | ✅ | 驗證準確度 | 0 ≤ acc ≤ 1 |
| training_time | float | ✅ | 訓練時間（秒） | > 0 |
| timestamp | datetime64 | ✅ | 試驗時間戳記 | ISO 8601 格式 |

### 關係
- 由 `LSTMModel` 訓練過程產生
- 用於選擇最佳模型（min `val_loss`）

---

## 實體 7: PredictionResult (預測結果)

**描述**: 模型預測的輸出結果

### 欄位

| 欄位名稱 | 資料型別 | 必填 | 說明 | 驗證規則 |
|---------|---------|------|------|---------|
| input_date | datetime64 | ✅ | 輸入日期 | 必須存在於歷史資料中 |
| prediction_date | datetime64 | ✅ | 預測日期 (input_date + 20 交易日) | 計算得出 |
| probability_vector | ndarray | ✅ | 5 類機率向量 | 形狀: (5,), 總和 = 1.0 |
| predicted_class | int | ✅ | 預測類別 | 0, 1, 2, 3, 4 |
| confidence | float | ✅ | 預測信心度 | max(probability_vector) |
| predicted_price_range | dict | ✅ | 預測收盤價區間 | {"min": float, "max": float} |
| actual_close_price | float | ❌ | 實際收盤價 | 若超過歷史資料則為 None |
| timestamp | datetime64 | ✅ | 預測時間戳記 | 預測執行時間 |

### 類別映射
| predicted_class | 漲跌幅區間 | 說明 |
|----------------|----------|------|
| 0 | R < -5.0% | 極度下跌 |
| 1 | -5.0% ≤ R < -2.5% | 溫和下跌 |
| 2 | -2.5% ≤ R ≤ +2.5% | 區間震盪 |
| 3 | +2.5% < R ≤ +5.0% | 溫和上漲 |
| 4 | R > +5.0% | 極度上漲 |

### 關係
- 由 `LSTMModel` 產生
- 使用 `ProcessedFeatures` 作為輸入

### 狀態轉換
```
輸入日期 → 載入歷史資料 → 特徵工程 → 模型預測 → PredictionResult
                                          ↓
                                    查詢實際收盤價 (若存在)
```

---

## 資料流程圖

```
CSV 檔案 (19980601-20251111-converted.csv)
    ↓
[載入與驗證]
    ↓
RawStockData (DataFrame, 6,800 筆)
    ↓
[特徵工程] → FeatureSet (Set A/B/C 配置)
    ↓
ProcessedFeatures (n_samples, n_features)
    ↓
[時間窗口切割 + 資料集分割]
    ↓
TrainingDataset (X_train, y_train, X_val, y_val, X_test, y_test)
    ↓
[LSTM 模型訓練]
    ↓
LSTMModel (已訓練) ← HyperparameterTrial (試驗紀錄)
    ↓
[預測]
    ↓
PredictionResult (預測類別 + 信心度 + 價格區間)
```

---

## 資料驗證總結

### 關鍵驗證點
1. **CSV 載入**: 欄位完整性、資料型別、數值範圍
2. **特徵工程**: 特徵矩陣形狀、無 NaN/Inf 值、縮放後範圍正確
3. **資料集分割**: 時間序列順序、無資料洩漏、分割比例正確
4. **模型訓練**: 輸入形狀匹配、損失函數收斂、驗證指標合理
5. **預測輸出**: 機率總和為 1、類別範圍正確、日期計算正確

### 錯誤處理策略
- **資料載入失敗**: 記錄錯誤訊息，中止流程
- **特徵工程錯誤**: 記錄哪個特徵計算失敗，提供預設值或跳過
- **模型訓練失敗**: 記錄試驗紀錄，繼續下一次試驗
- **預測失敗**: 返回錯誤訊息，不儲存結果

下一階段將建立 **模組介面合約 (contracts/)**。
