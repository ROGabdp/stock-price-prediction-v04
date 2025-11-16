# 資料前處理模組合約

## 模組職責

負責從 CSV 檔案載入原始股價資料，進行資料驗證、清理，並分割為訓練/驗證/測試集。

## 公開介面

### 1. load_csv_data()

**功能**: 載入並驗證 CSV 檔案

```python
def load_csv_data(file_path: str) -> pd.DataFrame:
    """
    載入台股歷史資料 CSV 檔案並進行驗證

    參數:
        file_path: CSV 檔案路徑 (絕對路徑)

    返回:
        pd.DataFrame: 載入且驗證通過的資料框架

    拋出:
        FileNotFoundError: 檔案不存在
        ValueError: 欄位缺失或資料型別錯誤
        DataValidationError: 資料驗證失敗
    """
```

**輸入**:
- `file_path`: `str`, 如 `"D:/000-github-repositories/.../19980601-20251111-converted.csv"`

**輸出**:
- `pd.DataFrame`: 包含 23 個欄位，約 6,800 筆資料
- 欄位: date, open, high, low, close, volume, SMA系列, MA系列, DIF12-26, MACD9, OSC, K(9,3), D(9,3), 籌碼系列

**驗證規則**:
- ✅ 所有必填欄位存在
- ✅ 資料型別正確 (date 轉 datetime64, 其他為 float64)
- ✅ 數值範圍合理 (價格 > 0, K/D 在 [0, 1])

---

### 2. split_time_series()

**功能**: 時間序列分割為訓練/驗證/測試集

```python
def split_time_series(
    data: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    按時間順序分割資料集

    參數:
        data: 完整資料框架
        train_ratio: 訓練集比例 (預設 0.7)
        val_ratio: 驗證集比例 (預設 0.15)

    返回:
        (train_df, val_df, test_df): 三個資料框架

    拋出:
        ValueError: 比例總和超過 1.0 或資料量不足
    """
```

**輸入**:
- `data`: `pd.DataFrame`, 約 6,800 筆
- `train_ratio`: 0.7
- `val_ratio`: 0.15

**輸出**:
- `train_df`: 約 4,760 筆 (70%)
- `val_df`: 約 1,020 筆 (15%)
- `test_df`: 約 1,020 筆 (15%)

**保證**:
- ✅ train_df 時間範圍在 val_df 之前
- ✅ val_df 時間範圍在 test_df 之前
- ✅ 無資料重疊

---

## 錯誤處理

| 錯誤類型 | 觸發條件 | 錯誤訊息範例 |
|---------|---------|-------------|
| FileNotFoundError | CSV 檔案不存在 | "找不到檔案: {file_path}" |
| ValueError | 欄位缺失 | "缺少必要欄位: {missing_columns}" |
| DataValidationError | 數值範圍錯誤 | "日期 {date} 的 K 值超出 [0, 1] 範圍" |

---

## 使用範例

```python
from src.data import data_loader, data_splitter

# 載入資料
file_path = "D:/000-github-repositories/.../19980601-20251111-converted.csv"
df = data_loader.load_csv_data(file_path)
print(f"✅ 載入 {len(df)} 筆資料")

# 分割資料集
train_df, val_df, test_df = data_splitter.split_time_series(df)
print(f"訓練集: {len(train_df)} 筆")
print(f"驗證集: {len(val_df)} 筆")
print(f"測試集: {len(test_df)} 筆")
```
