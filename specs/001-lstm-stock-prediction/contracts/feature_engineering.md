# 特徵工程模組合約

## 模組職責

負責從原始股價資料產生衍生特徵、實作三組特徵集 (Set A/B/C)、並進行特徵縮放。

## 公開介面

### 1. engineer_features()

**功能**: 執行完整特徵工程流程

```python
def engineer_features(
    df: pd.DataFrame,
    feature_set_id: str
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    執行特徵工程並返回特徵矩陣與目標變數

    參數:
        df: 原始資料框架
        feature_set_id: 特徵集 ID ("Set A", "Set B", "Set C")

    返回:
        (features, target, scaler): 特徵矩陣、目標變數、縮放器

    拋出:
        ValueError: 特徵集 ID 不合法
        FeatureEngineeringError: 特徵計算失敗
    """
```

**輸入**:
- `df`: `pd.DataFrame`, 包含原始股價資料
- `feature_set_id`: `"Set A"`, `"Set B"`, 或 `"Set C"`

**輸出**:
- `features`: `np.ndarray`, 形狀 `(n_samples, n_features)`
  - Set A: (n, ~12-15)
  - Set B: (n, ~12-15)
  - Set C: (n, ~18-22)
- `target`: `np.ndarray`, 形狀 `(n_samples, 5)`, One-Hot 編碼
- `scaler`: `StandardScaler`, 已 fit 的縮放器

---

### 2. get_feature_set_config()

**功能**: 獲取特徵集配置

```python
def get_feature_set_config(feature_set_id: str) -> Dict[str, Any]:
    """
    獲取特徵集定義配置

    參數:
        feature_set_id: 特徵集 ID

    返回:
        dict: 包含 name, features, exclude 的配置字典

    拋出:
        ValueError: 特徵集 ID 不存在
    """
```

**輸出範例 (Set A)**:
```python
{
    "id": "Set A",
    "name": "動能型",
    "features": [
        "price_returns",    # open/high/low/close 報酬率
        "volume_change",    # 成交量變化率
        "sma_ratio",        # SMA20/SMA60 比值
        "sma_slope",        # SMA20 斜率
        "macd_features",    # DIF12-26, MACD9
        "institutional"     # 法人籌碼及變化率
    ],
    "exclude": ["kd_features", "osc"]
}
```

---

### 3. calculate_target_variable()

**功能**: 計算 20 日報酬率目標變數

```python
def calculate_target_variable(
    df: pd.DataFrame,
    forward_days: int = 20
) -> np.ndarray:
    """
    計算未來 N 日報酬率並轉換為 One-Hot 編碼

    參數:
        df: 包含 'close' 欄位的資料框架
        forward_days: 未來天數 (預設 20)

    返回:
        np.ndarray: One-Hot 編碼的目標變數, 形狀 (n_samples, 5)

    拋出:
        ValueError: 資料量不足以計算目標變數
    """
```

**類別閾值**:
- 0 (極度下跌): R < -5.0%
- 1 (溫和下跌): -5.0% ≤ R < -2.5%
- 2 (區間震盪): -2.5% ≤ R ≤ +2.5%
- 3 (溫和上漲): +2.5% < R ≤ +5.0%
- 4 (極度上漲): R > +5.0%

---

## 錯誤處理

| 錯誤類型 | 觸發條件 | 錯誤訊息範例 |
|---------|---------|-------------|
| ValueError | 特徵集 ID 不合法 | "不支援的特徵集: {feature_set_id}" |
| FeatureEngineeringError | 特徵計算失敗 | "計算 {feature_name} 時發生錯誤: {error}" |
| InsufficientDataError | 資料量不足 | "需要至少 {min_samples} 筆資料，目前僅有 {actual_samples} 筆" |

---

## 使用範例

```python
from src.features import feature_engineer

# 執行特徵工程 (Set A)
features, target, scaler = feature_engineer.engineer_features(
    df=train_df,
    feature_set_id="Set A"
)

print(f"特徵矩陣形狀: {features.shape}")  # (4760, 14)
print(f"目標變數形狀: {target.shape}")    # (4760, 5)
print(f"目標變數分佈: {target.sum(axis=0)}")  # 各類別樣本數

# 獲取特徵集配置
config = feature_engineer.get_feature_set_config("Set A")
print(f"特徵集名稱: {config['name']}")  # "動能型"
```
