# Contract: validator.py (歷史驗證模組)

**Module**: `src/visualization/validator.py`
**Version**: 1.0.0
**Status**: Stable

## 函式: get_actual_data

### 簽名

```python
def get_actual_data(
    df_raw: pd.DataFrame,
    input_date: str,
    prediction_date: str,
    predicted_class_5: int,
    predicted_class_3: str
) -> Optional[Dict[str, Any]]:
    """
    查詢預測日期的實際市場資料並計算驗證結果

    Args:
        df_raw: 原始歷史資料 DataFrame (包含 'date' 和 'close' 欄位)
        input_date: 輸入日期 "YYYY-MM-DD"
        prediction_date: 預測日期 "YYYY-MM-DD"
        predicted_class_5: 5分類預測類別 (0-4)
        predicted_class_3: 3分類預測類別 ("看跌"/"震盪"/"看漲")

    Returns:
        Optional[Dict]: 若預測日期存在於歷史資料,回傳 ActualDataDict,否則 None

    Example:
        >>> actual = get_actual_data(df, "2024-02-15", "2024-03-15", 3, "看漲")
        >>> print(actual["is_correct_5class"])
        True
    """
```

### 前置條件

1. `df_raw` 必須包含 'date' (datetime64 或 string) 和 'close' (float) 欄位
2. `input_date` 和 `prediction_date` 必須為有效日期格式
3. `predicted_class_5` 必須在 0-4 範圍內
4. `predicted_class_3` 必須為 "看跌", "震盪", "看漲" 之一

### 後置條件

1. 若回傳 None,表示預測日期不存在於歷史資料
2. 若回傳 Dict,必須包含所有 ActualDataDict 定義的欄位
3. `actual_class_5` 的計算必須與 `CLASS_MAPPING` 100%一致
4. `is_correct_5class` = (predicted_class_5 == actual_class_5)
5. `is_correct_3class` = (predicted_class_3 == actual_class_3)

### 錯誤處理

- 資料不存在: 回傳 None (不拋出例外)
- 資料格式錯誤: 記錄警告並回傳 None
- 計算錯誤: 拋出 RuntimeError

### 效能保證

- 時間複雜度: O(log n) (DataFrame 索引查詢)
- 執行時間: < 5 ms (單次查詢)

---

## 函式: calculate_actual_class_5

### 簽名

```python
def calculate_actual_class_5(actual_change_pct: float) -> int:
    """
    根據實際漲跌幅計算5分類類別

    Args:
        actual_change_pct: 實際漲跌幅百分比 (如 0.032 表示 +3.2%)

    Returns:
        int: 5分類類別 (0-4)

    Example:
        >>> calculate_actual_class_5(0.035)  # +3.5%
        3  # 溫和上漲
    """
```

### 前置條件

1. `actual_change_pct` 為 float 型別
2. 合理範圍: -1.0 <= actual_change_pct <= 1.0

### 後置條件

1. 回傳值必須在 0-4 範圍內
2. 分類邏輯必須與 `src/prediction/predictor.CLASS_MAPPING` 完全一致
3. 邊界值處理: 使用 `<` 而非 `<=` (與 CLASS_MAPPING 一致)

### 分類規則

| actual_change_pct | 回傳值 | 類別 |
|-------------------|--------|------|
| < -0.05 | 0 | 極度下跌 |
| -0.05 <= x < -0.025 | 1 | 溫和下跌 |
| -0.025 <= x <= 0.025 | 2 | 區間震盪 |
| 0.025 < x <= 0.05 | 3 | 溫和上漲 |
| > 0.05 | 4 | 極度上漲 |

---

## 函式: map_5class_to_3class

### 簽名

```python
def map_5class_to_3class(class_5: int) -> str:
    """
    將5分類索引映射為3分類標籤

    Args:
        class_5: 5分類索引 (0-4)

    Returns:
        str: 3分類標籤 ("看跌"/"震盪"/"看漲")

    Example:
        >>> map_5class_to_3class(0)
        '看跌'
        >>> map_5class_to_3class(3)
        '看漲'
    """
```

### 映射規則

- 0, 1 → "看跌"
- 2 → "震盪"
- 3, 4 → "看漲"

### 錯誤處理

- 若 `class_5` 不在 0-4 範圍: 拋出 ValueError

---

## 模組版本歷史

| 版本 | 日期 | 變更內容 |
|------|------|---------|
| 1.0.0 | 2025-11-16 | 初始版本 |
