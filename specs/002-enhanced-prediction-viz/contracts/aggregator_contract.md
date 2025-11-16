# Contract: aggregator.py (3分類聚合模組)

**Module**: `src/visualization/aggregator.py`
**Version**: 1.0.0
**Status**: Stable

## 函式: aggregate_to_3_categories

### 簽名

```python
def aggregate_to_3_categories(prob_5class: np.ndarray) -> Dict[str, float]:
    """
    將5分類機率向量聚合為3分類機率字典

    Args:
        prob_5class (np.ndarray): 長度為5的numpy陣列,順序為:
            [0] 極度下跌機率
            [1] 溫和下跌機率
            [2] 區間震盪機率
            [3] 溫和上漲機率
            [4] 極度上漲機率

    Returns:
        Dict[str, float]: 3分類機率字典,鍵為 "看跌", "震盪", "看漲"

    Raises:
        ValueError: 若 prob_5class 形狀不為 (5,)
        AssertionError: 若機率總和不為 1.0 (容忍誤差 1e-10)

    Example:
        >>> prob_5 = np.array([0.05, 0.15, 0.25, 0.35, 0.20])
        >>> result = aggregate_to_3_categories(prob_5)
        >>> print(result)
        {'看跌': 0.20, '震盪': 0.25, '看漲': 0.55}
    """
```

### 前置條件 (Preconditions)

1. `prob_5class` 必須是 numpy.ndarray 型別
2. `prob_5class.shape` 必須為 (5,)
3. `prob_5class.dtype` 必須為 float64 或可轉換為 float64
4. `np.sum(prob_5class)` 必須在 1.0 ± 1e-10 範圍內

### 後置條件 (Postconditions)

1. 回傳值為 dict,包含正好3個鍵: "看跌", "震盪", "看漲"
2. 所有值均為 float 型別
3. 所有值均 >= 0.0
4. 三個值的總和必須在 1.0 ± 1e-10 範圍內
5. 聚合邏輯滿足:
   - `result["看跌"] = prob_5class[0] + prob_5class[1]`
   - `result["震盪"] = prob_5class[2]`
   - `result["看漲"] = prob_5class[3] + prob_5class[4]`

### 不變量 (Invariants)

- 函式必須是純函式 (pure function),無副作用
- 相同輸入必須產生相同輸出
- 不修改輸入的 `prob_5class` 陣列

### 錯誤處理

| 錯誤情境 | 例外型別 | 錯誤訊息 |
|---------|---------|---------|
| 輸入不是 numpy array | TypeError | "prob_5class 必須是 numpy.ndarray 型別" |
| 輸入形狀不為 (5,) | ValueError | "prob_5class 形狀必須為 (5,),實際為 {actual_shape}" |
| 機率總和不為 1.0 | AssertionError | "5分類機率總和必須為1.0,實際為 {actual_sum}" |

### 效能保證

- 時間複雜度: O(1)
- 空間複雜度: O(1)
- 執行時間: < 1 ms (單次呼叫)

### 測試案例

```python
# 測試案例 1: 正常情況
input = np.array([0.05, 0.15, 0.25, 0.35, 0.20])
expected = {"看跌": 0.20, "震盪": 0.25, "看漲": 0.55}
assert aggregate_to_3_categories(input) == expected

# 測試案例 2: 邊界情況 - 極度下跌100%
input = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
expected = {"看跌": 1.0, "震盪": 0.0, "看漲": 0.0}
assert aggregate_to_3_categories(input) == expected

# 測試案例 3: 邊界情況 - 均勻分佈
input = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
expected = {"看跌": 0.4, "震盪": 0.2, "看漲": 0.4}
assert aggregate_to_3_categories(input) == expected

# 測試案例 4: 錯誤情況 - 形狀錯誤
try:
    aggregate_to_3_categories(np.array([0.5, 0.5]))
    assert False, "應拋出 ValueError"
except ValueError:
    pass
```

---

## 函式: get_predicted_class_3

### 簽名

```python
def get_predicted_class_3(prob_3class: Dict[str, float]) -> str:
    """
    根據3分類機率字典,回傳機率最高的類別

    Args:
        prob_3class (Dict[str, float]): 3分類機率字典,鍵為 "看跌", "震盪", "看漲"

    Returns:
        str: 機率最高的類別名稱 ("看跌", "震盪", "看漲" 之一)

    Raises:
        ValueError: 若字典缺少必要的鍵
        ValueError: 若所有機率值都為 0.0

    Example:
        >>> prob_3 = {"看跌": 0.20, "震盪": 0.25, "看漲": 0.55}
        >>> get_predicted_class_3(prob_3)
        '看漲'
    """
```

### 前置條件

1. `prob_3class` 必須包含 "看跌", "震盪", "看漲" 三個鍵
2. 所有值必須為 float 或可轉換為 float
3. 至少有一個值 > 0.0

### 後置條件

1. 回傳值必須是 "看跌", "震盪", "看漲" 之一
2. 回傳值對應的機率必須是三者中最大的
3. 若多個類別機率相同且最高,回傳順序優先級: 看跌 > 震盪 > 看漲

### 錯誤處理

| 錯誤情境 | 例外型別 | 錯誤訊息 |
|---------|---------|---------|
| 缺少必要鍵 | ValueError | "prob_3class 必須包含 '看跌', '震盪', '看漲' 三個鍵" |
| 所有機率為0 | ValueError | "所有3分類機率不可同時為0" |

### 效能保證

- 時間複雜度: O(1) (僅3個鍵)
- 執行時間: < 0.1 ms

---

## 模組版本歷史

| 版本 | 日期 | 變更內容 |
|------|------|---------|
| 1.0.0 | 2025-11-16 | 初始版本,定義 aggregate_to_3_categories 和 get_predicted_class_3 |
