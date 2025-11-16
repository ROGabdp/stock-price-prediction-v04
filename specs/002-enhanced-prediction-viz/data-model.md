# Data Model: Enhanced Stock Price Prediction Visualization

**Feature**: 002-enhanced-prediction-viz
**Date**: 2025-11-16
**Purpose**: 定義系統中的資料實體、欄位、關係與驗證規則

## 核心實體

### 1. PredictionResult (預測結果)

**用途**: 封裝單一日期的完整預測結果,包含5分類機率、3分類聚合、歷史驗證資訊

**欄位**:
| 欄位名稱 | 型別 | 必填 | 說明 | 驗證規則 |
|---------|------|------|------|---------|
| input_date | str | 是 | 輸入日期 (格式: YYYY-MM-DD) | 必須是有效日期格式 |
| prediction_date | str | 是 | 預測日期 (input_date + 20交易日) | 必須是有效日期格式 |
| prob_5class | np.ndarray | 是 | 5分類機率向量 [極度下跌, 溫和下跌, 震盪, 溫和上漲, 極度上漲] | shape=(5,), dtype=float64, sum=1.0±1e-10 |
| prob_3class | Dict[str, float] | 是 | 3分類聚合機率 {"看跌": float, "震盪": float, "看漲": float} | 所有值>=0, sum=1.0±1e-10 |
| predicted_class_5 | int | 是 | 5分類預測類別 (0-4) | 0 <= value <= 4 |
| predicted_class_3 | str | 是 | 3分類預測類別 | 值必須為 "看跌", "震盪", "看漲" 之一 |
| confidence | float | 是 | 預測信心度 (預測類別的機率值) | 0.0 <= value <= 1.0 |
| actual_data | Optional[ActualData] | 否 | 實際市場資料 (若預測日期已過) | 若存在,必須完整包含所有欄位 |

**狀態**: 無狀態轉換 (唯讀資料結構)

**關係**:
- `actual_data`: 0..1 關聯到 `ActualData` 實體

**業務規則**:
- `prob_5class` 的總和必須精確等於 1.0 (容忍誤差 1e-10)
- `prob_3class` 的計算必須滿足: 看跌 = prob_5class[0] + prob_5class[1], 震盪 = prob_5class[2], 看漲 = prob_5class[3] + prob_5class[4]
- `predicted_class_5` 必須是 `prob_5class` 中機率最高的索引
- `predicted_class_3` 必須是 `prob_3class` 中機率最高的類別
- `confidence` 必須等於預測類別對應的機率值

---

### 2. ActualData (實際市場資料)

**用途**: 封裝預測日期的實際股價資料,用於驗證預測準確度

**欄位**:
| 欄位名稱 | 型別 | 必填 | 說明 | 驗證規則 |
|---------|------|------|------|---------|
| actual_close | float | 是 | 實際收盤價 | value > 0 |
| actual_change_pct | float | 是 | 實際漲跌幅百分比 (如 0.032 表示 +3.2%) | -1.0 <= value <= 1.0 (極端情況) |
| actual_class_5 | int | 是 | 實際5分類類別 (0-4) | 0 <= value <= 4 |
| actual_class_3 | str | 是 | 實際3分類類別 | 值必須為 "看跌", "震盪", "看漲" 之一 |
| is_correct_5class | bool | 是 | 5分類預測是否正確 | True 或 False |
| is_correct_3class | bool | 是 | 3分類預測是否正確 | True 或 False |

**狀態**: 無狀態轉換 (唯讀資料結構)

**關係**:
- 被 `PredictionResult.actual_data` 引用

**業務規則**:
- `actual_class_5` 必須根據 `actual_change_pct` 與 `CLASS_MAPPING` 邊界計算
- `actual_class_3` 必須根據 `actual_class_5` 映射: [0,1]→看跌, [2]→震盪, [3,4]→看漲
- `is_correct_5class` = (predicted_class_5 == actual_class_5)
- `is_correct_3class` = (predicted_class_3 == actual_class_3)

---

### 3. VisualizationConfig (視覺化配置)

**用途**: 封裝視覺化圖表的樣式配置,確保輸出一致性與可讀性

**欄位**:
| 欄位名稱 | 型別 | 必填 | 說明 | 預設值 |
|---------|------|------|------|--------|
| figure_size | Tuple[int, int] | 是 | 圖表尺寸 (寬, 高) in inches | (12, 10) |
| dpi | int | 是 | 圖片解析度 | 150 |
| font_family | str | 是 | 中文字型家族 | "Microsoft JhengHei" |
| font_size_title | int | 是 | 標題字型大小 | 14 |
| font_size_label | int | 是 | 標籤字型大小 | 11 |
| colors_3class | Dict[str, str] | 是 | 3分類顏色映射 | {"看跌": "#DC143C", "震盪": "#808080", "看漲": "#228B22"} |
| colors_5class | List[str] | 是 | 5分類顏色列表 | ["#B22222", "#FF6347", "#A9A9A9", "#32CD32", "#006400"] |
| hatches_3class | Dict[str, str] | 是 | 3分類紋理映射 (色盲輔助) | {"看跌": "///", "震盪": "", "看漲": "\\\\\\"} |
| bar_height | float | 是 | 橫條高度 | 0.6 |
| show_percentage | bool | 是 | 是否在橫條上顯示百分比 | True |

**狀態**: 無狀態轉換 (配置物件)

**關係**: 無

**業務規則**:
- `dpi` 必須 >= 100 以確保規格 SC-005 (清晰可讀)
- `colors_5class` 長度必須為 5
- 顏色值必須是有效的 matplotlib 顏色格式

---

## 資料流程

```
[CSV 歷史資料]
    ↓ (data_loader.load_csv_data)
[DataFrame]
    ↓ (predictor.predict_for_date)
[5分類機率向量]
    ↓
[aggregator.aggregate_to_3_categories] ───→ [prob_3class]
    ↓
[validator.get_actual_data] ───→ [ActualData] (若日期存在於歷史資料)
    ↓
[PredictionResult] (封裝完整結果)
    ↓
[plotter.plot_dual_view] ───→ [PNG 圖片檔案]
```

---

## 分類映射規則

### 5分類 → 3分類映射表

| 5分類索引 | 5分類標籤 | 漲跌幅範圍 | 3分類標籤 |
|----------|----------|-----------|----------|
| 0 | 極度下跌 | < -5.0% | 看跌 |
| 1 | 溫和下跌 | -5.0% ~ -2.5% | 看跌 |
| 2 | 區間震盪 | -2.5% ~ +2.5% | 震盪 |
| 3 | 溫和上漲 | +2.5% ~ +5.0% | 看漲 |
| 4 | 極度上漲 | > +5.0% | 看漲 |

**重要**: 此映射表必須與 `src/prediction/predictor.py` 中的 `CLASS_MAPPING` 常數100%一致。

---

## 驗證規則摘要

### 機率總和驗證
```python
# 5分類機率總和
assert abs(np.sum(prob_5class) - 1.0) < 1e-10, "5分類機率總和必須為1.0"

# 3分類機率總和
total_3class = prob_3class["看跌"] + prob_3class["震盪"] + prob_3class["看漲"]
assert abs(total_3class - 1.0) < 1e-10, "3分類機率總和必須為1.0"
```

### 聚合一致性驗證
```python
# 3分類聚合必須匹配5分類
assert abs(prob_3class["看跌"] - (prob_5class[0] + prob_5class[1])) < 1e-10
assert abs(prob_3class["震盪"] - prob_5class[2]) < 1e-10
assert abs(prob_3class["看漲"] - (prob_5class[3] + prob_5class[4])) < 1e-10
```

### 分類邊界驗證
```python
# 實際類別計算必須與 CLASS_MAPPING 一致
from src.prediction.predictor import CLASS_MAPPING

def verify_actual_class(actual_change_pct: float, actual_class: int) -> bool:
    """驗證實際類別是否正確"""
    info = CLASS_MAPPING[actual_class]
    min_c = info["min_change"] if info["min_change"] is not None else float('-inf')
    max_c = info["max_change"] if info["max_change"] is not None else float('inf')
    return min_c <= actual_change_pct < max_c
```

---

## 型別定義 (TypedDict)

```python
from typing import TypedDict, Optional
import numpy as np

class ActualDataDict(TypedDict):
    """實際市場資料的型別定義"""
    actual_close: float
    actual_change_pct: float
    actual_class_5: int
    actual_class_3: str
    is_correct_5class: bool
    is_correct_3class: bool

class PredictionResultDict(TypedDict):
    """預測結果的型別定義"""
    input_date: str
    prediction_date: str
    prob_5class: np.ndarray
    prob_3class: dict[str, float]
    predicted_class_5: int
    predicted_class_3: str
    confidence: float
    actual_data: Optional[ActualDataDict]

class VisualizationConfigDict(TypedDict):
    """視覺化配置的型別定義"""
    figure_size: tuple[int, int]
    dpi: int
    font_family: str
    font_size_title: int
    font_size_label: int
    colors_3class: dict[str, str]
    colors_5class: list[str]
    hatches_3class: dict[str, str]
    bar_height: float
    show_percentage: bool
```

---

## 實體關係圖 (ER Diagram)

```
┌─────────────────────────────┐
│   PredictionResult          │
├─────────────────────────────┤
│ + input_date: str           │
│ + prediction_date: str      │
│ + prob_5class: ndarray      │
│ + prob_3class: Dict         │
│ + predicted_class_5: int    │
│ + predicted_class_3: str    │
│ + confidence: float         │
│ + actual_data: ActualData?  │◄────┐
└─────────────────────────────┘     │
                                    │ 0..1
                                    │
                         ┌──────────┴──────────────┐
                         │   ActualData            │
                         ├─────────────────────────┤
                         │ + actual_close: float   │
                         │ + actual_change_pct: f  │
                         │ + actual_class_5: int   │
                         │ + actual_class_3: str   │
                         │ + is_correct_5class: b  │
                         │ + is_correct_3class: b  │
                         └─────────────────────────┘

┌─────────────────────────────┐
│   VisualizationConfig       │  (獨立配置)
├─────────────────────────────┤
│ + figure_size: Tuple        │
│ + dpi: int                  │
│ + colors_3class: Dict       │
│ + colors_5class: List       │
│ + hatches_3class: Dict      │
│ + ...                       │
└─────────────────────────────┘
```

---

## 資料來源與目的地

| 實體 | 資料來源 | 資料目的地 |
|------|---------|-----------|
| PredictionResult | `src/prediction/predictor.predict_for_date()` 回傳 + 聚合器/驗證器擴充 | `plotter.plot_dual_view()` 輸入 |
| ActualData | CSV 歷史資料 (透過 `validator.get_actual_data()` 查詢) | `PredictionResult.actual_data` 欄位 |
| VisualizationConfig | 程式碼內建預設值或使用者參數 | `plotter.plot_dual_view()` 配置參數 |

---

## 範例資料

```python
# 範例: PredictionResult (包含實際資料的情況)
result = {
    "input_date": "2024-02-15",
    "prediction_date": "2024-03-15",
    "prob_5class": np.array([0.05, 0.15, 0.25, 0.35, 0.20]),
    "prob_3class": {
        "看跌": 0.20,   # 0.05 + 0.15
        "震盪": 0.25,   # 0.25
        "看漲": 0.55    # 0.35 + 0.20
    },
    "predicted_class_5": 3,  # 溫和上漲 (機率最高 35%)
    "predicted_class_3": "看漲",
    "confidence": 0.35,
    "actual_data": {
        "actual_close": 18500.0,
        "actual_change_pct": 0.032,  # +3.2%
        "actual_class_5": 3,  # 溫和上漲 (+2.5% ~ +5.0%)
        "actual_class_3": "看漲",
        "is_correct_5class": True,   # 預測正確
        "is_correct_3class": True    # 預測正確
    }
}
```

---

## 資料模型版本

**版本**: 1.0.0
**最後更新**: 2025-11-16
**變更歷史**: 初始版本
