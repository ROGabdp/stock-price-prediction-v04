# Research: Enhanced Stock Price Prediction Visualization

**Feature**: 002-enhanced-prediction-viz
**Date**: 2025-11-16
**Purpose**: 解決技術決策疑問,為實作計畫提供明確方向

## 研究主題

### 1. matplotlib 橫條圖最佳實踐

**決策**: 使用 `matplotlib.pyplot.barh()` 繪製水平橫條圖

**理由**:
- 水平橫條圖 (barh) 更適合顯示類別機率比較,標籤可讀性佳
- matplotlib 是 Python 科學計算標準庫,與現有專案技術堆疊一致
- 支援精確控制起點對齊 (left parameter) 以滿足規格 FR-006
- 內建支援中文字型設定 (透過 `plt.rcParams['font.sans-serif']`)
- 可輸出 PNG/PDF 等多種格式,符合規格 FR-013

**替代方案考慮**:
- **seaborn**: 功能更豐富但引入額外依賴,違反「禁止過度設計」原則
- **plotly**: 互動式圖表但規格明確要求靜態圖片輸出
- **純文字 ASCII 橫條圖**: 無法滿足規格 SC-005 (1920x1080 解析度清晰可讀)

**實作細節**:
```python
# 確保所有橫條從 x=0 開始 (左側對齊)
plt.barh(y_positions, probabilities, left=0, ...)
# 設定中文字型
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
```

---

### 2. 分類邊界邏輯一致性策略

**決策**: 從 `src/prediction/predictor.py` 中的 `CLASS_MAPPING` 常數提取分類邊界

**理由**:
- 規格 SC-010 要求「實際類別計算準確度100%」
- 規格假設「分類邊界定義與訓練時一致」
- 現有程式碼已定義 `CLASS_MAPPING` 字典,包含 `min_change` 和 `max_change`
- 使用相同常數可確保訓練、預測、驗證三者100%一致
- 避免硬編碼重複定義,符合 DRY 原則

**替代方案考慮**:
- **重新硬編碼邊界**: 高風險,可能與訓練時不一致
- **從模型配置檔載入**: 現有配置檔不包含分類邊界資訊
- **動態計算邊界**: 不適用,分類邊界是固定的業務邏輯

**實作細節**:
```python
from src.prediction.predictor import CLASS_MAPPING

def calculate_actual_class(actual_change_pct: float) -> int:
    """根據實際漲跌幅計算5分類類別"""
    for class_id, info in CLASS_MAPPING.items():
        min_c = info["min_change"] if info["min_change"] is not None else float('-inf')
        max_c = info["max_change"] if info["max_change"] is not None else float('inf')
        if min_c <= actual_change_pct < max_c:
            return class_id
    # 處理邊界情況 (exact equality)
    # ...
```

---

### 3. 3分類聚合計算精確度保證

**決策**: 使用 numpy 進行浮點數加總,並加入斷言驗證誤差為0

**理由**:
- 規格 SC-003 要求「3分類聚合機率的計算誤差為0」
- numpy 陣列運算已在現有專案中使用,技術堆疊一致
- Python 原生 float 加總可能有微小誤差 (1e-15 等級)
- 使用 `np.isclose()` 或 `abs(sum - 1.0) < 1e-10` 驗證總和為100%

**替代方案考慮**:
- **Python decimal 模組**: 過度設計,numpy float64 精度已足夠
- **手動四捨五入**: 可能引入額外誤差,不符合「誤差為0」要求

**實作細節**:
```python
import numpy as np

def aggregate_to_3_categories(prob_5class: np.ndarray) -> dict:
    """
    將5分類機率聚合為3分類

    Args:
        prob_5class: 長度5的numpy陣列 [極度下跌, 溫和下跌, 震盪, 溫和上漲, 極度上漲]

    Returns:
        dict: {"看跌": float, "震盪": float, "看漲": float}
    """
    bearish = prob_5class[0] + prob_5class[1]  # 極度下跌 + 溫和下跌
    neutral = prob_5class[2]                    # 震盪
    bullish = prob_5class[3] + prob_5class[4]   # 溫和上漲 + 極度上漲

    # 驗證總和為100% (誤差容忍度 1e-10)
    total = bearish + neutral + bullish
    assert abs(total - 1.0) < 1e-10, f"3分類機率總和應為1.0,實際為{total}"

    return {"看跌": bearish, "震盪": neutral, "看漲": bullish}
```

---

### 4. 歷史資料查詢效率優化

**決策**: 使用 pandas DataFrame 的 `loc[]` 索引查詢實際收盤價

**理由**:
- 現有程式碼已使用 pandas 載入 CSV 資料 (`src/data/data_loader.py`)
- DataFrame 的日期索引查詢效能足夠 (O(log n) 二分搜尋)
- 規格 SC-004 要求執行時間增加不超過30%,單次查詢開銷可忽略
- 不需要額外的資料庫或快取機制,符合「禁止過度設計」原則

**替代方案考慮**:
- **SQLite 資料庫**: 過度設計,CSV 檔案已足夠
- **Redis 快取**: 不適用於離線 CLI 工具
- **記憶體快取字典**: 批次預測時可能有幫助,但增加複雜度

**實作細節**:
```python
def get_actual_price(df: pd.DataFrame, target_date: str) -> Optional[float]:
    """
    從歷史資料中查詢實際收盤價

    Args:
        df: 原始歷史資料 DataFrame (已排序)
        target_date: 目標日期字串 "YYYY-MM-DD"

    Returns:
        Optional[float]: 實際收盤價,若不存在則回傳 None
    """
    try:
        row = df.loc[df['date'] == target_date]
        if len(row) == 0:
            return None
        return float(row.iloc[0]['close'])
    except Exception:
        return None
```

---

### 5. 顏色編碼與色盲友善設計

**決策**: 使用紅/灰/綠色系,並加上圖案紋理作為輔助區分

**理由**:
- 規格 FR-015 明確要求「看跌(紅色系)、震盪(灰色系)、看漲(綠色系)」
- 紅綠色盲 (約8%男性) 可能無法區分紅綠,需加上紋理輔助
- matplotlib 支援 `hatch` 參數新增斜線/點狀紋理
- 品質檢查清單建議「Verify color scheme accessibility」

**替代方案考慮**:
- **僅使用紋理不用顏色**: 不符合規格 FR-015 的明確要求
- **使用藍/橙色系**: 色盲友善但不符合業界慣例 (紅跌綠漲)

**實作細節**:
```python
COLORS = {
    "看跌": "#DC143C",    # 深紅色 (Crimson)
    "震盪": "#808080",    # 灰色
    "看漲": "#228B22"     # 深綠色 (Forest Green)
}

HATCHES = {
    "看跌": "///",        # 斜線紋理
    "震盪": "",           # 無紋理
    "看漲": "\\\\\\"      # 反斜線紋理
}

plt.barh(y, width, color=COLORS[category], hatch=HATCHES[category])
```

---

### 6. 批次預測輸出格式設計

**決策**: 每個預測日期生成獨立的 PNG 圖片檔案,命名格式 `prediction_{input_date}_{prediction_date}.png`

**理由**:
- 規格 FR-012 要求「每個日期生成獨立的雙視圖輸出」
- 獨立圖片檔案易於比較與保存
- 檔名包含輸入日期與預測日期,便於識別
- 終端機輸出使用文字摘要,圖片檔案包含完整視覺化細節

**替代方案考慮**:
- **單一大圖包含所有預測**: 當預測數量多時圖表過於擁擠
- **互動式 HTML 報告**: 超出規格範圍 (Out of Scope)

**實作細節**:
```python
def save_prediction_visualization(
    input_date: str,
    prediction_date: str,
    prob_5class: np.ndarray,
    prob_3class: dict,
    actual_data: Optional[dict],
    output_dir: str = "outputs"
) -> str:
    """儲存雙視圖視覺化圖表"""
    filename = f"prediction_{input_date}_{prediction_date}.png"
    filepath = os.path.join(output_dir, filename)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    # 繪製3分類圖表於 ax1
    # 繪製5分類圖表於 ax2
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)

    return filepath
```

---

## 研究結論

所有技術決策已明確,無待解決的 NEEDS CLARIFICATION 項目。主要技術選擇:
1. matplotlib (橫條圖)
2. 複用 `CLASS_MAPPING` 常數 (分類邊界)
3. numpy 浮點數運算 + 斷言驗證 (精確度)
4. pandas DataFrame 索引查詢 (歷史資料)
5. 紅灰綠 + 紋理 (顏色編碼)
6. 獨立 PNG 檔案 (批次輸出)

所有決策符合「禁止過度設計」原則,採用最簡單可行的方案,可進入 Phase 1 設計階段。
