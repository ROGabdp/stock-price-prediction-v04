# Quick Start: Enhanced Stock Price Prediction Visualization

**Feature**: 002-enhanced-prediction-viz
**Date**: 2025-11-16
**Purpose**: 快速開始指南,協助開發者在5分鐘內理解並開始開發此功能

## 一、功能概述 (30秒)

增強版股價預測視覺化工具提供:
1. **3分類聚合視圖**: 將5類漲跌幅聚合為看跌/震盪/看漲,快速判斷市場方向
2. **5分類詳細視圖**: 對齊橫條圖起點,精確比較各類別機率
3. **歷史驗證功能**: 當預測日期已過時,自動載入實際股價並標示預測是否正確

## 二、快速執行 (2分鐘)

### 前置條件

```bash
# 1. 確認在 WSL2 Ubuntu 環境
wsl -d Ubuntu_D
cd /mnt/d/000-github-repositories/stock-price-prediction-v04

# 2. 啟動虛擬環境 (Python 3.12.3)
source venv/bin/activate

# 3. 安裝 matplotlib (若尚未安裝)
pip install matplotlib>=3.8
```

### 執行範例

```bash
# 預測單一歷史日期 (可驗證準確度)
python src/cli/predict_enhanced.py \
    --model-file models/best_tuned_model_20251116_153022.h5 \
    --data-file 19980601-20251111-converted.csv \
    --input-date "2024-01-15"

# 輸出:
# - 終端機顯示文字結果 (3分類+5分類+驗證)
# - outputs/prediction_2024-01-15_2024-02-15.png (視覺化圖表)
```

## 三、核心概念 (2分鐘)

### 3.1 資料流程

```
輸入日期 → 載入模型 → 5分類預測機率
                           ↓
                    ┌──────┴──────┐
                    ↓             ↓
            3分類聚合      歷史驗證
                    ↓             ↓
            看跌/震盪/看漲   實際vs預測
                    ↓             ↓
                雙視圖視覺化 (PNG)
```

### 3.2 模組結構

| 模組 | 職責 | 關鍵函式 |
|------|------|---------|
| `aggregator.py` | 3分類聚合 | `aggregate_to_3_categories()` |
| `validator.py` | 歷史驗證 | `get_actual_data()` |
| `plotter.py` | 視覺化繪圖 | `plot_dual_view()` |
| `predict_enhanced.py` | CLI入口 | `main()` |

### 3.3 關鍵約束

1. **精確度**: 3分類機率總和誤差必須為0 (1e-10容忍度)
2. **一致性**: 分類邊界必須與訓練時100%一致 (沿用 `CLASS_MAPPING`)
3. **對齊**: 5分類橫條圖所有類別從 x=0 開始
4. **效能**: 執行時間增加不超過30% (含視覺化)

## 四、開發任務檢查清單

```markdown
- [ ] 環境設置
  - [ ] WSL2 Ubuntu 已啟動
  - [ ] Python 3.12.3 虛擬環境已啟動
  - [ ] matplotlib 3.8+ 已安裝

- [ ] 建立模組檔案
  - [ ] src/visualization/__init__.py
  - [ ] src/visualization/aggregator.py
  - [ ] src/visualization/validator.py
  - [ ] src/visualization/plotter.py
  - [ ] src/cli/predict_enhanced.py

- [ ] 實作核心函式 (依序)
  - [ ] aggregator.aggregate_to_3_categories()
  - [ ] aggregator.get_predicted_class_3()
  - [ ] validator.calculate_actual_class_5()
  - [ ] validator.map_5class_to_3class()
  - [ ] validator.get_actual_data()
  - [ ] plotter.get_default_config()
  - [ ] plotter.plot_dual_view()
  - [ ] predict_enhanced.main()

- [ ] 撰寫單元測試
  - [ ] tests/unit/test_aggregator.py
  - [ ] tests/unit/test_validator.py
  - [ ] tests/unit/test_plotter.py

- [ ] 撰寫整合測試
  - [ ] tests/integration/test_predict_enhanced_pipeline.py

- [ ] 驗證規格需求
  - [ ] FR-001 至 FR-023 (23項功能需求)
  - [ ] SC-001 至 SC-011 (11項成功標準)

- [ ] 程式碼品質檢查
  - [ ] black 格式化
  - [ ] flake8 檢查通過
  - [ ] mypy 型別檢查 (選用)
  - [ ] 正體中文 docstring 完整
```

## 五、常見問題 FAQ

### Q1: 為什麼要新增獨立的 CLI 檔案而不直接修改 predict.py?

A: 符合「禁止過度設計」原則,避免破壞現有功能。新舊 CLI 可並存,使用者可選擇:
- `predict.py`: 原始預測 (已穩定運作)
- `predict_enhanced.py`: 增強版 (新功能)

### Q2: 3分類聚合計算如何保證誤差為0?

A: 使用 numpy 陣列運算並加上斷言驗證:
```python
bearish = prob_5class[0] + prob_5class[1]
neutral = prob_5class[2]
bullish = prob_5class[3] + prob_5class[4]
total = bearish + neutral + bullish
assert abs(total - 1.0) < 1e-10  # 確保總和精確為1.0
```

### Q3: 如何確保分類邊界與訓練時一致?

A: 直接複用 `src/prediction/predictor.py` 中的 `CLASS_MAPPING` 常數:
```python
from src.prediction.predictor import CLASS_MAPPING

def calculate_actual_class_5(actual_change_pct):
    for class_id, info in CLASS_MAPPING.items():
        min_c = info["min_change"] or float('-inf')
        max_c = info["max_change"] or float('inf')
        if min_c <= actual_change_pct < max_c:
            return class_id
```

### Q4: matplotlib 中文顯示亂碼怎麼辦?

A: 在 `plotter.py` 中設定字型:
```python
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
```

### Q5: 批次預測時如何避免記憶體溢出?

A: 每個預測完成後立即關閉 matplotlib figure:
```python
for date in input_dates:
    result = predict_for_date(...)
    plot_dual_view(result, ...)
    plt.close('all')  # 釋放記憶體
```

## 六、除錯技巧

### 檢查 3分類聚合正確性

```python
# 在 aggregator.py 加入除錯日誌
import logging

def aggregate_to_3_categories(prob_5class):
    bearish = prob_5class[0] + prob_5class[1]
    neutral = prob_5class[2]
    bullish = prob_5class[3] + prob_5class[4]
    total = bearish + neutral + bullish

    logging.debug(f"5分類: {prob_5class}")
    logging.debug(f"聚合: 看跌={bearish:.6f}, 震盪={neutral:.6f}, 看漲={bullish:.6f}")
    logging.debug(f"總和: {total:.10f}")  # 顯示10位小數

    assert abs(total - 1.0) < 1e-10
    return {"看跌": bearish, "震盪": neutral, "看漲": bullish}
```

### 驗證橫條圖對齊

```python
# 在 plotter.py 中驗證 left 參數
for i, prob in enumerate(probs):
    bar = ax.barh(i, prob, left=0, ...)  # 所有橫條必須 left=0
    assert bar.patches[0].get_x() == 0, "橫條起點必須為0"
```

### 測試歷史驗證準確性

```python
# 建立測試案例,使用已知結果
test_cases = [
    {"date": "2024-01-15", "expected_class_5": 3, "expected_class_3": "看漲"},
    {"date": "2024-02-20", "expected_class_5": 1, "expected_class_3": "看跌"},
]

for case in test_cases:
    actual = get_actual_data(df, case["date"], ...)
    assert actual["actual_class_5"] == case["expected_class_5"]
    assert actual["actual_class_3"] == case["expected_class_3"]
```

## 七、下一步

1. **閱讀詳細設計文件**:
   - [data-model.md](data-model.md) - 資料實體定義
   - [research.md](research.md) - 技術決策理由
   - [contracts/](contracts/) - 模組介面契約

2. **開始實作**:
   - 按照 contracts/ 中定義的函式簽名實作
   - 遵循 PEP 8 與 Pythonic 風格
   - 所有 docstring 使用正體中文

3. **執行測試**:
   - 撰寫單元測試驗證每個函式
   - 執行整合測試驗證完整流程
   - 使用真實歷史資料驗證準確度

4. **提交程式碼**:
   - 使用 black 格式化
   - 執行 flake8 檢查
   - 提交前執行完整測試套件

---

**預估開發時間**: 8-12 小時 (依開發者經驗)
**關鍵文件**: [spec.md](spec.md), [plan.md](plan.md), [contracts/](contracts/)
**聯絡支援**: 參考 CLAUDE.md 中的專案憲章與開發流程
