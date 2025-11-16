# Implementation Plan: Enhanced Stock Price Prediction Visualization

**Branch**: `002-enhanced-prediction-viz` | **Date**: 2025-11-16 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/002-enhanced-prediction-viz/spec.md`

**Note**: This template is filled in by the `/speckit.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

建立增強版股價預測視覺化工具,提供3分類聚合視圖(看跌/震盪/看漲)、5分類詳細視圖(對齊橫條圖)、以及歷史資料驗證功能。當預測日期的實際資料存在時,系統將自動載入實際股價、計算實際類別、比對預測準確度,並在視覺化輸出中清楚標示預測是否正確。技術方案採用 Python 3.12.3,擴充現有的 `src/cli/predict.py` CLI 工具,使用 matplotlib 進行橫條圖視覺化,並完全沿用現有的 LSTM 預測邏輯。

## Technical Context

**Language/Version**: Python 3.12.3 (使用者指定)
**Primary Dependencies**:
- matplotlib 3.8+ (橫條圖視覺化)
- numpy 1.24+ (數值計算,已存在)
- pandas 2.0+ (資料處理,已存在)
- 現有依賴: TensorFlow 2.16.1, keras (模型載入,已存在)

**Storage**: CSV 檔案 (歷史資料輸入,已存在機制)
**Testing**: pytest (與現有測試框架一致)
**Target Platform**: WSL2 Ubuntu (Ubuntu_D, 與專案憲章一致)
**Project Type**: Single project (CLI 工具擴充)
**Performance Goals**:
- 單次預測執行時間增加不超過30% (規格SC-004)
- 3秒內完成視覺化圖表生成 (規格SC-009)
- 支援批次預測10個日期不超過總時間限制

**Constraints**:
- 必須完全沿用現有 LSTM 預測邏輯,不修改模型架構 (規格Out of Scope)
- 分類邊界必須與訓練時100%一致 (規格SC-010)
- 3分類聚合計算誤差為0 (規格SC-003)
- 視覺化必須在1920x1080解析度清晰可讀 (規格SC-005)

**Scale/Scope**:
- 單一 Python 模組 (~300-400行程式碼預估)
- 支援批次預測 (現有功能)
- 輸出格式: 終端機文字 + matplotlib 圖片

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

### I. 正體中文優先 ✅
- 所有文件使用正體中文撰寫 ✓
- 程式碼註解與 docstring 將使用正體中文 ✓
- CLI 輸出訊息使用正體中文 ✓

### II. Pythonic 程式碼風格 ✅
- 遵循 PEP 8 規範 ✓
- 使用型別提示 (Type Hints) ✓
- 函式命名 snake_case, 類別命名 PascalCase ✓
- 將使用 black + flake8 進行程式碼檢查 ✓

### III. 規格驅動開發 ✅
- spec.md 已完整定義使用者情境與需求 ✓
- 本 plan.md 完全基於 spec.md 設計 ✓
- 實作將嚴格遵循規格定義 ✓

### IV. 禁止過度設計 ✅
- 不引入不必要的抽象層 ✓
- 不預先設計未來功能 ✓
- 採用最簡單可行的方案 (新增單一 Python 模組,擴充現有 CLI) ✓
- 沿用現有架構,不重構無關程式碼 ✓

### V. GPU 加速機器學習訓練 ✅
- 本功能不涉及模型訓練,僅使用已訓練模型進行預測 ✓
- 預測階段沿用現有 GPU 檢測機制 ✓
- 無需新增 GPU 相關程式碼 ✓

### VI. 可測試性優先 ✅
- 每個使用者情境都有明確的驗收場景 (spec.md 定義) ✓
- 視覺化模組將設計為可獨立測試 (輸入機率向量,輸出圖表檔案) ✓
- 歷史驗證邏輯可獨立測試 (輸入日期,輸出比對結果) ✓
- 3分類聚合計算可單元測試驗證誤差為0 ✓

**結論**: 所有憲章原則皆符合,無違規項目,可進入 Phase 0 研究階段。

## Project Structure

### Documentation (this feature)

```text
specs/[###-feature]/
├── plan.md              # This file (/speckit.plan command output)
├── research.md          # Phase 0 output (/speckit.plan command)
├── data-model.md        # Phase 1 output (/speckit.plan command)
├── quickstart.md        # Phase 1 output (/speckit.plan command)
├── contracts/           # Phase 1 output (/speckit.plan command)
└── tasks.md             # Phase 2 output (/speckit.tasks command - NOT created by /speckit.plan)
```

### Source Code (repository root)

```text
src/
├── cli/
│   ├── predict.py           # 現有: 預測 CLI (將擴充以呼叫新視覺化模組)
│   └── predict_enhanced.py  # 新增: 增強版預測 CLI 入口點
├── visualization/           # 新增目錄
│   ├── __init__.py
│   ├── aggregator.py        # 新增: 3分類聚合邏輯
│   ├── validator.py         # 新增: 歷史資料驗證邏輯
│   └── plotter.py           # 新增: matplotlib 橫條圖繪製
└── prediction/
    └── predictor.py         # 現有: 保持不變,提供5分類機率

tests/
├── unit/
│   ├── test_aggregator.py   # 新增: 測試3分類聚合計算
│   ├── test_validator.py    # 新增: 測試歷史驗證邏輯
│   └── test_plotter.py      # 新增: 測試視覺化輸出
└── integration/
    └── test_predict_enhanced_pipeline.py  # 新增: 端到端測試
```

**Structure Decision**:
- 採用 Single project 結構 (符合現有專案架構)
- 新增 `src/visualization/` 模組目錄,包含3個核心模組
- 新增 `src/cli/predict_enhanced.py` 作為獨立 CLI 入口點
- 現有 `src/prediction/predictor.py` 完全不修改,僅作為依賴使用
- 測試檔案遵循現有 `tests/unit/` 和 `tests/integration/` 結構

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

**無違規項目** - 所有設計決策符合專案憲章,無需額外說明。
