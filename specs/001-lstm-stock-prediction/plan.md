# 實作計畫：LSTM 台股價格預測系統

**分支**: `001-lstm-stock-prediction` | **日期**: 2025-11-15 | **規格**: [spec.md](spec.md)
**輸入**: 功能規格來自 `specs/001-lstm-stock-prediction/spec.md`

**註記**: 本文件由 `/speckit.plan` 指令填寫。執行工作流程請參閱 `.specify/templates/commands/plan.md`。

## 摘要

本專案旨在建立一個 LSTM 深度學習模型，用於預測台股未來 20 個交易日的漲跌幅區間（5 類分類）及其信心度。系統將使用 1998-2025 年的歷史股價資料（約 6,800 筆交易日），透過自動超參數調整（含特徵集選擇）尋找最佳模型配置。技術方案採用 Python + TensorFlow/Keras，運行於 WSL2 + NVIDIA GPU 環境，並支援使用者指定日期進行預測與實際價格比較。

## 技術背景

**語言/版本**: Python 3.9+
**主要依賴套件**:
- TensorFlow 2.x (with GPU support) 或 PyTorch (with CUDA support)
- Keras Tuner 或 Optuna (超參數調整)
- pandas, numpy (資料處理)
- scikit-learn (特徵縮放與評估指標)
- matplotlib, seaborn (視覺化，選用)

**儲存**:
- 輸入資料: CSV 檔案 (`19980601-20251111-converted.csv`)
- 模型檔案: HDF5 (.h5) 或 SavedModel 格式
- 試驗紀錄: Keras Tuner / Optuna / MLflow 本地資料庫

**測試**: pytest (單元測試與整合測試)
**目標平台**: WSL2 (Ubuntu_D, D:\wsl\Ubuntu_D) + NVIDIA GPU with CUDA
**專案類型**: 單一專案 (single project)
**效能目標**:
- 資料預處理: 10 年資料 < 10 分鐘
- 基準模型訓練 (GPU): 100 epoch < 2 小時
- 預測響應時間: < 10 秒
- GPU 記憶體使用率: < 90%

**限制條件**:
- 訓練資料集: 約 6,800 筆交易日 (1998-2025)
- 驗證準確度: 基準模型 ≥ 40%，最佳模型 ≥ 50% (目標)
- 超參數試驗次數: 至少 50 次

**規模/範圍**:
- 輸入特徵: 12-22 個（依特徵集選擇）
- 輸出類別: 5 類（漲跌幅區間）
- 時間窗口: 20-80 天（可調整）
- 訓練樣本數: 約 6,000-6,700 筆（扣除時間窗口與目標變數計算需求）

## 憲章檢查

*關卡: 必須在 Phase 0 研究前通過。Phase 1 設計後重新檢查。*

### 核心原則合規性

| 原則 | 檢查項目 | 狀態 | 說明 |
|------|---------|------|------|
| **I. 正體中文優先** | 所有文件、註解使用正體中文 | ✅ 通過 | 規格、計畫、程式碼註解均使用正體中文 |
| **II. Pythonic 程式碼風格** | 遵循 PEP 8，使用 black/flake8 | ✅ 通過 | 將使用 black 格式化、flake8 檢查、型別提示 |
| **III. 規格驅動開發** | 規格完整，實作符合規格 | ✅ 通過 | spec.md 已完成，plan.md 依規格設計 |
| **IV. 禁止過度設計** | 保持簡潔，YAGNI 原則 | ✅ 通過 | 無不必要的抽象層，專注 MVP 功能 |
| **V. GPU 加速訓練** | 使用 WSL2 + NVIDIA GPU | ✅ 通過 | TensorFlow-GPU，GPU 可用性檢查機制 |
| **VI. 可測試性優先** | 獨立測試場景，Given-When-Then | ✅ 通過 | 三個使用者情境均可獨立測試 |

### 技術規範合規性

| 規範類別 | 檢查項目 | 狀態 | 說明 |
|---------|---------|------|------|
| **程式語言** | Python 3.9+ | ✅ 通過 | 使用 Python 3.9+ |
| **深度學習框架** | TensorFlow 2.x (GPU) 或 PyTorch | ✅ 通過 | 選用 TensorFlow 2.x + Keras |
| **版本控制** | Git，預設分支 main | ✅ 通過 | 已使用 Git，分支命名符合規範 |
| **開發環境** | WSL2 (Ubuntu_D) | ✅ 通過 | 訓練環境為 WSL2 + NVIDIA GPU |
| **程式碼品質工具** | black, flake8/pylint | ✅ 通過 | 將整合 black, flake8, mypy |
| **文件規範** | .specify/ 模板，docstring | ✅ 通過 | 使用 speckit 框架，函式包含中文 docstring |

**結論**: 所有憲章檢查項目均通過，無違規需要特別說明。

## 專案結構

### 文件結構 (本功能)

```text
specs/001-lstm-stock-prediction/
├── spec.md              # 功能規格
├── plan.md              # 本文件 (實作計畫)
├── research.md          # Phase 0 輸出 (技術研究)
├── data-model.md        # Phase 1 輸出 (資料模型)
├── quickstart.md        # Phase 1 輸出 (快速開始指南)
├── contracts/           # Phase 1 輸出 (模組介面定義)
│   ├── data_preprocessing.md
│   ├── feature_engineering.md
│   ├── model_training.md
│   ├── hyperparameter_tuning.md
│   └── prediction.md
└── tasks.md             # Phase 2 輸出 (/speckit.tasks 產生)
```

### 原始碼結構 (專案根目錄)

```text
src/
├── data/
│   ├── __init__.py
│   ├── data_loader.py           # CSV 載入與驗證
│   └── data_splitter.py         # 訓練/驗證/測試集分割
├── features/
│   ├── __init__.py
│   ├── feature_engineer.py      # 特徵工程主邏輯
│   ├── feature_sets.py          # Set A/B/C 定義
│   └── scalers.py               # 特徵縮放
├── models/
│   ├── __init__.py
│   ├── lstm_baseline.py         # 基準 LSTM 模型
│   └── model_builder.py         # 動態模型建構器
├── tuning/
│   ├── __init__.py
│   ├── hyperparameter_tuner.py  # Keras Tuner/Optuna 整合
│   └── feature_set_selector.py  # 特徵集選擇邏輯
├── prediction/
│   ├── __init__.py
│   └── predictor.py             # 預測與結果輸出
├── utils/
│   ├── __init__.py
│   ├── gpu_checker.py           # GPU 可用性檢查
│   └── logger.py                # 訓練日誌記錄
└── cli/
    ├── __init__.py
    ├── train.py                 # 訓練指令
    ├── tune.py                  # 超參數調整指令
    └── predict.py               # 預測指令

tests/
├── integration/
│   ├── test_end_to_end.py       # 端到端整合測試
│   ├── test_training_pipeline.py
│   └── test_prediction_pipeline.py
└── unit/
    ├── test_data_loader.py
    ├── test_feature_engineer.py
    ├── test_lstm_baseline.py
    └── test_predictor.py

models/                          # 訓練好的模型儲存目錄
├── baseline_model.h5
└── best_tuned_model.h5

logs/                            # 訓練日誌與試驗紀錄
├── training_logs/
└── tuning_logs/

notebooks/                       # 探索性分析 (選用)
└── eda.ipynb
```

**結構決策**:

選用 **單一專案結構 (Option 1)**，因為：
- 本專案為機器學習模型訓練系統，無前後端分離需求
- 所有功能均在本地 Python 環境執行
- 模組劃分清晰: data (資料處理) → features (特徵工程) → models (模型) → tuning (調參) → prediction (預測)
- CLI 介面提供訓練、調參、預測三個主要指令

## 複雜度追蹤

> **僅在憲章檢查有違規需要說明時填寫**

| 違規項目 | 為何需要 | 被拒絕的簡單替代方案及原因 |
|---------|---------|--------------------------|
| 無 | N/A | N/A |

**說明**: 本專案所有設計決策均符合憲章要求，無需特別說明的複雜度引入。
