# Tasks: LSTM 台股價格預測系統

**Input**: 設計文件來自 `/specs/001-lstm-stock-prediction/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/

**Organization**: 任務按使用者情境組織，以實現每個情境的獨立實作與測試

## 格式: `[ID] [P?] [Story] Description`

- **[P]**: 可並行執行（不同檔案，無依賴關係）
- **[Story]**: 任務所屬的使用者情境（例如：US1, US2, US3）
- 描述中包含完整檔案路徑

## 路徑慣例

- **單一專案**: `src/`, `tests/` 位於專案根目錄
- 所有路徑遵循 plan.md 定義的專案結構

---

## Phase 1: Setup（專案初始化）

**目的**: 建立專案結構與基礎配置

- [X] T001 建立專案目錄結構 (src/, tests/, models/, logs/, notebooks/)
- [X] T002 建立 Python 虛擬環境與安裝依賴套件 (requirements.txt: tensorflow-gpu==2.10.0, pandas, numpy, scikit-learn, keras-tuner, pytest, black, flake8, mypy)
- [X] T003 [P] 配置程式碼品質工具 (.flake8, pyproject.toml for black, mypy.ini)
- [X] T004 [P] 建立 .gitignore 檔案（排除 models/, logs/, venv/, __pycache__/）
- [X] T005 [P] 建立 README.md 說明檔（專案描述、安裝步驟、使用方式）

---

## Phase 2: Foundational（基礎建設）

**目的**: 所有使用者情境共用的核心基礎建設，**必須**在任何使用者情境開始前完成

**⚠️ 關鍵**: 在此階段完成前，無法開始任何使用者情境的工作

- [X] T006 實作 GPU 可用性檢查與配置 in src/utils/gpu_checker.py
- [X] T007 [P] 實作訓練日誌記錄工具 in src/utils/logger.py
- [X] T008 [P] 實作資料載入模組 in src/data/data_loader.py（load_csv_data, validate_schema）
- [X] T009 [P] 實作資料集分割模組 in src/data/data_splitter.py（split_time_series）
- [X] T010 實作特徵集定義配置 in src/features/feature_sets.py（FEATURE_SETS: Set A/B/C 定義）
- [X] T011 實作特徵縮放工具 in src/features/scalers.py（create_scaler, fit_scaler, transform_features）
- [X] T012 建立所有模組的 __init__.py 檔案 (src/data/, src/features/, src/models/, src/tuning/, src/prediction/, src/utils/, src/cli/)

**Checkpoint**: 基礎建設完成 - 使用者情境實作可開始並行進行

---

## Phase 3: 使用者情境 1 - 訓練基準模型並評估準確度 (優先級: P1) 🎯 MVP

**目標**: 建立基準 LSTM 模型，使用歷史台股資料訓練並評估預測準確度

**獨立測試**: 載入歷史資料 → 執行訓練腳本 → 獲得驗證損失與準確度指標 → 模型儲存成功

### 實作任務

- [X] T013 [P] [US1] 實作特徵工程主邏輯 in src/features/feature_engineer.py（engineer_features, get_feature_set_config）
- [X] T014 [P] [US1] 實作目標變數計算 in src/features/feature_engineer.py（calculate_target_variable，5 類 One-Hot 編碼）
- [X] T015 [US1] 實作基準 LSTM 模型建構 in src/models/lstm_baseline.py（build_lstm_model: 3 層 128-64-32, Dropout 0.2）
- [X] T016 [US1] 實作模型訓練邏輯 in src/models/model_builder.py（train_model, 配置 Callbacks: EarlyStopping, ModelCheckpoint, CSVLogger）
- [X] T017 [US1] 實作訓練 CLI 指令 in src/cli/train.py（argparse 參數: data-file, feature-set, time-steps, epochs, batch-size, output-dir）
- [X] T018 [US1] 整合完整訓練流程 in src/cli/train.py（資料載入 → 特徵工程 → 資料集分割 → 模型建構 → 訓練 → 儲存）
- [X] T019 [US1] 驗證基準模型訓練流程（使用完整歷史資料，驗證準確度 ≥ 40%）

**Checkpoint**: 此時使用者情境 1 應完全可運作且可獨立測試（訓練基準模型並獲得評估指標）

---

## Phase 4: 使用者情境 2 - 自動超參數調整尋找最佳模型 (優先級: P2)

**目標**: 使用 Keras Tuner 系統性探索超參數組合，找到驗證損失最小的最佳模型

**獨立測試**: 配置超參數搜尋空間 → 執行調整程序 → 驗證試驗紀錄完整性 → 選定最佳模型

### 實作任務

- [X] T020 [P] [US2] 實作動態模型建構器 in src/models/model_builder.py（build_dynamic_lstm_model，支援可變層數與單元數）
- [X] T021 [US2] 實作 Keras Tuner 整合 in src/tuning/hyperparameter_tuner.py（create_tuner, define_search_space）
- [X] T022 [US2] 實作超參數搜尋空間定義 in src/tuning/hyperparameter_tuner.py（7 維度: time_steps, num_layers, units, dropout, learning_rate, batch_size, feature_set_id）
- [X] T023 [US2] 實作超參數調整執行邏輯 in src/tuning/hyperparameter_tuner.py（run_tuning, get_best_model）
- [X] T024 [US2] 實作特徵集選擇邏輯 in src/tuning/feature_set_selector.py（select_feature_set，根據 hp.Choice 動態載入 Set A/B/C）
- [X] T025 [US2] 實作超參數調整 CLI 指令 in src/cli/tune.py（argparse 參數: data-file, max-trials, epochs-per-trial, output-dir）
- [X] T026 [US2] 整合完整超參數調整流程 in src/cli/tune.py（資料載入 → Tuner 建立 → 執行調整 → 儲存最佳模型）
- [ ] T027 [US2] 驗證超參數調整流程（執行至少 10 次試驗，確認最佳模型驗證準確度 ≥ 50%）

**Checkpoint**: 此時使用者情境 1 與 2 應皆可獨立運作（基準模型訓練 + 超參數調整）

---

## Phase 5: 使用者情境 3 - 根據指定日期預測未來收盤價 (優先級: P3)

**目標**: 載入最佳模型，根據使用者指定日期預測未來 20 個交易日的漲跌幅區間與收盤價

**獨立測試**: 提供測試日期 → 執行預測腳本 → 驗證輸出格式與準確性

### 實作任務

- [X] T028 [P] [US3] 實作模型載入功能 in src/prediction/predictor.py（load_model）
- [X] T029 [P] [US3] 實作日期預測邏輯 in src/prediction/predictor.py（predict_for_date，載入輸入日期前 60 天資料）
- [X] T030 [US3] 實作預測結果格式化 in src/prediction/predictor.py（format_prediction_result，包含預測類別、信心度、收盤價區間、實際收盤價比較）
- [X] T031 [US3] 實作預測 CLI 指令 in src/cli/predict.py（argparse 參數: model-file, data-file, input-date, feature-set, time-steps）
- [X] T032 [US3] 整合完整預測流程 in src/cli/predict.py（模型載入 → 歷史資料載入 → 日期驗證 → 特徵工程 → 預測 → 結果格式化與輸出）
- [ ] T033 [US3] 驗證預測流程（使用歷史日期測試，驗證預測結果格式正確且響應時間 < 10 秒）

**Checkpoint**: 此時所有使用者情境應皆可獨立運作（訓練 + 調參 + 預測）

---

## Phase 6: Polish & 跨情境改善

**目的**: 影響多個使用者情境的改善項目

- [X] T034 [P] 實作單元測試 in tests/unit/test_data_loader.py（測試 CSV 載入、欄位驗證、缺失值處理）
- [X] T035 [P] 實作單元測試 in tests/unit/test_feature_engineer.py（測試特徵工程邏輯、Set A/B/C 定義正確性、目標變數計算）
- [X] T036 [P] 實作單元測試 in tests/unit/test_lstm_baseline.py（測試模型建構、輸入/輸出形狀正確性）
- [X] T037 [P] 實作單元測試 in tests/unit/test_predictor.py（測試預測邏輯、結果格式化）
- [X] T038 [P] 實作整合測試 in tests/integration/test_training_pipeline.py（測試完整訓練流程：資料載入 → 特徵工程 → 訓練）
- [X] T039 [P] 實作整合測試 in tests/integration/test_prediction_pipeline.py（測試完整預測流程：模型載入 → 預測 → 結果輸出）
- [X] T040 [P] 實作端到端測試 in tests/integration/test_end_to_end.py（測試完整工作流程：訓練 → 調參 → 預測）
- [ ] T041 執行程式碼品質檢查（black 格式化、flake8 檢查、mypy 型別檢查）
- [ ] T042 [P] 完善錯誤處理與邊界案例（資料不足、GPU 不可用、類別不平衡、超參數調整中斷恢復）
- [ ] T043 驗證 quickstart.md 指南（依照指南執行完整流程，確認所有步驟正確無誤）
- [ ] T044 [P] 更新文件與程式碼註解（確保所有函式包含正體中文 docstring，README.md 完整）

---

## 依賴關係與執行順序

### 階段依賴關係

- **Setup (Phase 1)**: 無依賴 - 可立即開始
- **Foundational (Phase 2)**: 依賴 Setup 完成 - **阻塞所有使用者情境**
- **使用者情境 (Phase 3-5)**: 皆依賴 Foundational 完成
  - 使用者情境可並行進行（若有多位開發者）
  - 或依優先級順序執行（P1 → P2 → P3）
- **Polish (Phase 6)**: 依賴所需的使用者情境完成

### 使用者情境依賴關係

- **使用者情境 1 (P1)**: Foundational 完成後即可開始 - 無其他情境依賴
- **使用者情境 2 (P2)**: Foundational 完成後即可開始 - 依賴 US1 的模型訓練邏輯，但應可獨立測試
- **使用者情境 3 (P3)**: Foundational 完成後即可開始 - 依賴 US1 的模型訓練，但應可獨立測試

### 各使用者情境內部依賴

- 特徵工程 → 模型建構 → 訓練 → CLI 整合
- 核心實作 → 整合邏輯 → 驗證測試
- 情境完成後才移至下一優先級

### 並行機會

- 所有標記 [P] 的 Setup 任務可並行執行
- 所有標記 [P] 的 Foundational 任務可並行執行（在 Phase 2 內）
- Foundational 完成後，所有使用者情境可並行開始（若團隊容量允許）
- 各使用者情境內標記 [P] 的任務可並行執行
- 不同使用者情境可由不同團隊成員並行開發

---

## 並行範例: 使用者情境 1

```bash
# 同時執行使用者情境 1 的並行任務:
Task: "實作特徵工程主邏輯 in src/features/feature_engineer.py"
Task: "實作目標變數計算 in src/features/feature_engineer.py"
```

---

## 實作策略

### MVP 優先（僅使用者情境 1）

1. 完成 Phase 1: Setup
2. 完成 Phase 2: Foundational（**關鍵** - 阻塞所有情境）
3. 完成 Phase 3: 使用者情境 1
4. **停止並驗證**: 獨立測試使用者情境 1
5. 若準備好則部署/展示

### 增量交付

1. 完成 Setup + Foundational → 基礎建設就緒
2. 新增使用者情境 1 → 獨立測試 → 部署/展示（MVP！）
3. 新增使用者情境 2 → 獨立測試 → 部署/展示
4. 新增使用者情境 3 → 獨立測試 → 部署/展示
5. 每個情境新增價值且不破壞先前情境

### 並行團隊策略

若有多位開發者:

1. 團隊共同完成 Setup + Foundational
2. Foundational 完成後:
   - 開發者 A: 使用者情境 1
   - 開發者 B: 使用者情境 2
   - 開發者 C: 使用者情境 3
3. 各情境獨立完成與整合

---

## 注意事項

- [P] 任務 = 不同檔案，無依賴關係
- [Story] 標籤將任務映射至特定使用者情境，以便追蹤
- 每個使用者情境應可獨立完成與測試
- 在每個 checkpoint 停止以獨立驗證情境
- 避免: 模糊任務、相同檔案衝突、破壞獨立性的跨情境依賴
- 每個任務或邏輯群組完成後提交 commit
- 所有程式碼必須遵循 PEP 8 規範並包含正體中文註解
- GPU 不可用時系統應自動降級使用 CPU 並記錄警告訊息

---

## 任務統計

- **總任務數**: 44 個任務
- **使用者情境 1 (P1)**: 7 個任務（T013-T019）
- **使用者情境 2 (P2)**: 8 個任務（T020-T027）
- **使用者情境 3 (P3)**: 6 個任務（T028-T033）
- **並行任務數**: 17 個標記 [P] 的任務
- **建議 MVP 範圍**: Phase 1 + Phase 2 + Phase 3（使用者情境 1）
