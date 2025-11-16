# Tasks: Enhanced Stock Price Prediction Visualization

**Input**: Design documents from `/specs/002-enhanced-prediction-viz/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/

**Tests**: 測試任務已包含,基於規格中的獨立測試需求

**Organization**: 任務按用戶故事分組,每個故事可獨立實作與測試

## Format: `[ID] [P?] [Story] Description`

- **[P]**: 可並行執行 (不同檔案,無相依性)
- **[Story]**: 所屬用戶故事 (US1, US2, US3, US4)
- 包含完整檔案路徑

## Path Conventions

- **Single project**: `src/`, `tests/` at repository root
- Paths assume current project structure from plan.md

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: 專案初始化與基礎架構設定

- [ ] T001 建立 `src/visualization/` 目錄結構
- [ ] T002 建立 `src/visualization/__init__.py` 模組初始化檔案
- [ ] T003 [P] 建立 `tests/unit/` 測試目錄 (若不存在)
- [ ] T004 [P] 建立 `tests/integration/` 測試目錄 (若不存在)
- [ ] T005 [P] 確認 matplotlib 3.8+ 已安裝至虛擬環境 (`pip install matplotlib>=3.8`)
- [ ] T006 [P] 建立 `outputs/` 目錄用於儲存視覺化圖片

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: 核心基礎設施,所有用戶故事的前置需求

**⚠️ CRITICAL**: 此階段必須完成才能開始任何用戶故事實作

- [ ] T007 在 `src/visualization/aggregator.py` 建立模組骨架 (含 docstring 與型別提示)
- [ ] T008 在 `src/visualization/validator.py` 建立模組骨架 (含 docstring 與型別提示)
- [ ] T009 在 `src/visualization/plotter.py` 建立模組骨架 (含 docstring 與型別提示)
- [ ] T010 在 `src/cli/predict_enhanced.py` 建立 CLI 骨架 (含 argparse 參數定義)
- [ ] T011 驗證可從 `src/prediction/predictor.py` 正確 import `CLASS_MAPPING` 常數

**Checkpoint**: 基礎架構就緒 - 用戶故事實作可並行開始

---

## Phase 3: User Story 1 - View Simplified 3-Category Prediction (Priority: P1) 🎯 MVP

**Goal**: 實作3分類聚合功能,使用者能快速查看看跌/震盪/看漲預測

**Independent Test**: 執行預測程式並驗證3分類聚合圖表正確顯示機率總和

**Acceptance Scenarios** (from spec.md):
1. 系統顯示3分類預測結果,其中「看跌」機率 = 極度下跌機率 + 溫和下跌機率
2. 系統顯示3分類預測結果,其中「看漲」機率 = 極度上漲機率 + 溫和上漲機率
3. 系統顯示3分類預測結果,其中「震盪」機率保持不變
4. 可以清楚看到3個類別的橫條圖,每個類別標示機率百分比

### 實作任務

- [ ] T012 [P] [US1] 實作 `aggregate_to_3_categories()` 函式於 `src/visualization/aggregator.py`
  - 輸入: np.ndarray 長度5 (5分類機率)
  - 輸出: Dict[str, float] {"看跌": float, "震盪": float, "看漲": float}
  - 驗證: 總和誤差 < 1e-10 (assert)

- [ ] T013 [P] [US1] 實作 `get_predicted_class_3()` 函式於 `src/visualization/aggregator.py`
  - 輸入: Dict[str, float] (3分類機率)
  - 輸出: str (機率最高的類別)
  - 處理相同機率情況 (優先級: 看跌 > 震盪 > 看漲)

- [ ] T014 [US1] 在 `src/cli/predict_enhanced.py` 整合 aggregator 模組
  - 呼叫 `predict_for_date()` 取得5分類機率
  - 呼叫 `aggregate_to_3_categories()` 聚合為3分類
  - 呼叫 `get_predicted_class_3()` 取得預測類別

- [ ] T015 [US1] 實作3分類文字輸出於 `src/cli/predict_enhanced.py`
  - 格式化3分類機率為百分比
  - 使用 ASCII 橫條圖呈現 (█ 字元)
  - 標示預測類別與信心度

### 測試任務

- [ ] T016 [P] [US1] 撰寫 `tests/unit/test_aggregator.py` 單元測試
  - 測試正常聚合 (sum=1.0)
  - 測試邊界情況 (極度下跌100%, 均勻分佈)
  - 測試錯誤處理 (形狀錯誤, 總和錯誤)
  - 測試 `get_predicted_class_3()` 各種情況

- [ ] T017 [US1] 執行測試並驗證3分類聚合功能正確
  - 運行 `pytest tests/unit/test_aggregator.py -v`
  - 確認所有測試通過
  - 驗證機率總和誤差為0

---

## Phase 4: User Story 2 - View Aligned 5-Category Detail (Priority: P2)

**Goal**: 實作5分類視覺化,所有橫條圖起點對齊,易於比較機率高低

**Independent Test**: 執行預測程式並驗證5分類圖表正確對齊(所有橫條從同一起點開始)

**Acceptance Scenarios** (from spec.md):
1. 系統顯示5分類詳細結果,包含所有5個類別的機率
2. 所有5個類別的橫條圖從相同的起點(左側對齊)開始繪製
3. 可以快速識別出機率最高的類別,視覺判讀誤差降低

### 實作任務

- [ ] T018 [P] [US2] 實作 `get_default_config()` 函式於 `src/visualization/plotter.py`
  - 回傳 VisualizationConfigDict (data-model.md 定義)
  - 設定中文字型、顏色、DPI 等預設值
  - 包含5分類顏色列表 (5個元素)

- [ ] T019 [P] [US2] 實作 matplotlib 中文字型設定於 `src/visualization/plotter.py`
  - 設定 `plt.rcParams['font.sans-serif']` = ['Microsoft JhengHei', 'SimHei']
  - 設定 `plt.rcParams['axes.unicode_minus']` = False
  - 新增字型缺失警告處理

- [ ] T020 [US2] 實作 `plot_5class_bar()` 輔助函式於 `src/visualization/plotter.py`
  - 輸入: ax (matplotlib axes), prob_5class (np.ndarray), config (dict)
  - 繪製5分類橫條圖,確保所有橫條 `left=0` (對齊)
  - 標註機率百分比於橫條右側
  - 標記預測類別 (箭頭 "←")

- [ ] T021 [US2] 整合5分類視覺化至 CLI 輸出於 `src/cli/predict_enhanced.py`
  - 輸出5分類文字結果 (格式化)
  - 準備呼叫 plotter 繪製圖表 (下個story實作)

### 測試任務

- [ ] T022 [P] [US2] 撰寫 `tests/unit/test_plotter.py` 單元測試
  - 測試 `get_default_config()` 回傳正確格式
  - 測試 matplotlib 字型設定不拋出錯誤
  - 測試 `plot_5class_bar()` 生成有效圖表
  - 驗證橫條起點對齊 (所有 bar.get_x() == 0)

- [ ] T023 [US2] 執行測試並驗證5分類視覺化功能
  - 運行 `pytest tests/unit/test_plotter.py -v`
  - 確認所有測試通過
  - 手動檢查生成的測試圖表對齊正確

---

## Phase 5: User Story 3 - View Historical Data and Validation (Priority: P2)

**Goal**: 實作歷史驗證功能,當預測日期已過時顯示實際股價並比對預測準確度

**Independent Test**: 使用歷史日期執行預測並驗證系統正確顯示實際價格與預測比對結果

**Acceptance Scenarios** (from spec.md):
1. 若預測日期實際資料存在,系統顯示實際收盤價和實際漲跌幅
2. 系統顯示實際類別(基於實際漲跌幅計算的5分類和3分類)
3. 系統清楚標示5分類預測是否正確
4. 系統清楚標示3分類預測是否正確
5. 若預測日期實際資料不存在,系統僅顯示預測結果並標註「待驗證」

### 實作任務

- [ ] T024 [P] [US3] 實作 `calculate_actual_class_5()` 函式於 `src/visualization/validator.py`
  - 輸入: actual_change_pct (float, 實際漲跌幅百分比)
  - 輸出: int (0-4, 5分類類別)
  - 使用 `CLASS_MAPPING` 確保100%一致性
  - 處理邊界值 (使用 `<` 而非 `<=`)

- [ ] T025 [P] [US3] 實作 `map_5class_to_3class()` 函式於 `src/visualization/validator.py`
  - 輸入: class_5 (int, 0-4)
  - 輸出: str ("看跌"/"震盪"/"看漲")
  - 映射規則: [0,1]→看跌, [2]→震盪, [3,4]→看漲

- [ ] T026 [US3] 實作 `get_actual_data()` 函式於 `src/visualization/validator.py`
  - 輸入: df_raw, input_date, prediction_date, predicted_class_5, predicted_class_3
  - 輸出: Optional[ActualDataDict]
  - 查詢 DataFrame 取得實際收盤價
  - 計算實際漲跌幅
  - 呼叫 `calculate_actual_class_5()` 計算實際類別
  - 呼叫 `map_5class_to_3class()` 轉換為3分類
  - 比對預測正確性 (is_correct_5class, is_correct_3class)
  - 若日期不存在回傳 None

- [ ] T027 [US3] 整合歷史驗證至 CLI 於 `src/cli/predict_enhanced.py`
  - 呼叫 `get_actual_data()` 查詢實際資料
  - 若存在,輸出實際結果摘要
  - 標示預測正確性 (✓ 或 ✗)
  - 若不存在,輸出「待驗證 ⏳」

### 測試任務

- [ ] T028 [P] [US3] 撰寫 `tests/unit/test_validator.py` 單元測試
  - 測試 `calculate_actual_class_5()` 各漲跌幅範圍
  - 測試邊界值 (-5.0%, -2.5%, 0%, +2.5%, +5.0%)
  - 測試 `map_5class_to_3class()` 所有映射
  - 測試 `get_actual_data()` 存在與不存在情況
  - 驗證與 `CLASS_MAPPING` 100%一致

- [ ] T029 [US3] 執行測試並驗證歷史驗證功能
  - 運行 `pytest tests/unit/test_validator.py -v`
  - 確認所有測試通過
  - 使用真實歷史資料測試準確度計算

---

## Phase 6: User Story 4 - Dual-View Prediction Output (Priority: P3)

**Goal**: 實作雙視圖視覺化輸出,在同一次執行中同時看到3分類和5分類兩種視圖

**Independent Test**: 執行預測程式一次並驗證同時輸出兩種圖表

**Acceptance Scenarios** (from spec.md):
1. 執行一次預測程式,系統在同一輸出中顯示3分類聚合視圖和5分類詳細視圖
2. 3分類和5分類的數據保持一致性(3分類的總和等於對應5分類的總和)
3. 儲存結果至檔案時,兩種視圖都完整儲存在輸出檔案中

### 實作任務

- [ ] T030 [P] [US4] 實作 `plot_3class_bar()` 輔助函式於 `src/visualization/plotter.py`
  - 輸入: ax (matplotlib axes), prob_3class (dict), config (dict)
  - 繪製3分類橫條圖,使用顏色與紋理
  - 標註機率百分比
  - 標記預測類別 (箭頭 "←")

- [ ] T031 [US4] 實作 `plot_dual_view()` 主函式於 `src/visualization/plotter.py`
  - 輸入: prediction_result (PredictionResultDict), output_path, config
  - 輸出: str (儲存的檔案路徑)
  - 建立 2x1 子圖 (上方3分類,下方5分類)
  - 呼叫 `plot_3class_bar()` 繪製上方子圖
  - 呼叫 `plot_5class_bar()` 繪製下方子圖
  - 若實際資料存在,加入驗證摘要文字
  - 儲存為 PNG (DPI=150)
  - 關閉 figure 釋放記憶體

- [ ] T032 [US4] 整合雙視圖輸出至 CLI 於 `src/cli/predict_enhanced.py`
  - 組裝 PredictionResultDict 資料結構
  - 呼叫 `plot_dual_view()` 生成圖表
  - 輸出圖表儲存路徑至終端機
  - 新增 `--no-plot` 參數支援 (僅文字輸出)

- [ ] T033 [US4] 實作批次預測支援於 `src/cli/predict_enhanced.py`
  - 迴圈處理多個 `--input-date` 參數
  - 為每個日期生成獨立的視覺化圖表
  - 輸出批次預測摘要 (總數、可驗證數、準確率)

### 測試任務

- [ ] T034 [P] [US4] 擴充 `tests/unit/test_plotter.py` 測試雙視圖功能
  - 測試 `plot_3class_bar()` 正確繪製
  - 測試 `plot_dual_view()` 生成2個子圖
  - 驗證PNG檔案成功儲存
  - 測試包含實際資料的雙視圖輸出

- [ ] T035 [US4] 撰寫 `tests/integration/test_predict_enhanced_pipeline.py` 整合測試
  - 測試端到端流程: 載入模型 → 預測 → 聚合 → 驗證 → 視覺化
  - 測試單一日期預測
  - 測試批次預測 (3個日期)
  - 測試包含歷史驗證的預測
  - 測試未來日期預測 (無驗證)
  - 驗證輸出檔案格式與內容

- [ ] T036 [US4] 執行完整測試套件
  - 運行 `pytest tests/unit/ tests/integration/ -v`
  - 確認所有測試通過
  - 檢查測試覆蓋率 (應 >80%)

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: 程式碼品質、文件完整性、效能優化

- [ ] T037 [P] 使用 black 格式化所有新增程式碼
  - `black src/visualization/ src/cli/predict_enhanced.py`
  - 確認符合 PEP 8 規範

- [ ] T038 [P] 執行 flake8 檢查並修正所有警告
  - `flake8 src/visualization/ src/cli/predict_enhanced.py`
  - 目標: 零警告

- [ ] T039 [P] 驗證所有 docstring 使用正體中文且完整
  - 檢查所有函式與類別都有 docstring
  - 包含 Args, Returns, Raises, Example
  - 符合 Google Style 或 NumPy Style

- [ ] T040 [P] 新增型別提示至所有函式簽名
  - 使用 `typing` 模組 (Dict, Optional, List 等)
  - 執行 `mypy src/visualization/` 檢查 (選用)

- [ ] T041 測試效能符合規格需求 (SC-004, SC-009)
  - 單次預測執行時間增加 < 30%
  - 視覺化生成時間 < 3 秒
  - 批次預測10個日期 < 50 秒

- [ ] T042 建立使用範例與文件
  - 在 CLI help 中加入使用範例 (epilog)
  - 更新 README.md 加入增強版預測章節 (選用)
  - 驗證 quickstart.md 範例可執行

- [ ] T043 最終驗收測試
  - 執行完整規格驗收場景 (spec.md 中所有 Given-When-Then)
  - 驗證所有23項功能需求 (FR-001 至 FR-023)
  - 驗證所有11項成功標準 (SC-001 至 SC-011)
  - 確認無憲章違規項目

---

## Implementation Strategy

### MVP Scope (Minimum Viable Product)

**建議 MVP**: User Story 1 (P1) - 3分類聚合預測

**理由**:
- 提供核心價值:快速判斷市場方向 (看跌/震盪/看漲)
- 可獨立測試與驗證
- 不依賴視覺化圖表 (文字輸出即可驗證)
- 實作最簡單,風險最低

**MVP 任務**: T001-T011 (Setup+Foundation) + T012-T017 (US1 實作+測試)

### Incremental Delivery

1. **Sprint 1 (MVP)**: Phase 1-3 (Setup + Foundation + US1)
   - 交付物: 3分類聚合預測 (文字輸出)
   - 可驗證: 機率總和為1.0, 聚合邏輯正確

2. **Sprint 2**: Phase 4 (US2)
   - 交付物: 5分類視覺化 (matplotlib 橫條圖)
   - 可驗證: 橫條對齊, 中文顯示正確

3. **Sprint 3**: Phase 5 (US3)
   - 交付物: 歷史驗證功能
   - 可驗證: 準確度計算正確, 與 CLASS_MAPPING 一致

4. **Sprint 4**: Phase 6-7 (US4 + Polish)
   - 交付物: 完整雙視圖視覺化 + 批次預測
   - 可驗證: 所有規格需求滿足

### Parallel Execution Opportunities

#### Phase 1 (Setup)
可並行執行:
- T003, T004 (測試目錄建立)
- T005, T006 (依賴安裝與輸出目錄建立)

#### Phase 3 (US1)
可並行執行:
- T012, T013 (aggregator 模組兩個函式)
- T016 (測試撰寫可與實作並行,若採用 TDD)

#### Phase 4 (US2)
可並行執行:
- T018, T019 (plotter 模組配置與字型設定)

#### Phase 5 (US3)
可並行執行:
- T024, T025 (validator 模組兩個輔助函式)

#### Phase 6 (US4)
可並行執行:
- T030 (plot_3class_bar 可獨立實作)
- T034, T035 (測試撰寫)

#### Phase 7 (Polish)
可並行執行:
- T037, T038, T039, T040 (所有程式碼品質檢查)

---

## Dependencies & Execution Order

### Story Completion Order

```
Phase 1 (Setup)
    ↓
Phase 2 (Foundation) ← 必須完成才能開始任何 User Story
    ↓
    ├─→ User Story 1 (P1) 🎯 MVP ← 最優先,可獨立交付
    │
    ├─→ User Story 2 (P2) ← 依賴 US1 的聚合結果
    │
    ├─→ User Story 3 (P2) ← 可與 US2 並行,不依賴視覺化
    │
    └─→ User Story 4 (P3) ← 依賴 US1+US2+US3 全部完成
            ↓
        Phase 7 (Polish)
```

### Critical Path

最長路徑 (無並行):
```
Setup (6 tasks) → Foundation (5 tasks) → US1 (6 tasks) → US2 (6 tasks) → US4 (6 tasks) → Polish (7 tasks)
總計: 36 tasks
```

### Parallel Optimization

最佳並行執行 (理想狀態):
```
Setup (2 rounds) → Foundation (5 tasks) → US1 (3 rounds) → US2+US3 並行 (各3 rounds) → US4 (4 rounds) → Polish (2 rounds)
總計: 約 19 rounds (假設每 round 可並行3個任務)
```

---

## Task Summary

| Phase | Task Count | Parallelizable | User Story |
|-------|-----------|----------------|------------|
| Phase 1: Setup | 6 | 4 | - |
| Phase 2: Foundation | 5 | 0 | - |
| Phase 3: US1 (P1) | 6 | 3 | View Simplified 3-Category |
| Phase 4: US2 (P2) | 6 | 3 | View Aligned 5-Category |
| Phase 5: US3 (P2) | 6 | 3 | View Historical Validation |
| Phase 6: US4 (P3) | 6 | 3 | Dual-View Output |
| Phase 7: Polish | 7 | 5 | - |
| **Total** | **42** | **21** | **4 stories** |

**MVP Tasks**: 17 (Phase 1-2 + US1)
**Full Feature Tasks**: 42

---

## Validation Checklist

在完成所有任務後,驗證以下項目:

### 功能需求 (Functional Requirements)

- [ ] FR-001 至 FR-023: 所有23項功能需求已實作並通過測試

### 成功標準 (Success Criteria)

- [ ] SC-001: 使用者能在5秒內從3分類視圖判斷市場方向
- [ ] SC-002: 5分類橫條圖對齊後,識別準確度提升至95%以上
- [ ] SC-003: 3分類聚合機率計算誤差為0
- [ ] SC-004: 單次預測執行時間增加不超過30%
- [ ] SC-005: 視覺化圖表在1920x1080清晰可讀
- [ ] SC-006: 批次預測10個日期輸出格式一致且完整
- [ ] SC-007: 90%使用者能正確理解3分類和5分類關係
- [ ] SC-008: 系統處理邊界情況不產生錯誤
- [ ] SC-009: 3秒內判斷預測是否正確
- [ ] SC-010: 實際類別計算準確度100%
- [ ] SC-011: 95%使用者能理解預測信心度與準確性關係

### 憲章合規性 (Constitution Compliance)

- [ ] 所有程式碼使用正體中文 docstring
- [ ] 遵循 PEP 8 與 Pythonic 風格
- [ ] 無過度設計 (最簡單可行方案)
- [ ] 所有功能可獨立測試
- [ ] black + flake8 檢查通過

---

**最後更新**: 2025-11-16
**預估總工時**: 8-12 小時 (依開發者經驗)
**建議團隊規模**: 1-2 人 (單人可完成)
