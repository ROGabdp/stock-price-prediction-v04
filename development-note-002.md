/speckit.constitution 

所有文件、註解及使用者介面文字，一律使用正體中文。
在程式碼風格上，所有 Python 程式碼必須保持 Pythonic 並嚴格遵循 PEP 8 規範。
Git 預設分支名稱設定為 main。
本專案將堅守「規格驅動開發」的核心原則，專注於交付一個高品質、可測試 (Testable) 的 Minimum Viable Product (MVP)，並嚴格禁止任何過度設計 (Overdesign)，確保實作方案的簡潔與務實。
用GPU進行機器學習的訓練。運用WSL2 + NVIDIA GPU。
已將wsl2 Utunbu_D 安裝在D槽，WSL 發行版名稱是 Ubuntu_D，路徑為 D:\wsl\Ubuntu_D。

/speckit.specify

以下提供資訊為MD格式。

開一個新的檔案來寫。

套用目前的程式演算法來預測未來的股價漲跌，但除了目前可以看到的5類別漲跌幅區間，我還要把極度下跌和溫和下跌的機率加總，成為"看跌"，把極度上漲和溫和上漲的機率加總，成為"看漲"，並把這三類也視覺化呈現。

此外，5分類的結果也要同步呈現，並且目前極度下跌和極度上漲的橫條圖沒有和其他的3個分類對齊，造成我不容易閱讀，這個部分要把每個分類的橫條圖起點對齊，更容易看出哪個類別機率最高。

新的功能中，要確保我能同時看到歷史資料的正確股價 (如果有的話)，以及5類別和3類別的預測，以及這些預測對照於歷史資料是否正確。 


/speckit.plan

使用Python 3.12 撰寫程式。 

關鍵設計原則
  1. 最小侵入性: 新增獨立模組,不修改現有程式碼
  2. 100%一致性: 複用 CLASS_MAPPING 確保分類邊界一致

/speckit.tasks
/speckit.implement


  ✅ Implementation Complete - Final Summary

  I've successfully implemented the Enhanced Stock Price Prediction Visualization feature with full
  end-to-end testing. Here's what was delivered:

  📦 Deliverables

  4 Core Modules:
  1. src/visualization/aggregator.py - 3-class aggregation logic (103 lines)
  2. src/visualization/validator.py - Historical validation (161 lines)
  3. src/visualization/plotter.py - Dual-view visualization (351 lines)
  4. src/cli/predict_enhanced.py - Enhanced CLI (463 lines)

  3 Test Suites:
  1. tests/unit/test_aggregator.py - 18 tests ✓ all passing
  2. tests/unit/test_validator.py - 22 tests ✓ all passing
  3. tests/unit/test_plotter.py - 17 tests ✓ all passing (1 skipped)

  🎯 End-to-End Test Results

  Using actual historical data (2024-01-15):
  - ✅ Model loaded: baseline_model_20251116_151143.h5
  - ✅ Data loaded: 6,823 records (1998-06-01 to 2025-11-11)
  - ✅ Prediction generated: 2024-02-21 (20 trading days later)
  - ✅ 3-class aggregation: 看漲 42.6% (correct!)
  - ✅ 5-class prediction: 區間震盪 30.8% (incorrect, actual was 極度上漲)
  - ✅ Historical validation: Compared predicted vs actual (+6.44% rise)
  - ✅ Visualization saved: outputs/prediction_2024-01-15_2024-02-21.png (75KB)
  - ✅ Execution time: 4.51 seconds (well under spec requirement)

  📊 Features Implemented (All 23 Functional Requirements)

  - FR-001 to FR-015: 3/5-class visualization with aligned bar charts ✓
  - FR-016 to FR-023: Historical validation with accuracy checking ✓
  - SC-001 to SC-011: All 11 success criteria met ✓

  🚀 Ready to Use

  # Single prediction
  python src/cli/predict_enhanced.py \
      --model-file models/baseline_model_20251116_151143.h5 \
      --data-file 19980601-20251111-converted.csv \
      --input-date "2024-01-15"

  # Batch prediction
  python src/cli/predict_enhanced.py \
      -m models/baseline_model_20251116_151143.h5 \
      -d 19980601-20251111-converted.csv \
      -i "2024-01-15" "2024-02-20" "2024-03-10"

  # Text-only (no visualization)
  python src/cli/predict_enhanced.py \
      -m models/baseline_model_20251116_151143.h5 \
      -d 19980601-20251111-converted.csv \
      -i "2024-01-15" \
      --no-plot

  All code has been formatted with black, includes full Chinese docstrings, and adheres to the project
  constitution. The feature is production-ready! 🎉

