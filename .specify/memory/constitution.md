<!--
================================================================================
Sync Impact Report
================================================================================
Version Change: [NEW] → 1.0.0
Modified Principles: N/A (Initial constitution)
Added Sections:
  - 核心原則 (Core Principles) with 6 principles
  - 技術規範 (Technical Standards)
  - 開發流程 (Development Workflow)
  - 治理 (Governance)
Removed Sections: N/A
Templates Requiring Updates:
  ✅ .specify/templates/plan-template.md - reviewed, compatible
  ✅ .specify/templates/spec-template.md - reviewed, compatible
  ✅ .specify/templates/tasks-template.md - reviewed, compatible
Follow-up TODOs: None
================================================================================
-->

# 台股預測模型專案憲章

## 核心原則

### I. 正體中文優先

所有文件、程式碼註解及使用者介面文字，**必須 (MUST)** 使用正體中文撰寫。

**理由**: 確保專案的可讀性與維護性，降低語言障礙，提升團隊協作效率。

### II. Pythonic 程式碼風格

所有 Python 程式碼 **必須 (MUST)** 遵循 PEP 8 規範，並保持 Pythonic 的撰寫風格。

**理由**: 統一的程式碼風格可提升可讀性、降低維護成本，並符合 Python 社群的最佳實踐。

**具體要求**:
- 使用 4 個空格進行縮排
- 函式與類別命名遵循 PEP 8 (snake_case 與 PascalCase)
- 每行最多 79 個字元 (程式碼) 或 72 個字元 (註解與文件字串)
- 使用型別提示 (Type Hints) 提升程式碼清晰度
- 必須使用 `pylint`、`flake8` 或 `black` 等工具進行程式碼檢查與格式化

### III. 規格驅動開發 (Specification-Driven Development)

本專案 **必須 (MUST)** 堅守「規格驅動開發」的核心原則，專注於交付一個高品質、可測試 (Testable) 的 Minimum Viable Product (MVP)。

**理由**: 透過明確的規格文件，確保需求清晰、設計有據，避免過度設計，提升開發效率。

**具體要求**:
- 所有功能 **必須 (MUST)** 先撰寫規格文件 (spec.md)
- 規格文件 **必須 (MUST)** 包含使用者情境、測試場景、需求與成功標準
- 實作 **必須 (MUST)** 完全符合規格文件的定義
- 任何需求變更 **必須 (MUST)** 先更新規格文件後才能修改實作

### IV. 禁止過度設計 (No Overdesign)

所有設計與實作 **必須 (MUST)** 保持簡潔與務實，**嚴格禁止 (MUST NOT)** 任何過度設計。

**理由**: 過度設計會增加系統複雜度、降低可維護性，並延遲交付時程。應遵循 YAGNI (You Aren't Gonna Need It) 原則。

**具體要求**:
- 優先實作當前需求，不預先設計未來可能用到的功能
- 避免引入不必要的抽象層或設計模式
- 每個設計決策 **必須 (MUST)** 有明確的業務需求支持
- 如需引入複雜設計，**必須 (MUST)** 在實作計畫 (plan.md) 中的「複雜度追蹤」表格中說明理由

### V. GPU 加速機器學習訓練

所有機器學習訓練任務 **必須 (MUST)** 使用 GPU 進行加速，運用 WSL2 + NVIDIA GPU 環境。

**理由**: GPU 可大幅縮短訓練時間，提升模型迭代效率，特別是對於深度學習模型 (如 LSTM) 的訓練。

**具體要求**:
- 訓練環境: WSL 發行版名稱為 `Ubuntu_D`，路徑為 `D:\wsl\Ubuntu_D`
- 必須使用 TensorFlow-GPU 或 PyTorch (with CUDA support)
- 訓練腳本 **必須 (MUST)** 包含 GPU 可用性檢查與自動降級機制
- 訓練日誌 **必須 (MUST)** 記錄 GPU 使用狀態與記憶體資訊

### VI. 可測試性優先 (Testability First)

所有實作 **必須 (MUST)** 以可測試性為優先考量，確保每個功能都可以被獨立測試與驗證。

**理由**: 可測試性是高品質軟體的基石，可確保功能正確性、降低迴歸風險，並提升重構信心。

**具體要求**:
- 每個使用者情境 (User Story) **必須 (MUST)** 包含獨立的測試場景
- 優先使用 Given-When-Then 格式定義驗收測試
- 模型訓練與預測功能 **必須 (MUST)** 可獨立執行與驗證
- 測試涵蓋範圍包含: 資料預處理、模型訓練、模型預測、結果輸出

## 技術規範

### 程式語言與版本

- **程式語言**: Python 3.9+
- **深度學習框架**: TensorFlow 2.x (with GPU support) 或 PyTorch (with CUDA support)
- **版本控制**: Git，預設分支名稱為 `main`

### 開發環境

- **作業系統**: WSL2 (Ubuntu_D，路徑: D:\wsl\Ubuntu_D)
- **GPU**: NVIDIA GPU with CUDA support
- **Python 套件管理**: pip + virtualenv 或 conda

### 程式碼品質工具

- **程式碼格式化**: `black` (必須使用)
- **程式碼檢查**: `pylint` 或 `flake8` (必須通過)
- **型別檢查**: `mypy` (建議使用)
- **Import 排序**: `isort` (建議使用)

### 文件規範

- **規格文件**: 必須使用 `.specify/` 框架提供的模板
- **程式碼註解**: 所有函式與類別必須包含 docstring (使用正體中文)
- **README**: 必須包含專案說明、安裝步驟、使用方式與訓練流程

## 開發流程

### 規格驅動流程

1. **規格定義** (`/speckit.specify`): 撰寫功能規格 (spec.md)，定義使用者情境、需求與成功標準
2. **實作計畫** (`/speckit.plan`): 撰寫實作計畫 (plan.md)，定義技術方案、專案結構與複雜度追蹤
3. **任務拆解** (`/speckit.tasks`): 將實作計畫拆解為可執行的任務清單 (tasks.md)
4. **實作執行** (`/speckit.implement`): 按照任務清單逐步實作功能
5. **測試驗證**: 執行測試場景，確保功能符合規格定義
6. **文件更新**: 更新 README 與相關文件

### Git 工作流程

- **分支命名**: `feature/###-feature-name` (### 為編號)
- **提交訊息**: 使用正體中文，簡潔描述變更內容
- **Pull Request**: 必須包含規格文件連結與測試結果
- **程式碼審查**: 所有 PR 必須通過程式碼審查與自動化檢查

## 治理

### 憲章修訂程序

1. **提案**: 任何憲章修訂必須先提出提案，說明修訂理由與影響範圍
2. **審查**: 提案必須經過團隊審查與討論
3. **版本更新**: 憲章版本遵循語義化版本規範 (Semantic Versioning):
   - **MAJOR**: 移除或重新定義核心原則 (向後不相容的治理變更)
   - **MINOR**: 新增原則或重大擴充指引
   - **PATCH**: 釐清、措辭修正、錯字修正、非語義性調整
4. **同步更新**: 修訂後必須更新所有相依的模板與文件

### 合規性審查

- 所有 Pull Request **必須 (MUST)** 驗證是否符合本憲章的核心原則
- 實作計畫 (plan.md) **必須 (MUST)** 包含「憲章檢查」(Constitution Check) 區塊
- 任何違反憲章的設計決策 **必須 (MUST)** 在「複雜度追蹤」表格中說明理由

### 持續改進

- 定期審查憲章的適用性與有效性
- 根據專案發展與團隊回饋，適時調整憲章內容
- 保持憲章的簡潔性與可執行性

---

**版本**: 1.0.0 | **批准日期**: 2025-11-15 | **最後修訂**: 2025-11-15
