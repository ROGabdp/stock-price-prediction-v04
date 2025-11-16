# 技術研究：LSTM 台股價格預測系統

**Phase**: 0 (研究與技術選型)
**日期**: 2025-11-15
**目的**: 解決技術背景中的未知項目，並為關鍵技術決策提供依據

## 研究摘要

本文件記錄了 LSTM 台股預測系統的技術選型研究結果。主要研究領域包含：深度學習框架選擇、超參數調整工具比較、特徵縮放策略、GPU 環境配置、以及測試策略。所有決策均基於專案需求（GPU 訓練、自動超參數調整、特徵集選擇）與憲章要求（Pythonic、可測試性、禁止過度設計）。

---

## 決策 1: 深度學習框架選擇

### 選擇: TensorFlow 2.x + Keras

**理由**:
1. **GPU 支援成熟**: TensorFlow-GPU 在 WSL2 + CUDA 環境下支援良好，安裝與配置文件完整
2. **Keras 高階 API**: 提供直觀的 Sequential/Functional API 建構 LSTM 模型，符合「簡潔務實」原則
3. **Keras Tuner 整合**: 官方支援的超參數調整工具，與 Keras 模型無縫整合
4. **SavedModel 格式**: 標準化的模型儲存格式，方便部署與載入
5. **社群資源豐富**: LSTM 時間序列預測的範例與最佳實踐文件充足

**考慮的替代方案**:
- **PyTorch**: 優點是更靈活的動態計算圖與研究友善性；缺點是需額外學習曲線，且 Optuna 整合雖可行但不如 Keras Tuner 直觀
- **決策**: 選擇 TensorFlow + Keras，因為本專案為 MVP，優先考慮開發效率與穩定性，而非研究靈活性

**版本要求**:
- TensorFlow: 2.10+ (支援 Python 3.9-3.11)
- CUDA: 11.2+ (與 TensorFlow 2.10+ 相容)
- cuDNN: 8.1+

---

## 決策 2: 超參數調整工具

### 選擇: Keras Tuner (主要) + Optuna (備選)

**理由**:
1. **Keras Tuner 優勢**:
   - 與 TensorFlow/Keras 原生整合，API 設計一致
   - 支援多種搜尋演算法 (RandomSearch, Hyperband, BayesianOptimization)
   - 內建試驗紀錄與可視化功能
   - 特徵集選擇可透過 `hp.Choice()` 簡單實現

2. **Optuna 作為備選**:
   - 框架無關，可用於任意 ML 框架
   - 更先進的 TPE (Tree-structured Parzen Estimator) 演算法
   - 若 Keras Tuner 效能不佳可快速切換

**實作策略**:
- **Phase 1 (基準模型)**: 使用 Keras Tuner RandomSearch 快速驗證可行性
- **Phase 2 (最佳化)**: 若需更高效搜尋，可切換至 Keras Tuner BayesianOptimization 或 Optuna

**考慮的替代方案**:
- **Scikit-learn GridSearchCV**: 不適用於深度學習，且無法處理特徵集選擇
- **Hyperopt**: 功能強大但配置複雜，不符合「簡潔務實」原則
- **決策**: Keras Tuner 為主，保留 Optuna 作為備選以降低風險

---

## 決策 3: 特徵縮放策略

### 選擇: StandardScaler (主要) + MinMaxScaler (備選)

**理由**:
1. **StandardScaler 優勢**:
   - 將特徵標準化為均值 0、標準差 1
   - 對 LSTM 模型更友善，避免梯度爆炸/消失
   - 不受異常值影響較小 (相比 MinMaxScaler)

2. **MinMaxScaler 適用情境**:
   - 將特徵縮放至 [0, 1] 區間
   - 適用於已知特徵範圍且無極端值的情境
   - 可用於籌碼資料等有明確界限的特徵

**實作策略**:
- **預設使用 StandardScaler** 對所有特徵進行縮放
- **特徵集選擇時比較**: 在超參數調整中可測試不同縮放器的效果
- **分別縮放**: 對訓練/驗證/測試集使用相同的 scaler (fit on training set only)

**考慮的替代方案**:
- **RobustScaler**: 對異常值更穩健，但本專案資料已清理，不需額外穩健性
- **No Scaling**: LSTM 對未縮放資料敏感，會導致訓練不穩定
- **決策**: StandardScaler 為主，保留 MinMaxScaler 作為實驗選項

---

## 決策 4: GPU 環境配置與檢查機制

### 選擇: TensorFlow GPU 可用性檢查 + 自動降級

**理由**:
1. **環境需求明確**: WSL2 (Ubuntu_D) + NVIDIA GPU + CUDA 11.2+ + cuDNN 8.1+
2. **可用性檢查**: 使用 `tf.config.list_physical_devices('GPU')` 檢測 GPU
3. **自動降級機制**: 若 GPU 不可用，記錄警告訊息並使用 CPU 訓練
4. **記憶體管理**: 使用 `tf.config.experimental.set_memory_growth()` 防止 GPU 記憶體溢位

**實作策略**:
```python
# 範例: GPU 檢查與配置
import tensorflow as tf

def setup_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # 允許 GPU 記憶體動態增長
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✅ GPU 可用: {len(gpus)} 個 GPU")
            return True
        except RuntimeError as e:
            print(f"⚠️ GPU 配置失敗: {e}")
            return False
    else:
        print("⚠️ 未偵測到 GPU，使用 CPU 訓練")
        return False
```

**考慮的替代方案**:
- **強制 GPU**: 不提供降級機制，GPU 不可用時報錯退出
  - 缺點: 降低系統彈性，不符合憲章「自動降級機制」要求
- **決策**: 實作自動檢查與降級，同時記錄 GPU 使用狀態

---

## 決策 5: 測試策略

### 選擇: pytest + 分層測試 (單元測試 + 整合測試)

**理由**:
1. **pytest 優勢**:
   - Python 社群標準測試框架
   - 支援 fixture、parametrize、mock 等進階功能
   - 清晰的測試報告與錯誤訊息

2. **分層測試策略**:
   - **單元測試**: 測試獨立函式 (如 `data_loader`, `feature_engineer`)
   - **整合測試**: 測試完整流程 (如資料載入 → 特徵工程 → 模型訓練)
   - **端到端測試**: 驗證完整的訓練與預測流程

**測試覆蓋範圍**:
| 測試類型 | 測試對象 | 範例測試案例 |
|---------|---------|-------------|
| 單元測試 | `data_loader.py` | 測試 CSV 載入、欄位驗證、缺失值處理 |
| 單元測試 | `feature_engineer.py` | 測試特徵工程邏輯、Set A/B/C 定義正確性 |
| 單元測試 | `lstm_baseline.py` | 測試模型建構、輸入/輸出形狀正確性 |
| 整合測試 | 資料處理流程 | 測試資料載入 → 特徵工程 → 訓練集分割 |
| 整合測試 | 訓練流程 | 測試完整訓練流程（使用小資料集快速驗證）|
| 整合測試 | 預測流程 | 測試模型載入 → 預測 → 結果格式化 |

**考慮的替代方案**:
- **unittest**: Python 內建，但語法較冗長，不如 pytest 簡潔
- **只做整合測試**: 忽略單元測試會降低錯誤定位效率
- **決策**: 使用 pytest + 分層測試，平衡覆蓋率與開發效率

---

## 決策 6: 特徵集實作策略

### 選擇: 配置驅動 (Config-driven) 特徵集定義

**理由**:
1. **可維護性**: 將 Set A/B/C 定義集中在 `feature_sets.py` 模組
2. **可擴展性**: 新增特徵集無需修改核心邏輯，只需新增配置
3. **可測試性**: 可獨立測試特徵集定義的正確性

**實作策略**:
```python
# 範例: feature_sets.py
FEATURE_SETS = {
    "Set A": {
        "name": "動能型",
        "features": [
            "price_returns",  # 價格變化率
            "volume_change",  # Volume 變化率
            "sma_ratio",      # SMA 比例
            "macd_features",  # MACD (DIF12-26, MACD9)
            "institutional"   # 法人籌碼
        ],
        "exclude": ["kd_features", "osc"]
    },
    "Set B": {
        "name": "震盪型",
        "features": [
            "price_returns",
            "volume_change",
            "sma_ratio",
            "kd_features",    # K(9,3), D(9,3)
            "institutional"
        ],
        "exclude": ["macd_features"]
    },
    "Set C": {
        "name": "全特徵集",
        "features": "all",  # 所有處理過的特徵
        "exclude": []
    }
}
```

**考慮的替代方案**:
- **硬編碼特徵集**: 在每個函式中重複定義特徵清單
  - 缺點: 維護困難，容易不一致
- **動態計算**: 在訓練時動態決定特徵集
  - 缺點: 增加複雜度，不符合「簡潔務實」原則
- **決策**: 使用配置驅動方式，集中管理特徵集定義

---

## 決策 7: 資料集分割策略

### 選擇: 時間序列分割 (Time-based Split)

**理由**:
1. **避免資料洩漏**: 時間序列預測必須確保訓練集在驗證/測試集之前
2. **符合實際使用情境**: 使用歷史資料預測未來，符合業務邏輯
3. **分割比例**: 70% 訓練 / 15% 驗證 / 15% 測試（約 4,900 / 1,000 / 900 筆）

**實作策略**:
```python
# 範例: 時間序列分割
def time_based_split(data, train_ratio=0.7, val_ratio=0.15):
    n = len(data)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]

    return train_data, val_data, test_data
```

**考慮的替代方案**:
- **隨機分割 (Random Split)**: 不適用於時間序列，會產生資料洩漏
- **K-Fold Cross-Validation**: 不適用於時間序列預測
- **決策**: 使用時間序列分割，確保資料順序正確性

---

## 決策 8: 訓練日誌與監控

### 選擇: TensorFlow Callbacks + 自定義日誌記錄

**理由**:
1. **Early Stopping**: 監控驗證損失，Patience=10，防止過擬合
2. **ModelCheckpoint**: 自動儲存最佳模型（基於驗證損失）
3. **CSV Logger**: 記錄訓練歷史至 CSV 檔案，方便後續分析
4. **自定義 Logger**: 記錄 GPU 使用狀態、記憶體資訊

**實作策略**:
```python
# 範例: Callbacks 配置
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger

callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    ),
    ModelCheckpoint(
        filepath='models/best_model.h5',
        monitor='val_loss',
        save_best_only=True
    ),
    CSVLogger('logs/training_log.csv')
]
```

**考慮的替代方案**:
- **TensorBoard**: 功能強大但配置複雜，超出 MVP 需求
- **MLflow**: 完整的 MLOps 平台，但對本專案過於複雜
- **決策**: 使用內建 Callbacks + 簡單日誌，符合「禁止過度設計」原則

---

## 研究結論

所有關鍵技術決策已完成，無未解決的 NEEDS CLARIFICATION 項目。技術選型優先考慮：
1. **成熟穩定**: TensorFlow 2.x + Keras 在 GPU 環境支援良好
2. **簡潔務實**: Keras Tuner 整合簡單，避免過度複雜的工具
3. **可測試性**: pytest + 分層測試確保品質
4. **符合憲章**: 所有決策符合 Pythonic、GPU 加速、禁止過度設計等原則

下一階段將進入 **Phase 1: 資料模型與合約設計**。
