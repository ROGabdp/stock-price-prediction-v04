# 快速開始指南：LSTM 台股價格預測系統

**目的**: 提供開發人員快速上手指南，從環境設置到執行完整訓練與預測流程

## 前置需求

### 環境需求
- **作業系統**: WSL2 (Ubuntu_D, D:\wsl\Ubuntu_D)
- **Python**: 3.9+
- **GPU**: NVIDIA GPU with CUDA 11.2+ and cuDNN 8.1+
- **記憶體**: 建議 16GB+ RAM, 6GB+ VRAM

### 資料需求
- **歷史資料檔案**: `19980601-20251111-converted.csv` (專案根目錄)
- **資料範圍**: 1998/6/1 - 2025/11/11 (約 6,800 筆交易日)

---

## 步驟 1: 環境設置

### 1.1 進入 WSL2 環境

```bash
# 從 Windows 啟動 WSL2
wsl -d Ubuntu_D
```

### 1.2 建立 Python 虛擬環境

```bash
# 切換至專案目錄
cd /mnt/d/000-github-repositories/stock-price-prediction-v04

# 建立虛擬環境
python3.9 -m venv venv

# 啟動虛擬環境
source venv/bin/activate
```

### 1.3 安裝依賴套件

```bash
# 安裝 TensorFlow GPU
pip install tensorflow-gpu==2.10.0

# 安裝其他依賴
pip install pandas numpy scikit-learn keras-tuner matplotlib

# 安裝開發工具
pip install pytest black flake8 mypy
```

### 1.4 驗證 GPU 可用性

```python
python -c "import tensorflow as tf; print('GPU 可用:', len(tf.config.list_physical_devices('GPU')) > 0)"
```

**預期輸出**: `GPU 可用: True`

---

## 步驟 2: 訓練基準模型

### 2.1 執行訓練腳本

```bash
# 訓練基準模型 (使用 Set A 特徵集)
python src/cli/train.py \
    --data-file 19980601-20251111-converted.csv \
    --feature-set "Set A" \
    --time-steps 60 \
    --epochs 100 \
    --batch-size 32 \
    --output-dir models/
```

### 2.2 預期輸出

```
✅ 載入資料: 6,823 筆交易日
✅ 特徵工程完成 (Set A): 14 個特徵
✅ 資料集分割:
   訓練集: 4,776 筆
   驗證集: 1,024 筆
   測試集: 1,023 筆
✅ GPU 可用: 1 個 GPU
✅ 模型建構完成 (3 層 LSTM: 128-64-32)

訓練中...
Epoch 1/100: loss: 1.2345, val_loss: 1.1234, val_accuracy: 0.42
Epoch 2/100: loss: 1.1234, val_loss: 1.0987, val_accuracy: 0.45
...
Epoch 45/100: loss: 0.9123, val_loss: 0.9567, val_accuracy: 0.52
Early Stopping triggered (patience=10)

✅ 訓練完成
   最佳驗證損失: 0.9567
   最佳驗證準確度: 52.3%
   訓練時間: 87 分鐘
   模型已儲存至: models/baseline_model.h5
```

---

## 步驟 3: 超參數調整 (選用)

### 3.1 執行超參數調整

```bash
# 執行 50 次試驗，探索最佳模型配置 (含特徵集選擇)
python src/cli/tune.py \
    --data-file 19980601-20251111-converted.csv \
    --max-trials 50 \
    --epochs-per-trial 50 \
    --output-dir models/
```

### 3.2 預期輸出

```
✅ Keras Tuner 初始化完成
✅ 搜尋空間:
   - 時間窗口: [20, 40, 60, 80]
   - LSTM 層數: [2, 3, 4]
   - 單元數: [32, 64, 128]
   - Dropout: [0.1, 0.4]
   - 學習率: [0.001, 0.0005, 0.0001]
   - 批次大小: [32, 64, 128]
   - 特徵集: [Set A, Set B, Set C]

Trial 1/50
  超參數: time_steps=60, layers=3, feature_set=Set A, lr=0.001, batch=32
  驗證損失: 0.9567
Trial 2/50
  超參數: time_steps=40, layers=4, feature_set=Set B, lr=0.0005, batch=64
  驗證損失: 0.9234
...
Trial 50/50
  超參數: time_steps=80, layers=2, feature_set=Set C, lr=0.0001, batch=128
  驗證損失: 1.0123

✅ 超參數調整完成
   最佳試驗 ID: Trial 28
   最佳超參數:
     - 時間窗口: 60
     - LSTM 層數: 3
     - 單元數: [128, 96, 64]
     - Dropout: 0.3
     - 學習率: 0.0005
     - 批次大小: 64
     - 特徵集: Set B (震盪型)
   最佳驗證損失: 0.8945
   最佳驗證準確度: 55.7%
   總執行時間: 6.5 小時
   最佳模型已儲存至: models/best_tuned_model.h5
```

---

## 步驟 4: 執行預測

### 4.1 使用指定日期預測

```bash
# 預測 2024-01-15 後 20 個交易日的漲跌幅
python src/cli/predict.py \
    --model-file models/best_tuned_model.h5 \
    --data-file 19980601-20251111-converted.csv \
    --input-date "2024-01-15" \
    --feature-set "Set B" \
    --time-steps 60
```

### 4.2 預期輸出

```
✅ 模型載入成功: models/best_tuned_model.h5
✅ 歷史資料載入: 6,823 筆
✅ 輸入日期驗證通過: 2024-01-15

=== 台股 20 日預測結果 ===
輸入日期: 2024-01-15
預測日期: 2024-02-14 (20 個交易日後)

預測結果:
  類別: 3 (溫和上漲)
  信心度: 68.5%
  預測收盤價區間: 15,800 - 16,500 元

實際收盤價: 16,120 元 ✅ (落在預測區間內)

機率分佈:
  極度下跌 (<-5.0%):   2.3% ░
  溫和下跌 (-5.0~-2.5%): 8.7% ███
  區間震盪 (-2.5~+2.5%): 12.4% ████
  溫和上漲 (+2.5~+5.0%): 68.5% ████████████████████████ ← 預測
  極度上漲 (>+5.0%):   8.1% ███

預測時間: 2025-11-15 14:35:21
執行時間: 3.2 秒
```

---

## 步驟 5: 執行測試 (開發階段)

### 5.1 執行單元測試

```bash
# 執行所有單元測試
pytest tests/unit/ -v

# 執行特定模組測試
pytest tests/unit/test_feature_engineer.py -v
```

### 5.2 執行整合測試

```bash
# 執行整合測試 (使用小資料集)
pytest tests/integration/ -v
```

### 5.3 執行程式碼品質檢查

```bash
# 執行 black 格式化
black src/ tests/

# 執行 flake8 檢查
flake8 src/ tests/ --max-line-length=79

# 執行 mypy 型別檢查
mypy src/
```

---

## 常見問題

### Q1: GPU 不可用怎麼辦？

**A**: 系統會自動降級使用 CPU 訓練，但速度較慢。請檢查：
1. CUDA 與 cuDNN 是否正確安裝
2. TensorFlow-GPU 版本是否與 CUDA 版本相容
3. WSL2 的 NVIDIA GPU 驅動是否啟用

### Q2: 訓練時 GPU 記憶體不足？

**A**: 調整以下參數：
- 減少 `batch_size` (例如從 64 降至 32)
- 減少 `time_steps` (例如從 80 降至 60)
- 在程式碼中啟用 GPU 記憶體動態增長

### Q3: 驗證準確度低於 40% 怎麼辦？

**A**: 檢查以下項目：
1. 特徵工程是否正確 (檢查特徵矩陣形狀)
2. 目標變數分類是否平衡 (檢查各類別樣本數)
3. 模型架構是否合理 (嘗試調整層數與單元數)
4. 學習率是否過高或過低 (嘗試 0.001, 0.0005, 0.0001)

### Q4: 如何查看訓練歷史？

**A**: 查看 CSV 日誌檔案：
```bash
# 查看訓練歷史
cat logs/training_logs/training_log.csv

# 或使用 pandas 分析
python -c "import pandas as pd; df = pd.read_csv('logs/training_logs/training_log.csv'); print(df.head())"
```

---

## 後續步驟

1. **調整特徵集**: 嘗試不同的特徵集 (Set A/B/C) 比較效果
2. **優化超參數**: 增加 `max_trials` 以探索更多參數組合
3. **評估模型**: 使用測試集評估模型泛化能力
4. **部署模型**: 將最佳模型整合至生產環境

---

## 相關文件

- [功能規格](spec.md) - 完整的功能需求與使用者情境
- [實作計畫](plan.md) - 技術架構與專案結構
- [資料模型](data-model.md) - 資料實體定義與驗證規則
- [模組合約](contracts/) - 各模組的介面定義與使用範例
