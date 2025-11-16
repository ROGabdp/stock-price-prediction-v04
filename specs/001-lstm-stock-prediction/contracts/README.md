# 模組介面合約

本目錄包含 LSTM 台股預測系統各模組的介面定義合約。

## 合約清單

1. **data_preprocessing.md** - 資料載入與前處理模組
2. **feature_engineering.md** - 特徵工程模組
3. **model_training.md** - 模型訓練模組
4. **hyperparameter_tuning.md** - 超參數調整模組
5. **prediction.md** - 預測模組

## 合約規範

每個合約文件包含：
- **模組職責**: 該模組的主要功能
- **公開介面**: 函式簽章、輸入/輸出規格
- **錯誤處理**: 異常情況與錯誤訊息
- **使用範例**: 典型使用場景的程式碼範例

## 依賴關係

```
data_preprocessing
    ↓
feature_engineering
    ↓
model_training ← hyperparameter_tuning
    ↓
prediction
```
