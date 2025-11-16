# 快速開始指南（更新版：使用 TensorFlow 2.16.1）

## 🎉 重大簡化

**使用新的套件配置後，GPU 設定變得超簡單！**

- ✅ `tensorflow[and-cuda]==2.16.1` 自動安裝 CUDA 和 cuDNN
- ✅ Python 3.12（Ubuntu 24.04 預設版本，無需額外安裝）
- ✅ 無需手動安裝 CUDA Toolkit

---

## 建議的安裝流程

### 前置條件

1. **Windows 端已安裝** NVIDIA CUDA on WSL 驅動程式
   - 檢查：在 Windows PowerShell 執行 `nvidia-smi`
   - 下載：https://developer.nvidia.com/cuda/wsl（如果未安裝）

2. **WSL2 Ubuntu 環境**已設置

---

## 快速安裝步驟

### 步驟 1: 確認 Python 3.12（Ubuntu 24.04 預設已安裝）

```bash
# 在 WSL2 終端機執行
python3 --version  # 應顯示: Python 3.12.x

# 如果是 Ubuntu 22.04，需要安裝 Python 3.12
# sudo apt update
# sudo apt install software-properties-common -y
# sudo add-apt-repository ppa:deadsnakes/ppa -y
# sudo apt update
# sudo apt install python3.12 python3.12-venv python3.12-dev

# 安裝必要套件
sudo apt update
sudo apt install python3-pip python3-venv python3-dev build-essential
```

### 步驟 2: 建立虛擬環境

```bash
# 切換到專案目錄
cd /mnt/d/000-github-repositories/stock-price-prediction-v04

# 建立虛擬環境（使用 Python 3.12）
python3.12 -m venv venv

# 啟動虛擬環境
source venv/bin/activate

# 確認 Python 版本
python --version  # 應顯示: Python 3.12.x
```

### 步驟 3: 安裝依賴套件

```bash
# 升級 pip
pip install --upgrade pip

# 安裝所有依賴（包含 TensorFlow + CUDA 自動安裝）
pip install -r requirements.txt
```

**安裝時間提示**：
- 首次安裝約 5-10 分鐘
- 會自動下載 CUDA 12.3 和 cuDNN 8.9（約 2-3 GB）
- 無需任何手動 CUDA 設定

### 步驟 4: 驗證安裝

```bash
# 執行系統驗證腳本
python validate_pipeline.py
```

**預期結果**：
- ✅ Python 3.12.x
- ✅ TensorFlow 2.16.1
- ✅ GPU 裝置偵測（如果有 NVIDIA GPU）
- ✅ 所有必要套件安裝完成

---

## GPU 支援確認

```bash
# 檢查 GPU 是否可用
python -c "import tensorflow as tf; print('GPU 裝置:', tf.config.list_physical_devices('GPU'))"
```

**可能結果**：

### 情況 1: GPU 正常工作 ✅
```
GPU 裝置: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```
→ **完美！可以開始使用 GPU 加速訓練**

### 情況 2: GPU 未偵測 ⚠️
```
GPU 裝置: []
```
→ **系統會自動使用 CPU，功能完全相同**
→ 如需啟用 GPU，請參考 [WSL2_GPU_SETUP.md](WSL2_GPU_SETUP.md)

---

## 不再需要的步驟 ❌

~~以下步驟已被 `tensorflow[and-cuda]` 取代，無需執行~~：

- ~~手動安裝 CUDA Toolkit~~
- ~~手動安裝 cuDNN~~
- ~~設定 CUDA 環境變數~~
- ~~建立符號連結~~

---

## 開始使用系統

安裝完成後，您可以直接開始使用系統！

### 測試訓練（快速驗證）

```bash
# 快速訓練測試（10 epochs）
python src/cli/train.py \
    --data-file 19980601-20251111-converted.csv \
    --feature-set "Set A" \
    --time-steps 60 \
    --epochs 10 \
    --batch-size 16
```

**系統會自動**：
- 偵測 GPU 可用性（有 GPU 就用 GPU，沒有就用 CPU）
- 顯示訓練進度與指標
- 儲存訓練好的模型

### 完整訓練流程

### 1. 訓練基準模型（約 1-2 小時，GPU 模式）

```bash
python src/cli/train.py \
    --data-file 19980601-20251111-converted.csv \
    --feature-set "Set A" \
    --time-steps 60 \
    --epochs 100 \
    --batch-size 32 \
    --output-dir models/baseline
```

### 2. 超參數調整（約 6-12 小時，GPU 模式）

```bash
python src/cli/tune.py \
    --data-file 19980601-20251111-converted.csv \
    --max-trials 50 \
    --epochs-per-trial 50 \
    --tuner-type bayesian \
    --output-dir models/tuning
```

### 3. 預測未來價格（< 10 秒）

```bash
python src/cli/predict.py \
    --model-file models/best_tuned_model.h5 \
    --data-file 19980601-20251111-converted.csv \
    --input-date "2024-01-15" \
    --feature-set "Set A" \
    --time-steps 60
```

---

## 效能對比

| 操作 | CPU (估計) | GPU (估計) | 加速比 |
|------|------------|------------|--------|
| 訓練 100 epochs | 8-12 小時 | 1-2 小時 | 5-10x |
| 超參數調整 50 trials | 2-3 天 | 6-12 小時 | 4-6x |
| 預測單次 | 5-10 秒 | 1-3 秒 | 2-3x |

---

## 詳細文件

- **完整安裝指南**: [README.md](README.md)
- **GPU 故障排除**: [WSL2_GPU_SETUP.md](WSL2_GPU_SETUP.md)
- **實作細節**: [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)

---

## 常見問題

### Q1: 安裝需要多久時間？

**回答**：
- Python 3.12：Ubuntu 24.04 預設已安裝，無需時間
- pip 套件安裝：約 5-10 分鐘（首次安裝，包含下載 CUDA）
- 總計：約 5-10 分鐘（Ubuntu 24.04）

### Q2: 一定要有 GPU 嗎？

**回答**：
- 不一定！系統會自動偵測硬體
- 沒有 GPU 會自動使用 CPU（功能完全相同）
- GPU 主要影響訓練速度（快 5-10 倍）

### Q3: TensorFlow 版本為什麼選擇 2.16.1？

**回答**：
- 2.16.1 支援 `tensorflow[and-cuda]` 自動安裝 CUDA
- 與 Python 3.12、NumPy 1.26.2 相容性最佳
- 無需手動安裝 CUDA Toolkit 或 cuDNN

### Q4: 如果 GPU 沒有被偵測到怎麼辦？

**回答**：
1. 確認 Windows 已安裝 NVIDIA CUDA on WSL 驅動
   - 在 Windows PowerShell 執行 `nvidia-smi`
   - 如未安裝，從 https://developer.nvidia.com/cuda/wsl 下載
2. 重新啟動虛擬環境：`deactivate && source venv/bin/activate`
3. 詳細故障排除見 [WSL2_GPU_SETUP.md](WSL2_GPU_SETUP.md)
4. 如果無法解決，系統會自動使用 CPU（功能完全相同）

### Q5: 套件版本可以自己調整嗎？

**回答**：
- **不建議**！推薦的版本組合已經過測試
- Python 3.12 + TensorFlow 2.16.1 + NumPy 1.26.2 相容性最佳
- 自行調整可能導致相容性問題

---

## 🎉 開始您的 LSTM 台股預測之旅！

按照上述步驟，您將在 **5-10 分鐘內**完成安裝並開始訓練模型（Ubuntu 24.04）。祝您使用愉快！
