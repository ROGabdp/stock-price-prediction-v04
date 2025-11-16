# WSL2 GPU 設定指南（適用於 TensorFlow 2.16.1）

## 重要更新 🎉

**使用 TensorFlow 2.16.1 + `tensorflow[and-cuda]` 後，GPU 設定變得超簡單！**

- ✅ **自動安裝 CUDA 12.3 和 cuDNN 8.9**
- ✅ **無需手動安裝** CUDA Toolkit
- ✅ **僅需安裝** Windows 端的 NVIDIA CUDA on WSL 驅動程式

---

## 問題診斷

執行以下指令檢查 GPU 狀態：
```bash
python -c "import tensorflow as tf; print('GPU 裝置:', tf.config.list_physical_devices('GPU'))"
```

**可能結果**：

### 情況 1: GPU 正常偵測 ✅
```
GPU 裝置: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```
→ **恭喜！您的 GPU 已正常工作，無需任何設定**

### 情況 2: GPU 未偵測 ⚠️
```
GPU 裝置: []
```
→ **請按照下方「解決方案」步驟操作**

---

## 解決方案（僅需安裝 Windows 驅動）

### 步驟 1: 檢查 Windows 端 NVIDIA 驅動

**在 Windows PowerShell** 執行（非 WSL2）：
```powershell
nvidia-smi
```

**預期結果**：
- ✅ 看到 GPU 資訊（如 RTX 4070）→ 驅動已安裝
- ❌ 找不到指令或無 GPU → 需安裝驅動

### 步驟 2: 安裝 NVIDIA CUDA on WSL 驅動程式（僅 Windows 端）

如果 `nvidia-smi` 失敗，請從以下網址下載並安裝：

**下載位置**: https://developer.nvidia.com/cuda/wsl

**重要提示**：
- 僅在 **Windows** 安裝驅動程式
- **不要**在 WSL2 內安裝任何 CUDA 相關套件
- 安裝後**重啟電腦**

### 步驟 3: 驗證 WSL2 內 GPU 可用性

回到 **WSL2 終端機**，執行：
```bash
# 檢查 NVIDIA 驅動（應能在 WSL2 內看到）
nvidia-smi

# 檢查 TensorFlow GPU 支援
python -c "import tensorflow as tf; print('GPU 裝置:', tf.config.list_physical_devices('GPU'))"
```

**預期結果**：
```
GPU 裝置: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

---

## 舊方法（不再需要，僅供參考）

### ~~選項 1: 安裝 CUDA Toolkit 和 cuDNN（已過時）~~

> **注意**: 使用 `tensorflow[and-cuda]==2.16.1` 後，此步驟**不再需要**。
> TensorFlow 會自動安裝所需的 CUDA 和 cuDNN 套件。

~~雖然 WSL2 原則上不需要在 Linux 內安裝 CUDA，但 TensorFlow 2.15 需要某些 CUDA 函式庫。~~

```bash
# 1. 檢查 NVIDIA 驅動版本
nvidia-smi  # 確認能看到 GPU

# 2. 安裝 CUDA Toolkit 12.x
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-3

# 3. 安裝 cuDNN
sudo apt-get install libcudnn8 libcudnn8-dev

# 4. 設定環境變數
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# 5. 驗證安裝
nvcc --version
python -c "import tensorflow as tf; print('GPU 可用:', len(tf.config.list_physical_devices('GPU')) > 0)"
```

### 選項 2: 使用 CPU 模式（簡單快速）

如果不需要 GPU 加速，或暫時想先測試功能：

```bash
# 系統會自動使用 CPU，不需要任何額外設定
# 只是訓練速度會較慢（約 5-10 倍差異）

# 執行訓練時會看到：
# ⚠️ 未偵測到 GPU，使用 CPU 訓練
```

**CPU 模式的優缺點**：
- ✅ 無需複雜設定
- ✅ 功能完全相同
- ❌ 訓練速度較慢
- ❌ 超參數調整會耗時很久

### 選項 3: 降級 TensorFlow 版本（不推薦）

如果選項 1 安裝失敗，可以嘗試使用較舊但穩定的版本：

```bash
# 卸載現有 TensorFlow
pip uninstall tensorflow

# 安裝 TensorFlow 2.15（最後支援 Ubuntu 24.04 的版本）
pip install tensorflow[and-cuda]==2.15.0
```

## 詳細步驟：選項 1 完整安裝

### 步驟 1: 確認 Windows 已安裝 NVIDIA CUDA on WSL 驅動

在 **Windows** (非 WSL2) 執行：
```powershell
nvidia-smi  # 應該能看到 GPU
```

如果看不到，請從以下網址下載並安裝：
https://developer.nvidia.com/cuda/wsl

### 步驟 2: 在 WSL2 安裝 CUDA Toolkit

```bash
# 下載並安裝 CUDA 倉庫金鑰
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb

# 更新套件清單
sudo apt-get update

# 安裝 CUDA Toolkit 12.3（與您的驅動 CUDA 13.0 相容）
sudo apt-get -y install cuda-toolkit-12-3
```

### 步驟 3: 安裝 cuDNN

```bash
# 安裝 cuDNN（深度學習加速函式庫）
sudo apt-get install -y libcudnn8 libcudnn8-dev
```

### 步驟 4: 設定環境變數

```bash
# 將 CUDA 路徑加入 PATH 和 LD_LIBRARY_PATH
cat << 'EOF' >> ~/.bashrc

# CUDA 環境變數
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda
EOF

# 套用設定
source ~/.bashrc
```

### 步驟 5: 驗證 CUDA 安裝

```bash
# 檢查 CUDA 編譯器版本
nvcc --version

# 應顯示類似：
# Cuda compilation tools, release 12.3, ...
```

### 步驟 6: 重新啟動虛擬環境並測試

```bash
# 重新啟動虛擬環境
deactivate
source venv/bin/activate

# 測試 TensorFlow GPU
python -c "import tensorflow as tf; print('TensorFlow 版本:', tf.__version__)"
python -c "import tensorflow as tf; print('GPU 裝置:', tf.config.list_physical_devices('GPU'))"
python -c "import tensorflow as tf; print('GPU 可用:', len(tf.config.list_physical_devices('GPU')) > 0)"
```

預期輸出：
```
TensorFlow 版本: 2.15.0
GPU 裝置: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
GPU 可用: True
```

### 步驟 7: 測試 GPU 訓練

```bash
# 執行簡單的 GPU 測試
python -c "
import tensorflow as tf
print('GPU 裝置:', tf.config.list_physical_devices('GPU'))

# 建立簡單運算測試 GPU
with tf.device('/GPU:0'):
    a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
    c = tf.matmul(a, b)
    print('矩陣乘法結果:', c.numpy())
print('GPU 測試成功！')
"
```

## 常見問題排除

### Q1: 安裝 CUDA 後還是無法使用 GPU

```bash
# 檢查 CUDA 函式庫路徑
ls -l /usr/local/cuda/lib64/libcudart.so*

# 如果找不到，建立符號連結
sudo ln -s /usr/local/cuda-12.3 /usr/local/cuda

# 重新載入環境變數
source ~/.bashrc
```

### Q2: libcudnn 找不到

```bash
# 檢查 cuDNN 是否安裝
dpkg -l | grep cudnn

# 如果沒有，安裝：
sudo apt-get update
sudo apt-get install -y libcudnn8 libcudnn8-dev
```

### Q3: CUDA 版本不相容

您的驅動支援 CUDA 13.0，但可以使用較舊的 CUDA Toolkit（向下相容）：

```bash
# 檢查支援的 CUDA 版本
nvidia-smi | grep "CUDA Version"

# TensorFlow 2.15 支援 CUDA 12.x
# 可以安全使用 CUDA 12.3
```

### Q4: 訓練時 GPU 記憶體不足

```bash
# 減少批次大小
python src/cli/train.py --batch-size 16  # 而非 32

# 或在 Python 中設定記憶體動態增長（已內建於 gpu_checker.py）
```

## 效能對比

| 操作 | CPU (估計) | GPU (估計) | 加速比 |
|------|------------|------------|--------|
| 訓練 100 epochs | 8-12 小時 | 1-2 小時 | 5-10x |
| 超參數調整 50 trials | 2-3 天 | 6-12 小時 | 4-6x |
| 預測單次 | 5-10 秒 | 1-3 秒 | 2-3x |

## 建議方案

**如果您是**：
- 🚀 **追求速度**: 選擇**選項 1**（安裝 CUDA）
- 🎯 **快速測試**: 選擇**選項 2**（使用 CPU）
- 🔧 **遇到問題**: 先用**選項 2**測試功能，再回來處理**選項 1**

## 參考資源

- [TensorFlow GPU 支援](https://www.tensorflow.org/install/gpu)
- [NVIDIA CUDA on WSL](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)
- [WSL2 GPU 加速](https://learn.microsoft.com/en-us/windows/wsl/tutorials/gpu-compute)
