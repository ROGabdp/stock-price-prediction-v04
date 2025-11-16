# ✅ 已驗證的成功 GPU 環境配置

本文件記錄了**已驗證可成功啟動 GPU** 的完整環境配置資訊。

## 📋 完整環境清單

| 類別 | 套件名稱 | 精確版本 | 備註 |
|------|---------|---------|------|
| **作業系統** | Ubuntu 版本 | 24.04 (Noble Numbat) | WSL2 環境 |
| **Python 版本** | Python | 3.12.3 | 系統預設版本 |
| **系統 GPU 驅動** | NVIDIA 驅動版本 | 581.80 | Windows 主機提供 |
| **CUDA Runtime** | CUDA Version | 13.0 | Windows 主機提供 |
| **系統 CUDA Toolkit** | cuda (in WSL) | 13.0.2-1 | 確保 $LD_LIBRARY_PATH 正確設定 |

## 🐍 Python 套件版本

| 套件名稱 | 版本 | 重要性 |
|---------|------|--------|
| tensorflow[and-cuda] | 2.16.1 | ⭐ 解決 GPU 偵測問題的關鍵版本 |
| keras-tuner | 1.4.6 | ✅ 成功環境中的匹配版本 |
| pandas | 2.1.4 | ✅ 成功環境中的匹配版本 |
| numpy | 1.26.2 | ✅ 成功環境中的匹配版本 |
| scikit-learn | 1.3.2 | ✅ 成功環境中的匹配版本 |

## 🔑 成功關鍵要素

您成功啟動 GPU 的核心在於以下三點:

### 1. 正確的 TensorFlow 版本
```bash
# 在 Python 3.12 環境下,精確使用此版本
pip install tensorflow[and-cuda]==2.16.1
```

### 2. WSL2 內安裝完整 CUDA Toolkit
```bash
# 在 WSL2 Ubuntu 內執行
sudo apt install cuda
```

這會安裝 `cuda 13.0.2-1` 版本,提供必要的 CUDA 函式庫。

### 3. 設定 LD_LIBRARY_PATH 環境變數

在虛擬環境啟動腳本中設定正確的 `$LD_LIBRARY_PATH`:

```bash
# 編輯虛擬環境啟動腳本
nano venv/bin/activate

# 在檔案末尾加入(activate 函式之外):
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

或者在 `~/.bashrc` 中設定:

```bash
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

## ✅ 驗證步驟

完成設定後,執行以下驗證:

### 1. 檢查 NVIDIA 驅動
```bash
nvidia-smi
```

**預期輸出**:
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 581.80       Driver Version: 581.80       CUDA Version: 13.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        ...
```

### 2. 檢查 CUDA Toolkit
```bash
nvcc --version
```

**預期輸出**:
```
nvcc: NVIDIA (R) Cuda compiler driver
...
Cuda compilation tools, release 13.0, V13.0.xxx
```

### 3. 檢查 TensorFlow GPU
```bash
python -c "import tensorflow as tf; print('TensorFlow 版本:', tf.__version__)"
python -c "import tensorflow as tf; print('GPU 裝置:', tf.config.list_physical_devices('GPU'))"
```

**預期輸出**:
```
TensorFlow 版本: 2.16.1
GPU 裝置: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

### 4. 執行完整驗證腳本
```bash
python validate_pipeline.py
```

## 📝 環境設定檢查清單

- [ ] Ubuntu 24.04 (Noble Numbat) in WSL2
- [ ] Python 3.12.3 安裝並設定為預設
- [ ] Windows NVIDIA 驅動 581.80+ 已安裝
- [ ] WSL2 內可執行 `nvidia-smi` 並看到 GPU
- [ ] WSL2 內已安裝 `cuda` 套件 (13.0.2-1)
- [ ] `nvcc --version` 顯示 CUDA 13.0
- [ ] `$LD_LIBRARY_PATH` 包含 `/usr/local/cuda/lib64`
- [ ] Python 虛擬環境已建立 (`python3.12 -m venv venv`)
- [ ] 已安裝 `tensorflow[and-cuda]==2.16.1`
- [ ] TensorFlow 可偵測到 GPU

## 🚀 快速設定腳本

如果您需要在新環境中複製此配置,可使用以下腳本:

```bash
#!/bin/bash
# 適用於 Ubuntu 24.04 WSL2

# 1. 確認 Python 版本
python3 --version  # 應顯示 3.12.3

# 2. 安裝 CUDA Toolkit
sudo apt update
sudo apt install cuda -y

# 3. 設定環境變數
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# 4. 建立虛擬環境
python3.12 -m venv venv
source venv/bin/activate

# 5. 升級 pip 並安裝依賴
pip install --upgrade pip
pip install -r requirements.txt

# 6. 驗證 GPU
python -c "import tensorflow as tf; print('GPU:', tf.config.list_physical_devices('GPU'))"
```

## ⚠️ 常見陷阱與解決方案

### 問題 1: TensorFlow 找不到 libcudart.so
**原因**: `$LD_LIBRARY_PATH` 未設定或路徑不正確

**解決方案**:
```bash
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### 問題 2: GPU 裝置列表為空
**原因**: 可能是 CUDA Toolkit 未安裝或版本不相容

**解決方案**:
```bash
sudo apt install cuda
# 確認版本
nvcc --version
```

### 問題 3: nvidia-smi 在 WSL2 中無法執行
**原因**: Windows 端 NVIDIA 驅動未正確安裝

**解決方案**:
1. 下載並安裝: https://developer.nvidia.com/cuda/wsl
2. 重啟電腦
3. 重新啟動 WSL2

## 📚 相關文件

- [README.md](README.md) - 完整專案說明
- [WSL2_GPU_SETUP.md](WSL2_GPU_SETUP.md) - WSL2 GPU 詳細設定指南
- [requirements.txt](requirements.txt) - Python 依賴套件清單

## 🎯 總結

這份配置清單確保了未來在新電腦或新專案環境中,能夠快速複製成功的 GPU 設定。

**核心重點**:
1. **Ubuntu 24.04** + **Python 3.12.3**
2. **tensorflow[and-cuda]==2.16.1**
3. **WSL2 內安裝 CUDA Toolkit**
4. **正確設定 $LD_LIBRARY_PATH**

只要遵循這四個關鍵要素,就能穩定復現成功的 GPU 訓練環境。
