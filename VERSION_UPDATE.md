# 版本更新說明 - TensorFlow 2.16.1 升級

## 📋 更新摘要

本次更新將專案從 TensorFlow 2.15.0 升級到 2.16.1，並採用 Python 3.11，大幅簡化 GPU 設置流程。

---

## 🎯 主要變更

### 1. 套件版本更新

| 套件 | 舊版本 | 新版本 | 變更原因 |
|------|--------|--------|----------|
| **Python** | 3.9+ | **3.12** | Ubuntu 24.04 預設版本，無需額外安裝 |
| **TensorFlow** | `tensorflow>=2.12.0` | **`tensorflow[and-cuda]==2.16.1`** | 自動安裝 CUDA 支援 |
| **NumPy** | `>=1.23.0,<2.0.0` | **`==1.26.2`** | 精確版本，避免相容性問題 |
| **Pandas** | `>=1.5.0` | **`==2.1.4`** | 精確版本，避免相容性問題 |
| **scikit-learn** | `>=1.1.0` | **`==1.3.2`** | 精確版本，避免相容性問題 |
| **Keras Tuner** | `>=1.3.0` | **`==1.4.6`** | 精確版本，避免相容性問題 |

### 2. 重大簡化：GPU 設置

**舊方法（已廢棄）**：
```bash
# ❌ 需要手動安裝 CUDA Toolkit
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-3
sudo apt-get install -y libcudnn8 libcudnn8-dev

# ❌ 需要手動設定環境變數
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda
```

**新方法（自動化）**：
```bash
# ✅ 僅需安裝 Python 套件，CUDA 自動安裝
pip install tensorflow[and-cuda]==2.16.1
```

### 3. 安裝流程簡化

**舊流程（5 個步驟）**：
1. 安裝 Python
2. 建立虛擬環境
3. 安裝 pip 套件
4. **手動安裝 CUDA Toolkit**
5. **手動設定環境變數**

**新流程（2 個步驟，Ubuntu 24.04）**：
1. 建立虛擬環境（Python 3.12 已預裝）
2. 安裝 pip 套件（**CUDA 自動安裝**）

---

## 📝 已更新的文件

### 1. requirements.txt
- 更新為精確版本號
- `tensorflow[and-cuda]==2.16.1` 取代 `tensorflow>=2.12.0`
- 所有核心套件鎖定特定版本

### 2. README.md
- 更新環境需求（Python 3.12）
- 簡化 GPU 設置說明
- 強調 `tensorflow[and-cuda]` 自動安裝 CUDA
- 新增系統驗證腳本說明
- Ubuntu 24.04 使用預設 Python 3.12

### 3. WSL2_GPU_SETUP.md
- 標記舊方法為「已過時」
- 新增 TensorFlow 2.16.1 專用指南
- 簡化為僅需安裝 Windows 驅動
- 強調無需手動安裝 CUDA

### 4. QUICK_START.md
- 完全重寫為簡化版本
- 移除手動 CUDA 安裝步驟
- 新增 4 步驟快速安裝流程
- 更新 FAQ 反映新流程

### 5. VERSION_UPDATE.md（新建）
- 本文件，記錄所有變更

---

## 🚀 使用者需要做什麼？

### 如果您是新使用者

**按照新的 QUICK_START.md 操作即可**：
1. 安裝 Python 3.11
2. 建立虛擬環境
3. 安裝 requirements.txt
4. 開始使用

### 如果您已經安裝舊版本

**建議重新安裝以使用新版本**：

```bash
# 1. 備份現有模型（如果有）
cp -r models models_backup

# 2. 刪除舊虛擬環境
rm -rf venv

# 3. 確認 Python 3.12（Ubuntu 24.04 已內建）
python3 --version  # 應顯示 Python 3.12.x

# 4. 建立新虛擬環境
python3.12 -m venv venv
source venv/bin/activate

# 5. 升級 pip
pip install --upgrade pip

# 6. 安裝新版本套件
pip install -r requirements.txt

# 7. 驗證安裝
python validate_pipeline.py
```

---

## ✅ 優點

### 1. 簡化安裝
- **無需手動安裝 CUDA**：`tensorflow[and-cuda]` 自動處理
- **無需設定環境變數**：TensorFlow 自動配置
- **安裝時間減少**：從 20-30 分鐘縮短至 10-15 分鐘

### 2. 更好的相容性
- **Python 3.12**：Ubuntu 24.04 預設版本，無需安裝
- **精確版本號**：避免套件版本衝突
- **TensorFlow 2.16.1**：最新穩定版，效能更好

### 3. 更簡單的維護
- **統一環境**：所有開發者使用相同版本
- **減少錯誤**：版本衝突大幅減少
- **易於除錯**：已知版本組合，問題容易定位

---

## ⚠️ 注意事項

### 1. Python 版本
- **推薦使用 Python 3.12**（Ubuntu 24.04 預設版本）
- Ubuntu 24.04：無需安裝，已內建 Python 3.12
- Ubuntu 22.04：需手動安裝 Python 3.12

### 2. CUDA 自動安裝
- `tensorflow[and-cuda]` 會自動下載 **約 2-3 GB** 的 CUDA 套件
- 首次安裝時間較長（5-10 分鐘）
- 需要穩定的網路連線

### 3. Windows 驅動仍然需要
- **仍需**在 Windows 安裝 NVIDIA CUDA on WSL 驅動程式
- 下載：https://developer.nvidia.com/cuda/wsl
- 這是唯一需要手動安裝的 GPU 相關元件

### 4. 舊版本相容性
- 使用舊版本訓練的模型**應該**可以在新版本載入
- 但建議測試後再大規模使用
- 如有問題，保留舊環境備份

---

## 🧪 測試建議

安裝新版本後，建議執行以下測試：

```bash
# 1. 系統驗證
python validate_pipeline.py

# 2. 快速訓練測試
python src/cli/train.py \
    --data-file 19980601-20251111-converted.csv \
    --feature-set "Set A" \
    --epochs 5 \
    --batch-size 16

# 3. 單元測試
pytest tests/unit/ -v

# 4. GPU 測試（如果有 GPU）
python -c "import tensorflow as tf; print('GPU 裝置:', tf.config.list_physical_devices('GPU'))"
```

---

## 📚 參考資源

- **TensorFlow 2.16 Release Notes**: https://github.com/tensorflow/tensorflow/releases/tag/v2.16.1
- **tensorflow[and-cuda] 說明**: https://www.tensorflow.org/install/pip
- **Python 3.11 新功能**: https://docs.python.org/3/whatsnew/3.11.html
- **NumPy 1.26 Release Notes**: https://numpy.org/doc/stable/release/1.26.0-notes.html

---

## 💡 建議

**對於新專案**：
- ✅ 直接使用新版本配置
- ✅ 按照 QUICK_START.md 操作

**對於現有專案**：
- ⚠️ 評估升級的必要性
- ⚠️ 如果舊版本運作正常，可以暫時不升級
- ✅ 如果遇到相容性問題，建議升級

---

## 🔄 回退方案

如果新版本有問題，可以回退到舊版本：

```bash
# 保留舊虛擬環境作為備份
mv venv venv_old

# 重新建立使用舊版本配置的環境
# （需要自行準備舊版 requirements.txt）
```

---

**版本更新完成！** 🎉

如有任何問題，請參考 [QUICK_START.md](QUICK_START.md) 或 [README.md](README.md)。
