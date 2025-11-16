# WSL2 + GPU 快速驗證指南

基於成功配置的 WSL2 Ubuntu 24.04 + NVIDIA GPU 環境

## 🎯 一鍵驗證腳本

```bash
#!/bin/bash
# WSL2 GPU 環境快速驗證

echo "=== 1. 檢查作業系統版本 ==="
lsb_release -a

echo -e "\n=== 2. 檢查 Python 版本 ==="
python3 --version

echo -e "\n=== 3. 檢查 NVIDIA 驅動 ==="
nvidia-smi

echo -e "\n=== 4. 檢查 CUDA Toolkit ==="
nvcc --version

echo -e "\n=== 5. 檢查環境變數 ==="
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"

echo -e "\n=== 6. 檢查虛擬環境中的 TensorFlow ==="
source venv/bin/activate
python -c "import tensorflow as tf; print('TensorFlow 版本:', tf.__version__)"
python -c "import tensorflow as tf; print('GPU 裝置:', tf.config.list_physical_devices('GPU'))"
python -c "import tensorflow as tf; print('GPU 可用:', len(tf.config.list_physical_devices('GPU')) > 0)"

echo -e "\n=== 7. 檢查關鍵套件版本 ==="
pip list | grep -E "tensorflow|keras-tuner|pandas|numpy|scikit-learn"
```

## ✅ 預期輸出檢查表

### 1. 作業系統
```
No LSB modules are available.
Distributor ID: Ubuntu
Description:    Ubuntu 24.04 LTS
Release:        24.04
Codename:       noble
```
✅ 應顯示 **Ubuntu 24.04** (Noble)

### 2. Python 版本
```
Python 3.12.3
```
✅ 應顯示 **3.12.3**

### 3. NVIDIA 驅動
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 581.80       Driver Version: 581.80       CUDA Version: 13.0     |
```
✅ 應顯示 GPU 資訊,CUDA Version **13.0+**

### 4. CUDA Toolkit
```
Cuda compilation tools, release 13.0, V13.0.xxx
```
✅ 應顯示 CUDA **13.0** 或以上

### 5. 環境變數
```
LD_LIBRARY_PATH: /usr/local/cuda/lib64:...
```
✅ 應包含 `/usr/local/cuda/lib64`

### 6. TensorFlow GPU
```
TensorFlow 版本: 2.16.1
GPU 裝置: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
GPU 可用: True
```
✅ TensorFlow **2.16.1**,GPU 裝置**非空列表**,GPU 可用 **True**

### 7. 關鍵套件版本
```
tensorflow             2.16.1
tensorflow-io-gcs-filesystem 0.x.x
keras-tuner            1.4.6
pandas                 2.1.4
numpy                  1.26.2
scikit-learn           1.3.2
```
✅ 版本應與上述列表完全匹配

## ❌ 常見問題快速診斷

### 問題 A: `nvidia-smi` 找不到指令
**診斷**: Windows NVIDIA 驅動未安裝或 WSL2 未正確配置

**解決步驟**:
1. 在 **Windows** 下載並安裝 NVIDIA CUDA on WSL 驅動
   - https://developer.nvidia.com/cuda/wsl
2. 重啟電腦
3. 重新執行 `nvidia-smi`

---

### 問題 B: `nvcc` 找不到指令
**診斷**: WSL2 內未安裝 CUDA Toolkit

**解決步驟**:
```bash
sudo apt update
sudo apt install cuda -y
```

---

### 問題 C: GPU 裝置列表為空 `[]`
**診斷**: TensorFlow 無法找到 CUDA 函式庫

**解決步驟**:
```bash
# 1. 設定環境變數
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# 2. 將設定加入 ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# 3. 重新啟動虛擬環境
deactivate
source venv/bin/activate

# 4. 重新驗證
python -c "import tensorflow as tf; print('GPU:', tf.config.list_physical_devices('GPU'))"
```

---

### 問題 D: TensorFlow 版本不是 2.16.1
**診斷**: 錯誤的 TensorFlow 版本

**解決步驟**:
```bash
pip uninstall tensorflow -y
pip install tensorflow[and-cuda]==2.16.1
```

---

### 問題 E: Python 版本不是 3.12.3
**診斷**: 使用了錯誤的 Python 版本

**解決步驟**:
```bash
# Ubuntu 24.04 預設就是 3.12.3
# 確認使用正確的 Python
which python3
python3 --version

# 重新建立虛擬環境
rm -rf venv
python3.12 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 🚀 完整環境重建步驟

如果驗證失敗,可使用以下步驟完整重建環境:

```bash
# 1. 確認 Windows NVIDIA 驅動已安裝
# 在 Windows PowerShell 執行: nvidia-smi

# 2. 在 WSL2 安裝 CUDA Toolkit
sudo apt update
sudo apt install cuda -y

# 3. 設定環境變數
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# 4. 切換到專案目錄
cd /mnt/d/000-github-repositories/stock-price-prediction-v04

# 5. 刪除舊虛擬環境並重建
rm -rf venv
python3.12 -m venv venv
source venv/bin/activate

# 6. 安裝依賴
pip install --upgrade pip
pip install -r requirements.txt

# 7. 驗證 GPU
python -c "import tensorflow as tf; print('GPU:', tf.config.list_physical_devices('GPU'))"
```

## 📊 效能基準測試

驗證成功後,可執行簡單的效能測試:

```python
import tensorflow as tf
import time

# 建立測試資料
with tf.device('/GPU:0'):
    a = tf.random.normal([10000, 10000])
    b = tf.random.normal([10000, 10000])

    # 熱身
    c = tf.matmul(a, b)

    # 計時
    start = time.time()
    for _ in range(10):
        c = tf.matmul(a, b)
    elapsed = time.time() - start

    print(f'GPU 平均運算時間: {elapsed/10:.4f} 秒')
```

**預期結果**: GPU 運算應在 **0.01-0.1 秒** 範圍內完成

## 🎓 成功案例參考

此配置已在以下環境驗證成功:

- **作業系統**: WSL2 Ubuntu 24.04 (Noble Numbat)
- **Python**: 3.12.3
- **NVIDIA 驅動**: 581.80
- **CUDA Runtime**: 13.0
- **CUDA Toolkit**: 13.0.2-1
- **TensorFlow**: 2.16.1
- **GPU**: NVIDIA RTX 系列 (或相容 GPU)

## 📝 驗證完成後續步驟

一旦所有驗證項目都通過:

1. ✅ 執行完整系統驗證
   ```bash
   python validate_pipeline.py
   ```

2. ✅ 測試訓練功能
   ```bash
   python src/cli/train.py \
       --data-file 19980601-20251111-converted.csv \
       --feature-set "Set A" \
       --time-steps 60 \
       --epochs 10 \
       --batch-size 32
   ```

3. ✅ 監控 GPU 使用率
   ```bash
   # 在另一個終端視窗執行
   watch -n 1 nvidia-smi
   ```

## 🔗 相關資源

- [VERIFIED_ENVIRONMENT.md](VERIFIED_ENVIRONMENT.md) - 完整環境配置詳情
- [WSL2_GPU_SETUP.md](WSL2_GPU_SETUP.md) - 詳細設定指南
- [README.md](README.md) - 專案使用說明
