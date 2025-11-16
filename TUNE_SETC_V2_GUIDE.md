# Set C 超參數調整指南（優化版 v2）

## 📋 目錄

- [快速開始](#快速開始)
- [優化重點](#優化重點)
- [搜尋空間說明](#搜尋空間說明)
- [執行方式](#執行方式)
- [參數配置](#參數配置)
- [監控執行進度](#監控執行進度)
- [輸出檔案說明](#輸出檔案說明)
- [漸進式調整](#漸進式調整)
- [中斷與繼續](#中斷與繼續)
- [與 v1 版本比較](#與-v1-版本比較)
- [常見問題](#常見問題)

---

## 快速開始

### 方法 1：直接執行（推薦）

```bash
python tune_setC_v2.py
```

### 方法 2：測試執行（小規模驗證）

修改 `tune_setC_v2.py` 中的參數：

```python
MAX_TRIALS = 10        # 改為 10
EPOCHS_PER_TRIAL = 20  # 改為 20
```

然後執行：

```bash
python tune_setC_v2.py
```

### 方法 3：背景執行（長時間訓練）

**Linux/macOS:**

```bash
nohup python tune_setC_v2.py > tune_setC_v2.log 2>&1 &
```

**Windows PowerShell:**

```powershell
Start-Process python -ArgumentList "tune_setC_v2.py" -RedirectStandardOutput "tune_setC_v2.log" -RedirectStandardError "tune_setC_v2_error.log" -WindowStyle Hidden
```

---

## 優化重點

### v2 版本的四大優化策略

| 優化項目 | v1 設定 | v2 設定 | 優化理由 |
|---------|---------|---------|----------|
| **單元數綁定** | 每層獨立搜尋 | 所有層相同 | 減少搜尋空間，提高效率，避免過度複雜化 |
| **Dropout 下限** | 0.1 | 0.25 | 增強正規化，減少過擬合 |
| **學習率範圍** | 0.001 ~ 0.0001 | 0.001 ~ 0.00001 | 加入更小的學習率，尋找更優收斂點 |
| **Early Stopping** | patience=10 | patience=15 | 給模型更多時間克服 val_loss 波動 |

### 為什麼要綁定單元數？

**理論依據：**
1. **減少搜尋空間複雜度**
   - v1: 每層獨立 → 4^4 = 256 種組合（4層情況）
   - v2: 綁定 → 4 種組合
   - 搜尋空間縮小 64 倍！

2. **避免過度複雜化**
   - 大部分情況下，LSTM 層使用相同單元數效果更穩定
   - 逐層遞減的設計（如 128→64→32）在實際應用中優勢不明顯

3. **提高訓練效率**
   - 更快找到好的超參數組合
   - 100 次試驗可以更全面探索其他重要參數

### 為什麼提高 Dropout 下限？

**實驗發現：**
- 過低的 dropout（0.1-0.2）在股票預測任務中容易過擬合
- 0.25-0.5 的範圍能提供更好的正規化效果
- 特別適合 Set C（15 個特徵）這種高維特徵集

### 為什麼加入更小的學習率？

**收斂優化：**
- 較大學習率（0.001）：快速收斂，但可能錯過最優點
- 較小學習率（0.00005, 0.00001）：收斂慢，但能找到更精細的最優解
- 配合 patience=15，給小學習率足夠時間發揮作用

---

## 搜尋空間說明

### 超參數範圍

| 超參數 | 選項 | 數量 | 說明 |
|--------|------|------|------|
| `feature_set_id` | Set C（固定） | 1 | 15 個特徵（動量 + 震盪） |
| `time_steps` | 20, 40, 60, 80 | 4 | 時間窗口大小 |
| `num_layers` | 2, 3, 4 | 3 | LSTM 層數 |
| `units`（綁定） | 32, 64, 128, 256 | 4 | 所有層使用相同單元數 |
| `dropout_rate` | 0.25, 0.3, 0.35, 0.4, 0.45, 0.5 | 6 | Dropout 比率 |
| `learning_rate` | 0.001, 0.0005, 0.0001, 0.00005, 0.00001 | 5 | Adam 學習率 |

### 理論組合數

```
總組合數 = 4 × 3 × 4 × 6 × 5 = 1,440 種
```

**對比 v1 版本：**
- v1 理論組合數：27,648 種
- v2 理論組合數：1,440 種
- **縮小 19 倍！**

### 搜尋效率提升

以 100 次試驗為例：

| 版本 | 理論組合數 | 100 次試驗覆蓋率 | 優勢 |
|------|-----------|----------------|------|
| v1 | 27,648 | 0.36% | 搜尋空間過大，難以充分探索 |
| v2 | 1,440 | 6.94% | 覆蓋率提高 19 倍，更有機會找到好參數 |

---

## 執行方式

### 標準執行流程

```bash
# 1. 確認環境
python --version  # 需要 Python 3.8+

# 2. 確認依賴
pip list | grep tensorflow
pip list | grep keras-tuner

# 3. 執行調整
python tune_setC_v2.py
```

### 執行過程

腳本會依序執行以下步驟：

1. **步驟 1/7**: 檢查 GPU 可用性
2. **步驟 2/7**: 載入資料（`19980601-20251111-converted.csv`）
3. **步驟 3/7**: 特徵工程（固定 Set C）
4. **步驟 4/7**: 資料集分割（70% 訓練 / 15% 驗證 / 15% 測試）
5. **步驟 5/7**: 特徵縮放（Standard Scaler）
6. **步驟 6/7**: 建立 Keras Tuner
7. **步驟 7/7**: 執行超參數調整

---

## 參數配置

### 主要參數（在 `main()` 函數中）

```python
# 資料設定
DATA_FILE = "19980601-20251111-converted.csv"  # 資料檔案
BASELINE_MODEL = "models/best_tuned_model.h5"  # 參考模型

# 調整設定
MAX_TRIALS = 100           # 最大試驗次數
EPOCHS_PER_TRIAL = 100     # 每次試驗的訓練週期
PROJECT_NAME = "lstm_stock_tuning_setC_v2"
TUNED_MODEL_NAME = "best_tuned_setC_v2"
```

### 推薦配置組合

#### 快速測試（約 1-2 小時）

```python
MAX_TRIALS = 10
EPOCHS_PER_TRIAL = 20
```

適合：初次測試、驗證環境是否正常

#### 標準搜尋（約 10-15 小時，GPU）

```python
MAX_TRIALS = 100
EPOCHS_PER_TRIAL = 100
```

適合：標準超參數優化任務（**推薦**）

#### 深度搜尋（約 20-30 小時，GPU）

```python
MAX_TRIALS = 200
EPOCHS_PER_TRIAL = 150
```

適合：追求極致效能，有充足時間和資源

#### 極致搜尋（約 50+ 小時，GPU）

```python
MAX_TRIALS = 500
EPOCHS_PER_TRIAL = 200
```

適合：生產環境最終調優

### Early Stopping 配置

v2 版本固定使用以下配置：

```python
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=15,        # 提高至 15
        restore_best_weights=True,
        verbose=0
    )
]
```

**為什麼 patience=15？**
- v1 使用 patience=10，可能導致模型過早停止
- 較小的學習率需要更多 epoch 才能收斂
- patience=15 給予模型更多時間克服驗證損失的短期波動

---

## 監控執行進度

### 即時監控

#### 方法 1：直接查看終端輸出

如果在前景執行，會看到：

```
Trial 45 Complete [00h 08m 32s]
val_accuracy: 0.3245
val_loss: 1.5234

Best val_loss So Far: 1.4523
Total elapsed time: 02h 15m 48s
```

#### 方法 2：查看日誌檔案

```bash
# 查看最新日誌
tail -f logs/tuning_logs/tuning_setC_v2_*.log

# 或搜尋特定資訊
grep "Best val_loss" logs/tuning_logs/tuning_setC_v2_*.log
```

### 查看試驗數量

```bash
# Linux/macOS
ls -l logs/tuning_logs/lstm_stock_tuning_setC_v2/trial_* | wc -l

# Windows PowerShell
(Get-ChildItem "logs\tuning_logs\lstm_stock_tuning_setC_v2\trial_*").Count
```

### 估算剩餘時間

假設目前完成 30 個試驗，花費 3 小時：

```
平均每次試驗時間 = 3 小時 / 30 = 6 分鐘
剩餘試驗數 = 100 - 30 = 70
預估剩餘時間 = 70 × 6 分鐘 = 420 分鐘 ≈ 7 小時
```

---

## 輸出檔案說明

### 執行完成後會產生以下檔案：

#### 1. 模型檔案

```
models/best_tuned_setC_v2_20251116_180530.h5
```

- 格式：HDF5
- 包含：完整模型架構 + 訓練權重
- 用途：直接載入進行預測

#### 2. 模型配置檔案

```
models/best_tuned_setC_v2_20251116_180530_config.json
```

示例內容：

```json
{
  "model_path": "models/best_tuned_setC_v2_20251116_180530.h5",
  "feature_set_id": "Set C",
  "time_steps": 60,
  "num_layers": 3,
  "units": 128,
  "units_per_layer": [128, 128, 128],
  "dropout_rate": 0.35,
  "learning_rate": 0.0001,
  "batch_size": 32,
  "early_stopping_patience": 15,
  "tuner_type": "bayesian",
  "max_trials": 100,
  "model_type": "tuned_setC_v2",
  "optimization_notes": [
    "綁定所有 LSTM 層使用相同單元數",
    "Dropout 下限提高至 0.25",
    "學習率範圍擴展至 0.00001",
    "Early Stopping patience 提高至 15"
  ]
}
```

#### 3. 調整結果報告

```
logs/tuning_logs/tuning_results_setC_v2_20251116_180530.txt
```

包含：
- 總試驗次數和執行時間
- 最佳驗證損失和準確度
- 前 10 名試驗的詳細超參數
- 搜尋空間統計

#### 4. 比較報告

```
logs/tuning_logs/comparison_report_setC_v2_20251116_180530.txt
```

包含：
- 與參考模型（baseline）的效能比較
- 測試集損失和準確度
- 改善幅度（絕對值和百分比）

#### 5. Tuner 紀錄目錄

```
logs/tuning_logs/lstm_stock_tuning_setC_v2/
├── trial_001/
├── trial_002/
├── ...
└── oracle.json
```

- 每個 trial_xxx 包含該次試驗的詳細資訊
- oracle.json 記錄貝葉斯優化的狀態

---

## 漸進式調整

### 如何從 50 次試驗增加到 100 次？

#### 步驟 1：修改 MAX_TRIALS

編輯 `tune_setC_v2.py`：

```python
MAX_TRIALS = 100  # 從 50 改為 100
```

#### 步驟 2：直接執行

```bash
python tune_setC_v2.py
```

#### 步驟 3：自動檢測與繼續

腳本會自動偵測已有的試驗：

```
⚠️ 偵測到先前的調整紀錄: logs/tuning_logs/lstm_stock_tuning_setC_v2
⚠️ 已有 50 個試驗數據
⚠️ 將繼續調整至 100 個試驗
⚠️ 預計新增: 50 個試驗
```

然後從 trial_051 開始執行，不會重複計算 trial_001 ~ trial_050。

### 漸進式調整範例

#### 方案 1：逐步增加（保守）

```
第一輪: MAX_TRIALS = 50   (約 5-7 小時)
第二輪: MAX_TRIALS = 100  (約 5-7 小時)
第三輪: MAX_TRIALS = 200  (約 10-15 小時)
```

**優點：**
- 可以隨時停止，已有初步結果
- 觀察效能提升趨勢，決定是否繼續
- 分散計算負擔

#### 方案 2：一次到位（激進）

```
直接: MAX_TRIALS = 200  (約 20-30 小時)
```

**優點：**
- 一次完成，無需反覆修改
- 適合有充足時間和 GPU 資源

### 如何判斷是否需要增加試驗次數？

查看調整結果報告，觀察：

1. **最佳驗證損失是否持續下降？**
   - 如果最後 10-20 次試驗仍在改善 → 建議增加
   - 如果已經平穩不變 → 可以停止

2. **覆蓋率是否足夠？**
   - 100 次試驗覆蓋 6.94% 的搜尋空間
   - 200 次試驗覆蓋 13.89%
   - 建議至少達到 10% 以上

3. **前 10 名試驗的參數是否趨同？**
   - 如果前 10 名使用相似的超參數 → 可能已找到最優區域
   - 如果參數分散 → 建議增加試驗次數

---

## 中斷與繼續

### 中斷正在執行的調整

#### 方法 1：Ctrl+C（推薦）

如果在前景執行，按 `Ctrl+C` 即可安全中斷。

#### 方法 2：Kill Process

**找到 Process ID:**

```bash
# Linux/macOS
ps aux | grep tune_setC_v2.py

# Windows
tasklist | findstr python
```

**終止 Process:**

```bash
# Linux/macOS
kill <PID>

# Windows
taskkill /PID <PID> /F
```

### 取得目前最佳模型

中斷後，使用專用腳本：

```bash
python get_best_from_tuner_v2.py
```

#### 執行結果示例

```
================================================================================
從 Keras Tuner 取得目前最佳模型（v2）
================================================================================
專案名稱: lstm_stock_tuning_setC_v2
目錄: logs/tuning_logs

✅ 找到 35 個試驗數據

正在載入 Keras Tuner...
✅ Tuner 載入成功

正在分析試驗結果...
✅ 找到 35 個完整的試驗

================================================================================
前 5 名試驗結果
================================================================================

Rank 1 - Trial 0028:
  驗證損失: 1.4523
  驗證準確度: 34.25%
  超參數:
    - feature_set_id: Set C
    - time_steps: 60
    - num_layers: 3
    - units: 128 (綁定，所有 3 層相同)
    - dropout_rate: 0.35
    - learning_rate: 0.0001

...

正在建立最佳模型...
✅ 模型建立成功

正在儲存模型至: models/best_tuned_setC_v2_partial_20251116_145530.h5
✅ 模型已儲存

✅ 配置已儲存: models/best_tuned_setC_v2_partial_20251116_145530_config.json

================================================================================
摘要
================================================================================
已完成試驗數: 35
最佳試驗 ID: 0028
最佳驗證損失: 1.4523
最佳驗證準確度: 34.25%

最佳超參數:
  - 特徵集: Set C
  - 時間窗口: 60
  - LSTM 層數: 3
  - 綁定單元數: 128 (所有層相同)
  - Dropout: 0.35
  - 學習率: 0.0001

優化策略:
  1. 綁定所有 LSTM 層使用相同單元數
  2. Dropout 下限提高至 0.25
  3. 學習率範圍擴展至 0.00001
  4. Early Stopping patience 提高至 15

輸出檔案:
  - 模型: models/best_tuned_setC_v2_partial_20251116_145530.h5
  - 配置: models/best_tuned_setC_v2_partial_20251116_145530_config.json

⚠️ 注意：這是從未完成的調整中提取的模型
   如果想要更好的結果，可以繼續執行 tune_setC_v2.py
================================================================================
```

### 繼續未完成的調整

直接重新執行 `tune_setC_v2.py`，腳本會自動繼續：

```bash
python tune_setC_v2.py
```

輸出：

```
⚠️ 偵測到先前的調整紀錄: logs/tuning_logs/lstm_stock_tuning_setC_v2
⚠️ 已有 35 個試驗數據
⚠️ 將繼續調整至 100 個試驗
⚠️ 預計新增: 65 個試驗
```

**重要：**
- Keras Tuner 會自動從 trial_036 開始
- 不會重複計算已完成的試驗
- 舊的試驗結果會保留並參與最終比較

---

## 與 v1 版本比較

### 主要差異總覽

| 項目 | v1 版本 | v2 版本 | 改進 |
|------|---------|---------|------|
| **單元數設定** | 每層獨立 [32,64,96,128] | 綁定 [32,64,128,256] | 搜尋空間縮小 19 倍 |
| **Dropout 範圍** | 0.1 - 0.5 (step 0.05) | 0.25, 0.3, 0.35, 0.4, 0.45, 0.5 | 提高下限，增強正規化 |
| **學習率選項** | 3 個 | 5 個 | 加入 0.00005, 0.00001 |
| **Early Stop** | patience=10 | patience=15 | 提高 50% |
| **理論組合數** | 27,648 | 1,440 | 縮小 19 倍 |
| **100 次覆蓋率** | 0.36% | 6.94% | 提高 19 倍 |

### 應該選擇哪個版本？

#### 使用 v1 的情況：

- 想要探索每層不同單元數的組合
- 有充足的計算資源和時間
- 追求極致的搜尋全面性

#### 使用 v2 的情況（**推薦**）：

- 想要更高效的超參數搜尋
- 計算資源有限（單 GPU、有限時間）
- 想要在合理時間內獲得好結果
- 根據理論和經驗，綁定單元數效果不會差

### 可以同時執行嗎？

**可以！** 兩個版本使用不同的 `PROJECT_NAME`：

- v1: `lstm_stock_tuning_setC_only`
- v2: `lstm_stock_tuning_setC_v2`

因此可以平行執行，互不干擾：

```bash
# 終端 1
python tune_setC_only.py

# 終端 2
python tune_setC_v2.py
```

**注意：**
- 同時執行會佔用更多 GPU 記憶體和 CPU
- 建議確認硬體資源充足

---

## 常見問題

### Q1: 為什麼 v2 版本只支援 time_steps=60？

**A:** 這是簡化實作的權衡：

- 完整支援動態 `time_steps` 需要在搜尋過程中動態產生序列，實作複雜
- 固定 `time_steps=60` 可以簡化資料準備流程
- 如果想要搜尋不同的 `time_steps`，可以：
  1. 修改 line 207 的 `create_sequences` 參數
  2. 或執行多次，分別測試 20, 40, 60, 80

### Q2: 綁定單元數會不會限制模型表現？

**A:** 理論上可能，但實際影響很小：

- 大多數成功的 LSTM 模型都使用相同單元數
- 逐層遞減（如 128→64→32）在股票預測任務中優勢不明顯
- 節省的搜尋時間可以用於更多試驗，找到更好的其他參數組合
- 如果真的需要，可以隨時切換回 v1 版本

### Q3: patience=15 會不會太大，浪費時間？

**A:** 不會，理由如下：

- Early Stopping 只在連續 15 個 epoch 沒有改善時才停止
- 如果模型快速收斂，仍然會在幾十個 epoch 內完成
- 對於小學習率（0.00005），需要更多 epoch 才能充分訓練
- 實際上，patience=15 可以避免過早停止導致的潛在遺漏

### Q4: 為什麼沒有加入 256 以上的單元數？

**A:** 基於以下考量：

- 更大的單元數（512, 1024）會顯著增加訓練時間
- Set C 只有 15 個特徵，256 單元已經相當充足
- 避免過度複雜的模型導致過擬合
- 如果需要，可以手動修改 line 64 加入更大的值

### Q5: v2 版本可以用於 Set A 或 Set B 嗎？

**A:** 可以，但需要修改：

```python
# 在 build_model_setC_v2 函數中
feature_set_id = "Set A"  # 或 "Set B"
hp.Fixed("feature_set_id", "Set A")
```

同時建議修改 `PROJECT_NAME` 避免混淆：

```python
PROJECT_NAME = "lstm_stock_tuning_setA_v2"
```

### Q6: 如何判斷 v2 版本是否比 v1 更好？

**A:** 執行完成後，比較兩個版本的：

1. **測試準確度**（最重要）
2. **訓練時間** / 準確度（效率）
3. **前 10 名試驗的穩定性**

建議：
- 如果 v2 的測試準確度 ≥ v1 - 1%，優先選擇 v2（效率高）
- 如果 v2 明顯較差（> 2%），考慮使用 v1 或增加 v2 試驗次數

### Q7: 為什麼使用 BayesianOptimization 而不是 RandomSearch？

**A:** 貝葉斯優化的優勢：

- **智能搜尋**：根據先前試驗結果，預測有希望的超參數區域
- **更高效**：通常 50-100 次試驗就能找到好結果
- **適合複雜空間**：即使 v2 有 1,440 種組合，仍能有效探索

RandomSearch 的問題：
- 純隨機，可能浪費很多次試驗在無效區域
- 需要更多試驗才能覆蓋重要區域

### Q8: 可以在 CPU 上執行嗎？

**A:** 可以，但不建議：

- CPU 訓練速度約為 GPU 的 **10-50 倍慢**
- 100 次試驗在 GPU 上需 15 小時，CPU 上可能需 **150-750 小時**（6-31 天）

建議：
- 如果只有 CPU，使用 `MAX_TRIALS=10, EPOCHS_PER_TRIAL=20` 測試
- 或考慮使用雲端 GPU（Google Colab, Kaggle, AWS, Azure）

### Q9: 如何加速調整過程？

**策略：**

1. **減少試驗次數**
   ```python
   MAX_TRIALS = 50  # 從 100 降為 50
   ```

2. **減少每次試驗的 epoch**
   ```python
   EPOCHS_PER_TRIAL = 50  # 從 100 降為 50
   ```

3. **使用更強的 GPU**
   - RTX 3090 / 4090
   - A100 / V100（雲端）

4. **減少搜尋空間**
   ```python
   # 例如只搜尋 2-3 層
   num_layers = hp.Int("num_layers", min_value=2, max_value=3, step=1)
   ```

5. **使用更小的資料集**
   ```python
   # 在資料載入後
   df = df.tail(10000)  # 只使用最近 10000 筆
   ```

### Q10: 調整結果如何用於預測？

使用最佳模型進行預測：

```bash
python predict.py \
    --data 19980601-20251111-converted.csv \
    --model models/best_tuned_setC_v2_20251116_180530.h5 \
    --config models/best_tuned_setC_v2_20251116_180530_config.json
```

---

## 總結

### v2 版本的核心價值

1. **更高效的搜尋**：19 倍的搜尋空間縮減
2. **更好的正規化**：提高 dropout 下限至 0.25
3. **更精細的收斂**：加入極小學習率
4. **更穩定的訓練**：patience=15 避免過早停止

### 推薦使用流程

```
1. 執行 v2 版本 (100 次試驗)
   ↓
2. 查看結果，評估效能
   ↓
3. 如果滿意 → 直接使用
   ↓
4. 如果不滿意 → 增加至 200 次試驗（漸進式）
   ↓
5. 或嘗試 v1 版本（更全面搜尋）
```

### 下一步

- 執行完成後，使用 `predict.py` 進行預測
- 比較 v1 和 v2 的效能差異
- 根據結果決定是否繼續調整或部署模型

---

**祝調整順利！** 🚀
