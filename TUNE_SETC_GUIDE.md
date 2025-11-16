# Set C 專用超參數調整指南

## 概述

`tune_setC_only.py` 是專門針對 **Set C（全特徵集）** 進行深度超參數優化的腳本。

### 與標準 tune.py 的差異

| 特性 | tune.py | tune_setC_only.py |
|------|---------|-------------------|
| 特徵集搜尋 | Set A, B, C 都搜尋 | **固定使用 Set C** |
| 預設試驗次數 | 50 | **100** |
| 預設訓練週期 | 50 | **100** |
| Dropout 範圍 | 0.1 - 0.4 (步長 0.1) | **0.1 - 0.5 (步長 0.05)** |
| 參考模型 | 基準模型 | **最佳調整模型** |
| 使用場景 | 初次調整 | **深度優化** |

### 為什麼使用這個腳本？

1. ✅ **專注優化**：不浪費時間在 Set A/B 上
2. ✅ **更細緻搜尋**：Dropout 步長更小（0.05 vs 0.1）
3. ✅ **更多探索**：100 次試驗 vs 50 次
4. ✅ **更充分訓練**：100 epochs vs 50 epochs
5. ✅ **以最佳為基準**：與現有最好的模型比較

---

## 快速開始

### 方法 1：直接執行（預設參數）

```bash
# 確認在專案目錄且虛擬環境已啟動
cd /mnt/d/000-github-repositories/stock-price-prediction-v04
source venv/bin/activate

# 直接執行（使用預設 100 trials × 100 epochs）
python tune_setC_only.py
```

### 方法 2：快速測試（建議先執行）

```bash
# 編輯檔案
nano tune_setC_only.py

# 找到並修改這兩行：
#   MAX_TRIALS = 5        # 原本 100
#   EPOCHS_PER_TRIAL = 10 # 原本 100

# 儲存：Ctrl+X, Y, Enter

# 執行測試
python tune_setC_only.py
```

**測試預計時間**：
- GPU: 5-10 分鐘
- CPU: 20-30 分鐘

### 方法 3：背景執行（推薦用於完整訓練）

```bash
# 確保使用完整參數（MAX_TRIALS=100, EPOCHS_PER_TRIAL=100）

# 背景執行並記錄 log
nohup python tune_setC_only.py > tune_setC.log 2>&1 &

# 記下 process ID
echo $!

# 即時查看進度
tail -f tune_setC.log

# 按 Ctrl+C 停止查看（不會停止訓練）
```

---

## 參數設定

### 可調整參數（在 main() 函數中）

```python
def main():
    # ========== 可調整參數 ==========

    # 資料檔案
    DATA_FILE = "19980601-20251111-converted.csv"

    # 參考模型（用於比較）
    BASELINE_MODEL = "models/best_tuned_model.h5"

    # 試驗次數（建議：測試=5, 完整=100, 深度=200）
    MAX_TRIALS = 100

    # 每次試驗的訓練週期（建議：測試=10, 完整=100, 深度=150）
    EPOCHS_PER_TRIAL = 100

    # Keras Tuner 專案名稱（不同名稱會建立不同的試驗資料夾）
    PROJECT_NAME = "lstm_stock_tuning_setC_only"

    # 調整後模型的名稱前綴
    TUNED_MODEL_NAME = "best_tuned_setC_optimized"
```

### 建議配置組合

#### 配置 1: 快速測試
```python
MAX_TRIALS = 5
EPOCHS_PER_TRIAL = 10
```
- **時間**: GPU ~5-10 分鐘, CPU ~20-30 分鐘
- **用途**: 確認腳本可正常執行

#### 配置 2: 標準優化
```python
MAX_TRIALS = 100
EPOCHS_PER_TRIAL = 100
```
- **時間**: GPU ~10-15 小時, CPU ~3-5 天
- **用途**: 完整的超參數搜尋

#### 配置 3: 深度優化
```python
MAX_TRIALS = 200
EPOCHS_PER_TRIAL = 150
```
- **時間**: GPU ~30-40 小時, CPU ~7-10 天
- **用途**: 追求極致效能

---

## 超參數搜尋空間

### 固定參數
- **特徵集**: Set C（15 個特徵）
- **批次大小**: 32

### 搜尋參數

| 超參數 | 搜尋範圍 | 說明 |
|--------|----------|------|
| **time_steps** | 20, 40, 60, 80 | 時間窗口大小（天數） |
| **num_layers** | 2, 3, 4 | LSTM 層數 |
| **units_per_layer** | 32, 64, 96, 128 | 每層的單元數（獨立選擇） |
| **dropout_rate** | 0.1 - 0.5 (步長 0.05) | Dropout 比率 |
| **learning_rate** | 0.001, 0.0005, 0.0001 | 學習率 |

### 搜尋空間大小

理論組合數：
- 4 (time_steps) × 3 (num_layers) × 4^4 (units) × 9 (dropout) × 3 (lr)
- = **27,648 種組合**

實際探索：100 次試驗（約 0.36% 的搜尋空間）

---

## 執行監控

### 查看即時進度

```bash
# 持續查看 log
tail -f tune_setC.log

# 查看最後 50 行
tail -50 tune_setC.log

# 搜尋特定資訊
grep "Trial" tune_setC.log | tail -20          # 查看試驗進度
grep "驗證準確度" tune_setC.log                # 查看準確度
grep "✅" tune_setC.log                        # 查看完成步驟
```

### 檢查執行狀態

```bash
# 檢查 Python 程序是否還在執行
ps aux | grep tune_setC_only

# 查看 GPU 使用率（如有 GPU）
nvidia-smi

# 查看 CPU 和記憶體使用
top
# 或
htop
```

### 中斷執行

```bash
# 找到 process ID
ps aux | grep tune_setC_only

# 優雅地停止（會儲存目前進度）
kill <PID>

# 強制停止（不建議）
kill -9 <PID>
```

---

## 輸出檔案

### 執行後會產生的檔案

```
models/
├── best_tuned_setC_optimized_YYYYMMDD_HHMMSS.h5          # 優化後的模型
└── best_tuned_setC_optimized_YYYYMMDD_HHMMSS_config.json # 模型配置

logs/tuning_logs/
├── tuning_results_setC_YYYYMMDD_HHMMSS.txt               # 詳細調整結果
├── comparison_report_setC_YYYYMMDD_HHMMSS.txt            # 與參考模型比較
└── lstm_stock_tuning_setC_only/                          # Keras Tuner 試驗資料
    ├── trial_00/
    ├── trial_01/
    └── ...
```

### 檔案說明

#### 1. 模型檔案 (.h5)
包含訓練好的神經網路權重

#### 2. 配置檔案 (_config.json)
```json
{
  "model_file": "models/best_tuned_setC_optimized_20251116_180000.h5",
  "feature_set_id": "Set C",
  "time_steps": 60,
  "created_at": "2025-11-16 18:00:00",
  "num_layers": 4,
  "dropout_rate": 0.35,
  "learning_rate": 0.0005,
  "batch_size": 32,
  "epochs_trained": 100,
  "baseline_model": "models/best_tuned_model.h5",
  "tuning_timestamp": "20251116_180000",
  "model_type": "tuned_setC_only"
}
```

#### 3. 調整結果 (tuning_results_setC_*.txt)
包含：
- 調整統計（總試驗次數、執行時間）
- 最佳模型效能（驗證損失、準確度）
- 最佳超參數配置
- 前 10 名試驗的詳細結果

#### 4. 比較報告 (comparison_report_setC_*.txt)
包含：
- 參考模型效能
- 優化模型效能
- 改善幅度（絕對值和百分比）
- 是否優於參考模型的判斷

---

## 查看結果

### 快速查看

```bash
# 查看最新的配置檔案
ls -lht models/best_tuned_setC_optimized_*.json | head -1

# 查看配置內容
cat models/best_tuned_setC_optimized_*_config.json | tail -1

# 查看調整結果摘要
head -50 logs/tuning_logs/tuning_results_setC_*.txt | tail -1

# 查看比較報告
cat logs/tuning_logs/comparison_report_setC_*.txt | tail -1
```

### 詳細分析

```bash
# 查看完整調整結果（包含前 10 名）
cat logs/tuning_logs/tuning_results_setC_20251116_180000.txt

# 查看所有試驗的驗證準確度
grep "驗證準確度" logs/tuning_logs/tuning_results_setC_20251116_180000.txt

# 比較所有 Set C 調整的結果
ls -lht logs/tuning_logs/comparison_report_setC_*.txt
```

---

## 漸進式調整（增量搜尋）

### 為什麼要漸進式調整？

1. ✅ **降低風險**：先用少量試驗確認方向正確
2. ✅ **節省時間**：不用一次執行很長時間
3. ✅ **彈性調整**：可以根據初步結果決定是否繼續
4. ✅ **不浪費資源**：已完成的試驗會被保留

### 使用方式

#### 第一階段：快速探索（50 trials）

```python
# 編輯 tune_setC_only.py
MAX_TRIALS = 50
EPOCHS_PER_TRIAL = 50
PROJECT_NAME = "lstm_stock_tuning_setC_only"
```

```bash
python tune_setC_only.py
```

**預計時間**：GPU ~5-7 小時，CPU ~1-2 天

#### 第二階段：標準搜尋（100 trials）

檢視第一階段結果後，如果滿意可繼續：

```python
# 只需修改這一行
MAX_TRIALS = 100  # 從 50 改成 100

# 其他保持不變
EPOCHS_PER_TRIAL = 50
PROJECT_NAME = "lstm_stock_tuning_setC_only"  # ⚠️ 必須相同
```

```bash
python tune_setC_only.py
```

**實際執行**：
```
⚠️ 偵測到先前的調整紀錄: logs/tuning_logs/lstm_stock_tuning_setC_only
⚠️ 已有 50 個試驗數據
⚠️ 將繼續調整至 100 個試驗
⚠️ 預計新增: 50 個試驗
```

**預計時間**：GPU ~5-7 小時（只訓練新的 50 次），CPU ~1-2 天

#### 第三階段：深度優化（200 trials）

如果想追求更好的效能：

```python
MAX_TRIALS = 200  # 從 100 改成 200
EPOCHS_PER_TRIAL = 100  # 增加訓練週期
PROJECT_NAME = "lstm_stock_tuning_setC_only"  # ⚠️ 必須相同
```

```bash
python tune_setC_only.py
```

**實際執行**：
```
⚠️ 偵測到先前的調整紀錄: logs/tuning_logs/lstm_stock_tuning_setC_only
⚠️ 已有 100 個試驗數據
⚠️ 將繼續調整至 200 個試驗
⚠️ 預計新增: 100 個試驗
```

**預計時間**：GPU ~20-25 小時，CPU ~4-6 天

### 重要注意事項

#### ✅ 必須保持相同
- `PROJECT_NAME`：必須完全一致
- 搜尋空間定義：不能修改 `build_model_setC_only()` 函數

#### ❌ 可以改變
- `MAX_TRIALS`：可以增加（不能減少）
- `EPOCHS_PER_TRIAL`：可以調整
- `BASELINE_MODEL`：可以更換參考模型

#### 如何完全重新開始

方法 1：修改專案名稱
```python
PROJECT_NAME = "lstm_stock_tuning_setC_only_v2"  # 新名稱
```

方法 2：刪除舊試驗
```bash
rm -rf logs/tuning_logs/lstm_stock_tuning_setC_only/
```

### 查看累積進度

```bash
# 查看試驗數量
ls logs/tuning_logs/lstm_stock_tuning_setC_only/ | grep "trial_" | wc -l

# 列出所有試驗
ls -d logs/tuning_logs/lstm_stock_tuning_setC_only/trial_*

# 查看最新試驗
ls -lht logs/tuning_logs/lstm_stock_tuning_setC_only/trial_*/trial.json | head -5
```

---

## 中斷調整並取得目前最佳模型

### 為什麼需要這個功能？

調整過程可能需要很長時間（幾小時到幾天），但你可能想：
1. ✅ 先看看目前的最佳結果
2. ✅ 中斷後稍後再繼續
3. ✅ 避免浪費已完成的試驗

好消息：**Keras Tuner 會即時儲存每個試驗的結果**，所以可以安全中斷！

### 步驟 1：安全中斷執行

#### 前台執行（可以看到輸出）

```bash
# 直接按 Ctrl+C
Ctrl + C
```

**會看到**：
```
KeyboardInterrupt caught. Stopping training...
Saving current trial...
```

#### 背景執行（使用 nohup）

```bash
# 查找進程
ps aux | grep tune_setC_only

# 範例輸出：
# root     12345  99.0  5.2  ... python tune_setC_only.py

# 優雅停止（完成當前 trial 後停止）
kill 12345

# 如果不響應，強制停止（可能丟失當前 trial）
kill -9 12345
```

### 步驟 2：取得目前最佳模型

使用專用腳本：

```bash
python get_best_from_tuner.py
```

#### 腳本會做什麼？

1. 🔍 載入 Keras Tuner 的試驗紀錄
2. 📊 分析所有已完成的試驗
3. 🏆 找出驗證損失最低的模型
4. 💾 儲存最佳模型和配置
5. 📋 顯示前 5 名試驗結果

#### 預期輸出範例

```
================================================================================
從 Keras Tuner 取得目前最佳模型
================================================================================
專案名稱: lstm_stock_tuning_setC_only
目錄: logs/tuning_logs

✅ 找到 23 個試驗數據

正在載入 Keras Tuner...
✅ Tuner 載入成功

正在分析試驗結果...
✅ 找到 23 個完整的試驗

================================================================================
前 5 名試驗結果
================================================================================

Rank 1 - Trial 12:
  驗證損失: 1.4532
  驗證準確度: 38.21%
  超參數:
    - feature_set_id: Set C
    - time_steps: 60
    - num_layers: 3
    - units_layer_1: 128
    - units_layer_2: 96
    - dropout_rate: 0.25
    - learning_rate: 0.0005
    - units_layer_3: 64

Rank 2 - Trial 08:
  驗證損失: 1.4621
  驗證準確度: 37.89%
  ...

================================================================================
摘要
================================================================================
已完成試驗數: 23
最佳試驗 ID: 12
最佳驗證損失: 1.4532
最佳驗證準確度: 38.21%

最佳超參數:
  - 特徵集: Set C
  - 時間窗口: 60
  - LSTM 層數: 3
  - 每層單元數: [128, 96, 64]
  - Dropout: 0.25
  - 學習率: 0.0005

輸出檔案:
  - 模型: models/best_tuned_setC_partial_20251116_183045.h5
  - 配置: models/best_tuned_setC_partial_20251116_183045_config.json

⚠️ 注意：這是從未完成的調整中提取的模型
   如果想要更好的結果，可以繼續執行 tune_setC_only.py
================================================================================
```

### 步驟 3：使用提取的模型

#### 查看配置

```bash
# 查看最新提取的模型配置
cat models/best_tuned_setC_partial_*_config.json | tail -1
```

#### 進行預測

```bash
python src/cli/predict.py \
    --model-file models/best_tuned_setC_partial_20251116_183045.h5 \
    --data-file 19980601-20251111-converted.csv \
    --input-date "2024-06-01"
```

#### 與參考模型比較

```bash
# 可以手動載入兩個模型進行比較
# 或使用配置檔案中的 best_val_accuracy 數值比較
```

### 步驟 4：稍後繼續調整

當你準備好繼續時，**直接執行相同的指令**：

```bash
python tune_setC_only.py
```

**Keras Tuner 會自動**：
1. ✅ 偵測已完成的 23 個試驗
2. ✅ 從 trial_23 開始繼續
3. ✅ 訓練到 MAX_TRIALS 設定的數量
4. ✅ 從全部試驗中選出最佳模型

**會看到**：
```
⚠️ 偵測到先前的調整紀錄: logs/tuning_logs/lstm_stock_tuning_setC_only
⚠️ 已有 23 個試驗數據
⚠️ 將繼續調整至 100 個試驗
⚠️ 預計新增: 77 個試驗
```

### 檔案說明

提取的模型檔案命名：
```
best_tuned_setC_partial_YYYYMMDD_HHMMSS.h5
best_tuned_setC_partial_YYYYMMDD_HHMMSS_config.json
```

**`partial` 表示**：
- 這是從**未完成的調整**中提取的
- 可能還有更好的超參數組合未被探索
- 建議稍後繼續調整以獲得更好結果

### 常見問題

#### Q: 提取的模型品質如何？
**A**: 取決於已完成的試驗數量：
- 10-20 trials: 可能找到不錯的配置
- 30-50 trials: 通常能找到較好的配置
- 50+ trials: 接近完整調整的效果

#### Q: 需要重新訓練提取的模型嗎？
**A**: 不需要。提取的模型已經是完整訓練過的，可以直接使用。

#### Q: 如果想完全重新開始怎麼辦？
**A**: 有兩種方法：

方法 1：刪除試驗資料
```bash
rm -rf logs/tuning_logs/lstm_stock_tuning_setC_only/
python tune_setC_only.py
```

方法 2：使用新的專案名稱
```python
# 編輯 tune_setC_only.py
PROJECT_NAME = "lstm_stock_tuning_setC_only_v2"
```

#### Q: 如何查看已完成的試驗數量？
**A**: 使用以下指令：
```bash
# 方法 1：計算試驗目錄數量
ls logs/tuning_logs/lstm_stock_tuning_setC_only/trial_* | wc -l

# 方法 2：執行提取腳本（不儲存模型，只查看）
python get_best_from_tuner.py | head -20
```

### 手動查看試驗結果（進階）

如果想自己分析試驗結果：

```bash
# 使用 Python 快速解析
python -c "
import json
from pathlib import Path

trials = []
for p in Path('logs/tuning_logs/lstm_stock_tuning_setC_only').glob('trial_*/trial.json'):
    with open(p) as f:
        data = json.load(f)
        if 'score' in data and data.get('status') == 'COMPLETED':
            trial_id = p.parent.name
            score = data['score']
            trials.append((trial_id, score))

trials.sort(key=lambda x: x[1])  # 按損失排序

print(f'已完成試驗數: {len(trials)}')
print()
print('前 10 名試驗（按驗證損失排序）:')
for i, (trial_id, score) in enumerate(trials[:10], 1):
    print(f'{i}. {trial_id}: {score:.4f}')
"
```

### 快速參考

```bash
# 完整流程
Ctrl + C                            # 1. 中斷訓練
python get_best_from_tuner.py       # 2. 取得最佳模型
cat models/best_tuned_setC_partial_*_config.json  # 3. 查看配置
python tune_setC_only.py            # 4. 稍後繼續（可選）
```

---

## 使用優化後的模型

### 進行預測

```bash
python src/cli/predict.py \
    --model-file models/best_tuned_setC_optimized_20251116_180000.h5 \
    --data-file 19980601-20251111-converted.csv \
    --input-date "2024-06-01"
```

### 繼續優化

如果想基於此結果繼續優化：

```bash
# 編輯 tune_setC_only.py
nano tune_setC_only.py

# 修改參考模型為新的優化模型
BASELINE_MODEL = "models/best_tuned_setC_optimized_20251116_180000.h5"

# 修改專案名稱避免覆蓋
PROJECT_NAME = "lstm_stock_tuning_setC_only_v2"

# 增加試驗次數和訓練週期
MAX_TRIALS = 150
EPOCHS_PER_TRIAL = 150

# 執行
nohup python tune_setC_only.py > tune_setC_v2.log 2>&1 &
```

---

## 常見問題

### Q1: 執行時出現記憶體不足錯誤
**A**: 減少批次大小或減少 LSTM 層數
```python
# 在 build_model_setC_only 函數中修改
num_layers = hp.Int("num_layers", min_value=2, max_value=3, step=1)  # 改成最多 3 層
```

### Q2: 訓練速度太慢
**A**:
1. 使用 GPU（速度提升 5-10 倍）
2. 減少試驗次數
3. 減少每次試驗的 epochs

### Q3: 如何繼續中斷的調整或增加試驗次數？
**A**: 腳本已自動支援繼續調整（`overwrite=False`）

**範例：從 100 trials 繼續到 200 trials**

第一次執行：
```python
MAX_TRIALS = 100  # 先執行 100 次
```
```bash
python tune_setC_only.py
```

第二次執行（繼續）：
```python
MAX_TRIALS = 200  # 改成 200 次
PROJECT_NAME = "lstm_stock_tuning_setC_only"  # ⚠️ 名稱必須相同
```
```bash
python tune_setC_only.py
```

**會發生什麼**：
```
⚠️ 偵測到先前的調整紀錄: logs/tuning_logs/lstm_stock_tuning_setC_only
⚠️ 已有 100 個試驗數據
⚠️ 將繼續調整至 200 個試驗
⚠️ 預計新增: 100 個試驗
```

- ✅ 保留前 100 次試驗結果
- ✅ 只訓練額外的 100 次（trial_100 到 trial_199）
- ✅ 從全部 200 次中選出最佳配置
- ✅ 節省時間（不重複訓練）

**注意事項**：
- `PROJECT_NAME` 必須相同
- 如果要完全重新開始，改變 `PROJECT_NAME` 或手動刪除 `logs/tuning_logs/lstm_stock_tuning_setC_only/`

### Q4: 想要固定時間窗口為 60
**A**: 修改搜尋空間
```python
# 在 build_model_setC_only 函數中
time_steps = hp.Fixed("time_steps", 60)  # 固定為 60
```

### Q5: 優化後模型反而比參考模型差？
**A**: 可能原因：
1. 試驗次數不夠（建議至少 100 次）
2. 每次試驗的 epochs 太少（建議至少 100 epochs）
3. 參考模型已經是接近最優解
4. 隨機性影響（可以多跑幾次）

**解決方案**：
```bash
# 增加試驗次數和訓練週期
MAX_TRIALS = 200
EPOCHS_PER_TRIAL = 150

# 使用不同的 tuner（Hyperband 可能更有效）
# 需要修改程式碼中的 kt.BayesianOptimization 為 kt.Hyperband
```

### Q6: 如何查看試驗的詳細進度？
**A**:
```bash
# 查看 Keras Tuner 的試驗資料夾
ls -lh logs/tuning_logs/lstm_stock_tuning_setC_only/

# 查看特定試驗的結果
cat logs/tuning_logs/lstm_stock_tuning_setC_only/trial_00/trial.json
```

---

## 效能優化建議

### 1. GPU 加速
確保使用 GPU 訓練：
```bash
# 檢查 GPU 是否可用
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# 查看 GPU 使用率
nvidia-smi
```

### 2. 平行調整（進階）
如果有多個 GPU，可以同時執行多個調整：
```bash
# Terminal 1
CUDA_VISIBLE_DEVICES=0 python tune_setC_only.py &

# Terminal 2（修改專案名稱避免衝突）
CUDA_VISIBLE_DEVICES=1 python tune_setC_only.py &
```

### 3. 調整搜尋策略
使用更有效的搜尋演算法：
```python
# Bayesian Optimization（預設，適合連續參數）
tuner = kt.BayesianOptimization(...)

# Hyperband（適合大量試驗）
tuner = kt.Hyperband(...)

# Random Search（最基本但有效）
tuner = kt.RandomSearch(...)
```

---

## 預期效能

基於 `best_tuned_model.h5` 的效能（測試準確度 26.14%），預期：

### 保守估計
- 測試準確度：**27-30%**
- 改善幅度：**+1-4%**

### 樂觀估計
- 測試準確度：**30-35%**
- 改善幅度：**+4-9%**

### 實際影響因素
1. 試驗次數（越多越好，但邊際效益遞減）
2. 訓練週期（太少會欠擬合，太多會過擬合）
3. 資料品質（限制了模型的理論上限）
4. 隨機種子（每次執行結果會略有不同）

---

## 最佳實踐

### 執行前
1. ✅ 確認虛擬環境已啟動
2. ✅ 確認 GPU 可用（如有）
3. ✅ 先執行快速測試（5 trials × 10 epochs）
4. ✅ 確認有足夠的磁碟空間（至少 5GB）

### 執行中
1. ✅ 使用 `nohup` 背景執行
2. ✅ 定期查看 log 確認正常運行
3. ✅ 監控系統資源使用率
4. ✅ 記錄開始時間以估算完成時間

### 執行後
1. ✅ 查看比較報告確認是否有改善
2. ✅ 檢查配置檔案確認參數正確
3. ✅ 備份最佳模型和配置
4. ✅ 使用新模型進行預測驗證

---

## 進階技巧

### 自訂搜尋空間

編輯 `build_model_setC_only()` 函數：

```python
# 增加更多單元數選項
units = hp.Choice(f"units_layer_{i+1}", [32, 64, 96, 128, 256])

# 增加更多學習率選項
learning_rate = hp.Choice("learning_rate", [0.001, 0.0007, 0.0005, 0.0003, 0.0001])

# 使用連續值而非離散值
dropout_rate = hp.Float("dropout_rate", min_value=0.1, max_value=0.5, step=0.01)
```

### 早停策略調整

```python
# 在 main() 函數中修改 callbacks
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=15,              # 增加耐心值（原本 10）
        min_delta=0.0001,         # 增加最小改善閾值
        restore_best_weights=True,
        verbose=0
    )
]
```

---

## 總結

這個腳本是為了在 Set C 的基礎上進行**深度優化**而設計的。相較於標準的 `tune.py`：

✅ **更專注**：只探索 Set C 的參數空間
✅ **更細緻**：Dropout 步長 0.05 vs 0.1
✅ **更充分**：100 trials × 100 epochs vs 50 × 50
✅ **更實用**：直接與最佳模型比較

**建議執行順序**：
1. 快速測試（5 trials）確認可運行
2. 完整執行（100 trials）尋找最佳配置
3. 如有需要，深度優化（200 trials）追求極致

祝訓練順利！🚀
