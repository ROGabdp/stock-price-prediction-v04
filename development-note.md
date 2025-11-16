/speckit.constitution 

所有文件、註解及使用者介面文字，一律使用正體中文。
在程式碼風格上，所有 Python 程式碼必須保持 Pythonic 並嚴格遵循 PEP 8 規範。
Git 預設分支名稱設定為 main。
本專案將堅守「規格驅動開發」的核心原則，專注於交付一個高品質、可測試 (Testable) 的 Minimum Viable Product (MVP)，並嚴格禁止任何過度設計 (Overdesign)，確保實作方案的簡潔與務實。
用GPU進行機器學習的訓練。運用WSL2 + NVIDIA GPU。
已將wsl2 Utunbu_D 安裝在D槽，WSL 發行版名稱是 Ubuntu_D，路徑為 D:\wsl\Ubuntu_D。

/speckit.specify

以下提供資訊為MD格式。

此模型旨在預測台股未來 $20$ 個交易日的漲跌幅區間及其信心度，屬於多類別時間序列分類任務。歷史資料來源為根目錄的19980601-20251111-converted.csv。

1. 數據預處理與目標變數定義 (Label & Scaling)目標變數 (Y)：計算 $20$ 日報酬率 $R_{t \to t+20} = (\text{Close}_{t+20} - \text{Close}_t) / \text{Close}_t$。將 $R_{t \to t+20}$ 轉換為 $5$ 個類別的 $\text{One-Hot}$ 編碼向量（例如 $[0,0,0,0,1]$）。類別閾值 (以 $\sigma_{20} = 5\%$ 為例)：0 (極度下跌)： $R_{t \to t+20} < -5.0\%$1 (溫和下跌)： $-5.0\% \le R_{t \to t+20} < -2.5\%$2 (區間震盪)： $-2.5\% \le R_{t \to t+20} \le +2.5\%$3 (溫和上漲)： $+2.5\% < R_{t \to t+20} \le +5.0\%$4 (極度上漲)： $R_{t \to t+20} > +5.0\%$數據縮放 (Scaling)： 對所有輸入特徵使用 $\text{MinMaxScaler}$ 或 $\text{StandardScaler}$ 進行標準化/正規化。2. 輸入特徵與時間窗口 (Input Features & Time Steps)輸入時間窗口 ($T_{in}$)： 採用 $T_{in} = \mathbf{60}$ 天（回顧過去約三個月的數據）。特徵工程 (Feature Engineering)：將價格類特徵 (如 open, close) 轉換為日報酬率。納入所有技術指標 (DIF12-26, MACD9, OSC, K(9,3), D(9,3))。納入法人籌碼 (net buy sell, cumulative net buy sell)，考慮使用籌碼的變化率。移動平均線，採用 長短期 MA 的差距或斜率，例如 $\text{SMA20}/\text{SMA60}$ 的比值，或 $\text{SMA20}$ 的每日斜率。這些更能反映長期動能。3. LSTM 模型架構與訓練 (Architecture & Training)模型類型： 3 層 Stacked LSTMLSTM 層配置 (需設置 $\text{return\_sequences=True}$)：$\text{LSTM Layer 1}$: $\mathbf{128}$ 單元$\text{LSTM Layer 2}$: $\mathbf{64}$ 單元$\text{LSTM Layer 3}$: $\mathbf{32}$ 單元 ($\text{return\_sequences=False}$)正則化 (Regularization)： 在各 $\text{LSTM}$ 層之間加入 $\text{Dropout}(\mathbf{0.2})$ 層。輸出層： $\text{Dense}(\mathbf{5}, \text{activation='softmax'})$損失函數： $\text{Categorical Cross-Entropy}$優化器： $\text{Adam}$ (建議 $\text{Learning Rate}$ 初始值為 $\mathbf{10^{-3}}$)訓練控制： 採用 $\text{Early Stopping}$ 監控 $\text{Validation Loss}$ ($\text{Patience}=10$ 左右)。4. 結果輸出與信心度解讀預測輸出： 模型輸出 $5$ 維的機率向量。漲跌幅區間： 機率最高的維度即為預測的區間類別 (0 到 4)。信心度： 機率最高的數值即為模型對該區間的預測信心度。

先用前面的建議作為起點 (Baseline Model)。

接著，使用 $\text{Keras Tuner}$ 或 $\text{Optuna}$ 執行自動超參數調整。自動調整的目的是系統性地探索不同的參數組合，找出能最小化驗證損失（Validation Loss）的那一組設定。以下是你在進行自動調整時，應該作為變量納入探索的關鍵參數(Search Space)：輸入時間窗口 ($T_{in}$): $\{20, 40, 60, 80\}$$\text{LSTM}$ 層數: $\{2, 3, 4\}$每層 $\text{LSTM}$ 單元數 (Unit Size): $\{32, 64, 128\}$ (應允許每層單元數遞減，例如 $128 \rightarrow 64 \rightarrow 32$)$\text{Dropout}$ Rate: $[0.1, 0.4]$學習率 ($\text{Learning Rate}$): $\{10^{-3}, 5 \times 10^{-4}, 10^{-4}\}$批次大小 ($\text{Batch Size}$): $\{32, 64, 128\}$

「超參數探索範圍 (Search Space)」**的新增項目：新增指令：特徵集選擇 (Feature Set Selection)必須將特徵集的選擇納入超參數調整範圍。新增超參數： $\text{Feature\_Set\_ID}$探索範圍： $\{\text{Set A}, \text{Set B}, \text{Set C}\}$實作要求： 程式設計師需要準備三組經過數據處理後的特徵矩陣，並讓自動優化工具選擇使用哪一組特徵來訓練模型。最終目標： 最佳模型將是以下所有參數組合後的冠軍：($\text{Time Step}, \text{Layer Size}, \text{Learning Rate}, \textbf{Feature\_Set\_ID}$)。

Set A (動能型) 專注於中期趨勢與動能:僅包含價格變化率、Volume 變化率、$\text{SMA}$ 比例、$\text{MACD}$、法人籌碼。 (排除 KDJ, OSC 等震盪指標)

Set B (震盪型)  專注於超買超賣反轉訊號: 包含價格變化率、Volume 變化率、$\text{SMA}$ 比例、$\text{K(9,3)}$、$\text{D(9,3)}$、法人籌碼。 (排除 MACD 等趨勢指標)

Set C (全特徵集)  包含所有處理過的特徵: 所有可用的特徵。

模型檔案儲存： 在整個超參數調整過程中，只將在驗證集上表現最佳 (Min Validation Loss) 的那一個模型檔案（例如 .h5 或 $\text{SavedModel}$ 格式）儲存下來。
紀錄追蹤： 必須使用 $\text{Keras Tuner}$、$\text{Optuna}$ 或 $\text{MLflow}$ 等工具，完整紀錄所有試驗 (Trials) 的參數組合、最終驗證指標以及訓練日誌。這些紀錄應可供後續分析和模型選擇使用。
最佳模型選定標準： 以 最小化 $\text{Validation Loss}$ 作為選定最終模型的唯一標準。

訓練完後，使用者要可以選擇某天的日期，並得到20日後的預測收盤價和歷史資料的實際收盤價格(如果超過歷史資料則顯示N/A)。

/speckit.plan

使用Python 3.12.3 撰寫程式。
GPU 環境配置清單整理為精確的條列式格式。

這份清單涵蓋了作業系統、Python 環境、核心函式庫以及關鍵的軟體版本鎖定資訊，這些都是確保 TensorFlow 在 WSL2/Ubuntu 3.12 環境下成功啟用 NVIDIA GPU 的必要條件。

✅ 成功 GPU 環境配置清單 (WSL2 / Python 3.12)
1. 作業系統與硬體配置
主機作業系統: Windows 10/11 (需安裝 WSL2)

WSL2 發行版: Ubuntu 24.04 LTS (Noble Numbat)

NVIDIA 驅動 (Windows): 581.80 或最新版本（主機驅動）

CUDA Runtime (Windows): 13.0 (由 Windows 驅動提供)

2. 系統級函式庫 (於 WSL2 內安裝)
CUDA Toolkit (核心): 13.0.2-1

安裝指令： 確保 APT 來源正確後執行 sudo apt install -y cuda nvidia-cudnn

環境變數： 確保虛擬環境啟動腳本 (venv/bin/activate) 末尾已加入以下路徑：

Bash

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64
export PATH=$PATH:/usr/local/cuda/bin
3. Python 虛擬環境與關鍵套件版本
Python 版本： 3.12.3

深度學習框架 (GPU 關鍵)： tensorflow[and-cuda] ==2.16.1

註：必須使用 [and-cuda] 額外依賴項，並且精確鎖定此版本以與 Python 3.12 和 CUDA 13.0 兼容。

超參數調整： keras-tuner ==1.4.6

資料處理 (NumPy)： numpy ==1.26.2

資料處理 (Pandas)： pandas ==2.1.4

機器學習工具： scikit-learn ==1.3.2

4. 驗證指令
最終 GPU 驗證：

Bash

python -c "import tensorflow as tf; print('GPU 可用:', tf.config.list_physical_devices('GPU'))"
預期結果： GPU 可用: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]

/speckit.tasks
/speckit.implement
