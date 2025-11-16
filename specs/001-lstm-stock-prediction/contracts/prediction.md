# 預測模組合約

## 模組職責

負責載入訓練好的模型，根據指定日期進行預測，並輸出預測結果（含實際價格比較）。

## 公開介面

### 1. load_model()

**功能**: 載入訓練好的模型

```python
def load_model(model_path: str) -> tf.keras.Model:
    """
    載入已儲存的 Keras 模型

    參數:
        model_path: 模型檔案路徑 (.h5 或 SavedModel 目錄)

    返回:
        tf.keras.Model: 載入的模型

    拋出:
        FileNotFoundError: 模型檔案不存在
        ModelLoadError: 模型載入失敗
    """
```

---

### 2. predict_for_date()

**功能**: 根據指定日期進行預測

```python
def predict_for_date(
    model: tf.keras.Model,
    input_date: str,
    historical_data: pd.DataFrame,
    feature_set_id: str,
    time_steps: int = 60
) -> Dict[str, Any]:
    """
    預測指定日期後 20 個交易日的漲跌幅區間

    參數:
        model: 已訓練的 Keras 模型
        input_date: 輸入日期 (格式: "YYYY-MM-DD")
        historical_data: 歷史資料框架
        feature_set_id: 使用的特徵集 ("Set A", "Set B", "Set C")
        time_steps: 時間窗口大小

    返回:
        dict: 預測結果，包含以下鍵值:
            - input_date: 輸入日期
            - prediction_date: 預測日期 (input_date + 20 交易日)
            - probability_vector: 5 類機率向量 (list of float)
            - predicted_class: 預測類別 (0-4)
            - confidence: 信心度 (max probability)
            - predicted_price_range: {"min": float, "max": float}
            - actual_close_price: float or None
            - timestamp: 預測執行時間

    拋出:
        ValueError: 輸入日期格式錯誤或不存在於歷史資料
        InsufficientDataError: 輸入日期前的資料不足 time_steps 天
    """
```

---

### 3. format_prediction_result()

**功能**: 格式化預測結果為可讀文字

```python
def format_prediction_result(result: Dict[str, Any]) -> str:
    """
    將預測結果格式化為使用者友善的文字

    參數:
        result: predict_for_date() 返回的結果字典

    返回:
        str: 格式化的預測結果文字
    """
```

**輸出範例**:
```
=== 台股 20 日預測結果 ===
輸入日期: 2024-01-15
預測日期: 2024-02-14 (20 個交易日後)

預測結果:
  類別: 3 (溫和上漲)
  信心度: 68.5%
  預測收盤價區間: 15,800 - 16,500 元

實際收盤價: 16,120 元

機率分佈:
  極度下跌 (<-5.0%):   2.3%
  溫和下跌 (-5.0~-2.5%): 8.7%
  區間震盪 (-2.5~+2.5%): 12.4%
  溫和上漲 (+2.5~+5.0%): 68.5% ← 預測
  極度上漲 (>+5.0%):   8.1%
```

---

## 錯誤處理

| 錯誤類型 | 觸發條件 | 錯誤訊息範例 |
|---------|---------|-------------|
| FileNotFoundError | 模型檔案不存在 | "找不到模型檔案: {model_path}" |
| ValueError | 日期格式錯誤 | "日期格式錯誤，應為 YYYY-MM-DD: {input_date}" |
| InsufficientDataError | 資料不足 | "輸入日期前需要至少 {time_steps} 天資料" |

---

## 使用範例

```python
from src.prediction import predictor

# 1. 載入最佳模型
model = predictor.load_model('models/best_tuned_model.h5')

# 2. 讀取歷史資料
df = pd.read_csv('19980601-20251111-converted.csv')

# 3. 進行預測
result = predictor.predict_for_date(
    model=model,
    input_date="2024-01-15",
    historical_data=df,
    feature_set_id="Set A",  # 與訓練時相同
    time_steps=60
)

# 4. 顯示結果
formatted_result = predictor.format_prediction_result(result)
print(formatted_result)
```
