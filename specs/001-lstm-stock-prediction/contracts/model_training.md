# 模型訓練模組合約

## 模組職責

負責建構 LSTM 模型、訓練基準模型、儲存最佳模型，並記錄訓練歷史。

## 公開介面

### 1. build_lstm_model()

**功能**: 建構 LSTM 模型

```python
def build_lstm_model(
    time_steps: int,
    n_features: int,
    num_layers: int = 3,
    units_per_layer: List[int] = [128, 64, 32],
    dropout_rate: float = 0.2,
    learning_rate: float = 0.001
) -> tf.keras.Model:
    """
    建構 Stacked LSTM 模型

    參數:
        time_steps: 時間窗口大小
        n_features: 輸入特徵數量
        num_layers: LSTM 層數
        units_per_layer: 每層單元數清單
        dropout_rate: Dropout 比率
        learning_rate: 學習率

    返回:
        tf.keras.Model: 編譯完成的 Keras 模型
    """
```

**預設配置 (基準模型)**:
- 3 層 LSTM: [128, 64, 32]
- Dropout: 0.2 (每層之間)
- 輸出層: Dense(5, activation='softmax')
- 損失函數: categorical_crossentropy
- 優化器: Adam(lr=0.001)

---

### 2. train_model()

**功能**: 訓練 LSTM 模型

```python
def train_model(
    model: tf.keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 100,
    batch_size: int = 32,
    callbacks: List[tf.keras.callbacks.Callback] = None
) -> tf.keras.callbacks.History:
    """
    訓練模型並返回訓練歷史

    參數:
        model: Keras 模型
        X_train, y_train: 訓練資料
        X_val, y_val: 驗證資料
        epochs: 最大訓練輪數
        batch_size: 批次大小
        callbacks: Keras callbacks 清單

    返回:
        History: 訓練歷史物件
    """
```

**預設 Callbacks**:
- EarlyStopping(monitor='val_loss', patience=10)
- ModelCheckpoint(filepath='models/best_model.h5', save_best_only=True)
- CSVLogger(filename='logs/training_log.csv')

---

## 使用範例

```python
from src.models import lstm_baseline, model_builder

# 建構基準模型
model = lstm_baseline.build_lstm_model(
    time_steps=60,
    n_features=14,  # Set A 特徵數
    num_layers=3,
    units_per_layer=[128, 64, 32],
    dropout_rate=0.2,
    learning_rate=0.001
)

print(model.summary())

# 訓練模型
history = model_builder.train_model(
    model=model,
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
    epochs=100,
    batch_size=32
)

print(f"✅ 訓練完成，最佳驗證損失: {min(history.history['val_loss']):.4f}")
```
