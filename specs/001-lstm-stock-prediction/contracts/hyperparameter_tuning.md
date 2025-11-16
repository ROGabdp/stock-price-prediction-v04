# 超參數調整模組合約

## 模組職責

負責使用 Keras Tuner 執行自動超參數調整，探索最佳模型配置（含特徵集選擇）。

## 公開介面

### 1. create_tuner()

**功能**: 建立 Keras Tuner 實例

```python
def create_tuner(
    objective: str = 'val_loss',
    max_trials: int = 50,
    executions_per_trial: int = 1,
    directory: str = 'logs/tuning',
    project_name: str = 'lstm_stock_prediction'
) -> kt.Tuner:
    """
    建立 Keras Tuner (RandomSearch 或 BayesianOptimization)

    參數:
        objective: 優化目標 ('val_loss' 或 'val_accuracy')
        max_trials: 最大試驗次數
        executions_per_trial: 每次試驗執行次數
        directory: 試驗紀錄目錄
        project_name: 專案名稱

    返回:
        kt.Tuner: Keras Tuner 實例
    """
```

---

### 2. define_search_space()

**功能**: 定義超參數搜尋空間

```python
def define_search_space(hp: kt.HyperParameters) -> tf.keras.Model:
    """
    定義超參數搜尋空間並建構模型

    參數:
        hp: Keras Tuner HyperParameters 物件

    返回:
        tf.keras.Model: 根據 hp 建構的模型

    超參數範圍:
        - time_steps: [20, 40, 60, 80]
        - num_layers: [2, 3, 4]
        - units: [32, 64, 128]
        - dropout_rate: [0.1, 0.4]
        - learning_rate: [0.001, 0.0005, 0.0001]
        - batch_size: [32, 64, 128]
        - feature_set_id: ["Set A", "Set B", "Set C"]
    """
```

---

### 3. run_tuning()

**功能**: 執行超參數調整

```python
def run_tuning(
    tuner: kt.Tuner,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 50,
    verbose: int = 1
) -> kt.Tuner:
    """
    執行超參數調整流程

    參數:
        tuner: Keras Tuner 實例
        X_train, y_train: 訓練資料
        X_val, y_val: 驗證資料
        epochs: 每次試驗的最大訓練輪數
        verbose: 詳細程度 (0, 1, 2)

    返回:
        kt.Tuner: 完成調整的 tuner (包含試驗歷史)
    """
```

---

### 4. get_best_model()

**功能**: 獲取最佳模型與超參數

```python
def get_best_model(tuner: kt.Tuner) -> Tuple[tf.keras.Model, Dict[str, Any]]:
    """
    獲取驗證損失最小的最佳模型

    參數:
        tuner: 完成調整的 Keras Tuner

    返回:
        (model, hyperparameters): 最佳模型與超參數字典
    """
```

---

## 使用範例

```python
from src.tuning import hyperparameter_tuner

# 1. 建立 tuner
tuner = hyperparameter_tuner.create_tuner(
    objective='val_loss',
    max_trials=50,
    directory='logs/tuning'
)

# 2. 執行調整 (自動探索所有超參數組合，含特徵集)
tuner = hyperparameter_tuner.run_tuning(
    tuner=tuner,
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
    epochs=50
)

# 3. 獲取最佳模型
best_model, best_hp = hyperparameter_tuner.get_best_model(tuner)

print("最佳超參數:")
print(f"  時間窗口: {best_hp['time_steps']}")
print(f"  LSTM 層數: {best_hp['num_layers']}")
print(f"  特徵集: {best_hp['feature_set_id']}")
print(f"  驗證損失: {tuner.get_best_models(1)[0].evaluate(X_val, y_val)[0]:.4f}")

# 4. 儲存最佳模型
best_model.save('models/best_tuned_model.h5')
```
