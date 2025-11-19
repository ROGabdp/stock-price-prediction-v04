"""
模型訓練邏輯

提供模型訓練、Callbacks 配置、模型儲存等功能。
"""

import logging
from pathlib import Path
from typing import Tuple, Optional
import time
from datetime import datetime

import numpy as np

try:
    from tensorflow import keras
    from tensorflow.keras.callbacks import (
        EarlyStopping,
        ModelCheckpoint,
        CSVLogger,
        ReduceLROnPlateau,
    )
except ImportError:
    keras = None


def train_model(
    model: "keras.Model",
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 100,
    batch_size: int = 32,
    output_dir: str = "models",
    model_name: str = "baseline_model",
    patience: int = 10,
    verbose: int = 1,
    add_timestamp: bool = True,
) -> Tuple["keras.Model", dict]:
    """
    訓練 LSTM 模型

    Args:
        model: 已編譯的 Keras 模型
        X_train: 訓練集輸入，形狀 (n_train, time_steps, n_features)
        y_train: 訓練集目標，形狀 (n_train, n_classes)
        X_val: 驗證集輸入，形狀 (n_val, time_steps, n_features)
        y_val: 驗證集目標，形狀 (n_val, n_classes)
        epochs: 訓練週期數
        batch_size: 批次大小
        output_dir: 模型與日誌儲存目錄
        model_name: 模型名稱（如果 add_timestamp=True，會自動加入時間戳記）
        patience: Early Stopping 的耐心值
        verbose: 訓練輸出詳細程度 (0=靜默, 1=進度條, 2=每個 epoch 一行)
        add_timestamp: 是否在模型名稱中加入時間戳記 (預設 True)

    Returns:
        Tuple[keras.Model, dict]: (訓練後的模型, 訓練歷史)

    Example:
        >>> model, history = train_model(
        ...     model, X_train, y_train, X_val, y_val,
        ...     epochs=100, batch_size=32
        ... )
    """
    if keras is None:
        raise ImportError("TensorFlow/Keras 未安裝")

    # 如果需要，在模型名稱中加入時間戳記
    if add_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name_with_timestamp = f"{model_name}_{timestamp}"
    else:
        model_name_with_timestamp = model_name

    # 建立輸出目錄
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    log_dir = Path("logs/training_logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    # 配置 Callbacks
    callbacks = create_callbacks(
        output_dir=output_dir,
        model_name=model_name_with_timestamp,
        patience=patience,
    )

    logging.info("=" * 60)
    logging.info("開始訓練模型")
    logging.info(f"模型名稱: {model_name_with_timestamp}")
    logging.info(f"訓練集: X {X_train.shape}, y {y_train.shape}")
    logging.info(f"驗證集: X {X_val.shape}, y {y_val.shape}")
    logging.info(f"訓練週期: {epochs}")
    logging.info(f"批次大小: {batch_size}")
    logging.info(f"Early Stopping Patience: {patience}")
    logging.info("=" * 60)

    # 記錄開始時間
    start_time = time.time()

    # 訓練模型
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=verbose,
    )

    # 計算訓練時間
    training_time = time.time() - start_time

    # 取得最佳結果
    best_epoch = np.argmin(history.history["val_loss"]) + 1
    best_val_loss = min(history.history["val_loss"])
    best_val_accuracy = max(history.history["val_accuracy"])

    logging.info("=" * 60)
    logging.info("訓練完成")
    logging.info(f"最佳 Epoch: {best_epoch}/{len(history.history['val_loss'])}")
    logging.info(f"最佳驗證損失: {best_val_loss:.4f}")
    logging.info(f"最佳驗證準確度: {best_val_accuracy:.2%}")
    logging.info(f"訓練時間: {training_time / 60:.1f} 分鐘")
    logging.info(f"模型已儲存至: {output_dir}/{model_name_with_timestamp}.h5")
    logging.info("=" * 60)

    # 將訓練時間和模型名稱加入歷史
    history.history["training_time"] = training_time
    history.history["best_epoch"] = best_epoch
    history.history["model_name"] = model_name_with_timestamp

    return model, history.history


def create_callbacks(
    output_dir: str,
    model_name: str,
    patience: int = 10,
) -> list:
    """
    建立訓練 Callbacks

    Callbacks 包含:
        - EarlyStopping: 監控驗證損失，patience=10
        - ModelCheckpoint: 儲存最佳模型
        - CSVLogger: 記錄訓練歷史至 CSV
        - ReduceLROnPlateau: 動態調整學習率

    Args:
        output_dir: 模型儲存目錄
        model_name: 模型名稱
        patience: Early Stopping 耐心值

    Returns:
        list: Callbacks 清單

    Example:
        >>> callbacks = create_callbacks("models", "baseline_model", patience=10)
    """
    if keras is None:
        raise ImportError("TensorFlow/Keras 未安裝")

    # 模型檔案路徑
    model_file = Path(output_dir) / f"{model_name}.h5"

    # 日誌檔案路徑
    log_file = Path("logs/training_logs") / f"{model_name}_training_log.csv"

    callbacks = [
        # Early Stopping: 監控驗證損失
        EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
            mode="min",
        ),
        # Model Checkpoint: 儲存最佳模型
        ModelCheckpoint(
            filepath=str(model_file),
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=False,
            mode="min",
            verbose=1,
        ),
        # CSV Logger: 記錄訓練歷史
        CSVLogger(
            filename=str(log_file),
            separator=",",
            append=False,
        ),
        # Reduce LR On Plateau: 動態調整學習率
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1,
            mode="min",
        ),
    ]

    logging.info(f"Callbacks 配置完成:")
    logging.info(f"  - EarlyStopping (patience={patience})")
    logging.info(f"  - ModelCheckpoint ({model_file})")
    logging.info(f"  - CSVLogger ({log_file})")
    logging.info(f"  - ReduceLROnPlateau (factor=0.5, patience=5)")

    return callbacks


def evaluate_model(
    model: "keras.Model", X_test: np.ndarray, y_test: np.ndarray
) -> Tuple[float, float]:
    """
    評估模型在測試集上的表現

    Args:
        model: 訓練好的模型
        X_test: 測試集輸入
        y_test: 測試集目標

    Returns:
        Tuple[float, float]: (測試損失, 測試準確度)

    Example:
        >>> test_loss, test_acc = evaluate_model(model, X_test, y_test)
        >>> print(f"測試準確度: {test_acc:.2%}")
    """
    if keras is None:
        raise ImportError("TensorFlow/Keras 未安裝")

    logging.info("評估模型...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)

    logging.info(f"測試損失: {test_loss:.4f}")
    logging.info(f"測試準確度: {test_accuracy:.2%}")

    return test_loss, test_accuracy


def build_dynamic_lstm_model(
    time_steps: int,
    n_features: int,
    n_classes: int = 5,
    num_layers: int = 3,
    units_per_layer: Optional[list] = None,
    dropout_rate: float = 0.2,
    learning_rate: float = 0.001,
) -> "keras.Model":
    """
    建立動態配置的 LSTM 模型（用於超參數調整）

    Args:
        time_steps: 時間窗口大小（輸入序列長度）
        n_features: 輸入特徵數量
        n_classes: 輸出類別數（預設 5 類）
        num_layers: LSTM 層數（2, 3, 或 4）
        units_per_layer: 每層的單元數清單（如 [128, 64, 32]）
            若為 None，則依層數自動產生遞減序列
        dropout_rate: Dropout 比率（0.1 - 0.4）
        learning_rate: 學習率（0.001, 0.0005, 0.0001）

    Returns:
        keras.Model: 已編譯的 LSTM 模型

    Example:
        >>> # 自動產生單元數序列
        >>> model = build_dynamic_lstm_model(
        ...     time_steps=60, n_features=14, num_layers=3,
        ...     dropout_rate=0.2, learning_rate=0.001
        ... )
        >>> # 或自訂單元數序列
        >>> model = build_dynamic_lstm_model(
        ...     time_steps=60, n_features=14, num_layers=3,
        ...     units_per_layer=[128, 96, 64],
        ...     dropout_rate=0.3, learning_rate=0.0005
        ... )
    """
    if keras is None:
        raise ImportError("TensorFlow/Keras 未安裝")

    # 若未提供單元數序列，則自動產生遞減序列
    if units_per_layer is None:
        # 預設起始單元數: 128
        # 每層遞減為前一層的 3/4 或 1/2
        base_units = 128
        units_per_layer = []
        for i in range(num_layers):
            units = max(32, int(base_units / (2 ** (i * 0.5))))
            units_per_layer.append(units)

    # 驗證輸入
    if len(units_per_layer) != num_layers:
        raise ValueError(
            f"units_per_layer 長度 ({len(units_per_layer)}) "
            f"必須等於 num_layers ({num_layers})"
        )

    if not (0.0 <= dropout_rate <= 0.5):
        raise ValueError(f"dropout_rate 必須在 0.0 - 0.5 之間，當前: {dropout_rate}")

    # 建立 Sequential 模型
    model = keras.Sequential(name="Dynamic_LSTM")

    # 第一層 LSTM（需要指定 input_shape）
    model.add(
        keras.layers.LSTM(
            units_per_layer[0],
            return_sequences=(num_layers > 1),
            input_shape=(time_steps, n_features),
            name=f"LSTM_Layer_1",
        )
    )
    model.add(keras.layers.Dropout(dropout_rate, name=f"Dropout_1"))

    # 中間 LSTM 層
    for i in range(1, num_layers - 1):
        model.add(
            keras.layers.LSTM(
                units_per_layer[i],
                return_sequences=True,
                name=f"LSTM_Layer_{i+1}",
            )
        )
        model.add(keras.layers.Dropout(dropout_rate, name=f"Dropout_{i+1}"))

    # 最後一層 LSTM（不返回序列）
    if num_layers > 1:
        model.add(
            keras.layers.LSTM(
                units_per_layer[-1],
                return_sequences=False,
                name=f"LSTM_Layer_{num_layers}",
            )
        )
        model.add(keras.layers.Dropout(dropout_rate, name=f"Dropout_{num_layers}"))

    # 輸出層（Dense + Softmax）
    model.add(
        keras.layers.Dense(
            n_classes,
            activation="softmax",
            name="Output_Layer",
        )
    )

    # 編譯模型
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    logging.info("動態 LSTM 模型建立完成:")
    logging.info(f"  - 層數: {num_layers}")
    logging.info(f"  - 每層單元數: {units_per_layer}")
    logging.info(f"  - Dropout: {dropout_rate}")
    logging.info(f"  - 學習率: {learning_rate}")
    logging.info(f"  - 輸入形狀: (batch, {time_steps}, {n_features})")
    logging.info(f"  - 輸出形狀: (batch, {n_classes})")

    return model


def save_model(model: "keras.Model", file_path: str) -> None:
    """
    儲存模型至檔案

    Args:
        model: Keras 模型
        file_path: 儲存路徑 (.h5 格式)

    Example:
        >>> save_model(model, "models/my_model.h5")
    """
    if keras is None:
        raise ImportError("TensorFlow/Keras 未安裝")

    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    model.save(file_path)
    logging.info(f"模型已儲存至: {file_path}")


def load_model(file_path: str) -> "keras.Model":
    """
    從檔案載入模型

    Args:
        file_path: 模型檔案路徑

    Returns:
        keras.Model: 載入的模型

    Raises:
        FileNotFoundError: 模型檔案不存在

    Example:
        >>> model = load_model("models/baseline_model.h5")
    """
    if keras is None:
        raise ImportError("TensorFlow/Keras 未安裝")

    if not Path(file_path).exists():
        raise FileNotFoundError(f"找不到模型檔案: {file_path}")

    model = keras.models.load_model(file_path)
    logging.info(f"模型已載入: {file_path}")

    return model


if __name__ == "__main__":
    # 測試 Callbacks 建立
    logging.basicConfig(level=logging.INFO)

    callbacks = create_callbacks("models", "test_model", patience=10)
    print(f"\n建立 {len(callbacks)} 個 Callbacks")
