"""
基準 LSTM 模型

定義基準 LSTM 模型架構（3 層：128-64-32 單元，Dropout 0.2）
"""

import logging
from typing import Tuple

try:
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.models import Sequential
except ImportError:
    keras = None
    layers = None
    Sequential = None


def build_lstm_model(
    time_steps: int,
    n_features: int,
    n_classes: int = 5,
    lstm_units: Tuple[int, int, int] = (128, 64, 32),
    dropout_rate: float = 0.2,
    learning_rate: float = 0.001,
) -> "keras.Model":
    """
    建構基準 LSTM 模型

    模型架構:
        - LSTM 層 1: 128 單元 (return_sequences=True)
        - Dropout: 0.2
        - LSTM 層 2: 64 單元 (return_sequences=True)
        - Dropout: 0.2
        - LSTM 層 3: 32 單元 (return_sequences=False)
        - Dropout: 0.2
        - Dense 層: 5 單元 (softmax 激活函數)

    Args:
        time_steps: 時間窗口大小（例如 60）
        n_features: 輸入特徵數量（例如 14）
        n_classes: 輸出類別數（固定為 5）
        lstm_units: 各層 LSTM 單元數（預設 128, 64, 32）
        dropout_rate: Dropout 比率（預設 0.2）
        learning_rate: 學習率（預設 0.001）

    Returns:
        keras.Model: 編譯後的 Keras 模型

    Example:
        >>> model = build_lstm_model(time_steps=60, n_features=14)
        >>> print(model.summary())
    """
    if keras is None:
        raise ImportError("TensorFlow/Keras 未安裝，請執行: pip install tensorflow-gpu==2.10.0")

    logging.info("=" * 60)
    logging.info("建構基準 LSTM 模型")
    logging.info(f"輸入形狀: ({time_steps}, {n_features})")
    logging.info(f"輸出類別數: {n_classes}")
    logging.info(f"LSTM 單元數: {lstm_units}")
    logging.info(f"Dropout 比率: {dropout_rate}")
    logging.info(f"學習率: {learning_rate}")

    # 建立模型
    model = Sequential(name="LSTM_Baseline")

    # LSTM 層 1
    model.add(
        layers.LSTM(
            units=lstm_units[0],
            return_sequences=True,
            input_shape=(time_steps, n_features),
            name="LSTM_1",
        )
    )
    model.add(layers.Dropout(dropout_rate, name="Dropout_1"))

    # LSTM 層 2
    model.add(
        layers.LSTM(
            units=lstm_units[1],
            return_sequences=True,
            name="LSTM_2",
        )
    )
    model.add(layers.Dropout(dropout_rate, name="Dropout_2"))

    # LSTM 層 3
    model.add(
        layers.LSTM(
            units=lstm_units[2],
            return_sequences=False,
            name="LSTM_3",
        )
    )
    model.add(layers.Dropout(dropout_rate, name="Dropout_3"))

    # 輸出層
    model.add(
        layers.Dense(
            units=n_classes,
            activation="softmax",
            name="Output",
        )
    )

    # 編譯模型
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    logging.info("✅ 模型建構完成")
    logging.info(f"總參數數量: {model.count_params():,}")
    logging.info("=" * 60)

    return model


def get_model_architecture_info(model: "keras.Model") -> dict:
    """
    取得模型架構資訊

    Args:
        model: Keras 模型

    Returns:
        dict: 模型架構資訊

    Example:
        >>> info = get_model_architecture_info(model)
        >>> print(info['total_params'])
    """
    if keras is None:
        return {"error": "TensorFlow/Keras 未安裝"}

    architecture = {
        "model_name": model.name,
        "total_params": model.count_params(),
        "trainable_params": sum(
            [keras.backend.count_params(w) for w in model.trainable_weights]
        ),
        "layers": [],
    }

    for layer in model.layers:
        layer_info = {
            "name": layer.name,
            "type": layer.__class__.__name__,
            "output_shape": layer.output_shape,
        }

        # 如果是 LSTM 或 Dense 層，加入參數資訊
        if hasattr(layer, "units"):
            layer_info["units"] = layer.units

        architecture["layers"].append(layer_info)

    return architecture


if __name__ == "__main__":
    # 測試模型建構
    logging.basicConfig(level=logging.INFO)

    try:
        # 建構模型
        model = build_lstm_model(time_steps=60, n_features=14)

        # 顯示模型摘要
        print("\n模型摘要:")
        model.summary()

        # 顯示架構資訊
        info = get_model_architecture_info(model)
        print(f"\n模型資訊:")
        print(f"  名稱: {info['model_name']}")
        print(f"  總參數數: {info['total_params']:,}")
        print(f"  可訓練參數數: {info['trainable_params']:,}")

    except Exception as e:
        print(f"錯誤: {str(e)}")
