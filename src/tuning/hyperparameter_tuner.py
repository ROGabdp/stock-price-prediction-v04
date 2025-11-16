"""
Keras Tuner 整合與超參數調整

實作 Keras Tuner 整合、搜尋空間定義、以及超參數調整執行邏輯
"""

import logging
from pathlib import Path
from typing import Tuple, Optional
import time

import numpy as np

try:
    from tensorflow import keras
    import keras_tuner as kt
except ImportError:
    keras = None
    kt = None

from src.models.model_builder import build_dynamic_lstm_model
from src.tuning.feature_set_selector import get_n_features_for_feature_set


def create_tuner(
    tuner_type: str = "random",
    max_trials: int = 50,
    project_name: str = "lstm_stock_tuning",
    directory: str = "logs/tuning_logs",
    overwrite: bool = False,
) -> "kt.Tuner":
    """
    建立 Keras Tuner 實例

    Args:
        tuner_type: Tuner 類型 ("random", "bayesian", "hyperband")
        max_trials: 最大試驗次數
        project_name: 專案名稱（用於儲存試驗紀錄）
        directory: 試驗紀錄儲存目錄
        overwrite: 是否覆寫現有的試驗紀錄

    Returns:
        kt.Tuner: Keras Tuner 實例

    Raises:
        ImportError: Keras Tuner 未安裝
        ValueError: 不支援的 tuner_type

    Example:
        >>> tuner = create_tuner("random", max_trials=50)
    """
    if kt is None:
        raise ImportError("Keras Tuner 未安裝，請執行: pip install keras-tuner")

    # 建立模型建構函式（使用 HyperParameters）
    def model_builder(hp):
        return build_model_with_hp(hp)

    # 根據類型建立 Tuner
    tuner_class_map = {
        "random": kt.RandomSearch,
        "bayesian": kt.BayesianOptimization,
        "hyperband": kt.Hyperband,
    }

    if tuner_type not in tuner_class_map:
        raise ValueError(
            f"不支援的 tuner_type: {tuner_type}，"
            f"可用選項: {list(tuner_class_map.keys())}"
        )

    tuner_class = tuner_class_map[tuner_type]

    # 建立 Tuner（依類型傳遞不同參數）
    common_params = {
        "hypermodel": model_builder,
        "objective": kt.Objective("val_loss", direction="min"),
        "max_trials": max_trials,
        "directory": directory,
        "project_name": project_name,
        "overwrite": overwrite,
    }

    if tuner_type == "hyperband":
        # Hyperband 需額外參數
        tuner = tuner_class(**common_params, max_epochs=50, factor=3)
    else:
        tuner = tuner_class(**common_params)

    logging.info(f"Keras Tuner 建立完成:")
    logging.info(f"  - Tuner 類型: {tuner_type}")
    logging.info(f"  - 最大試驗次數: {max_trials}")
    logging.info(f"  - 專案名稱: {project_name}")
    logging.info(f"  - 儲存目錄: {directory}")

    return tuner


def build_model_with_hp(hp: "kt.HyperParameters") -> "keras.Model":
    """
    使用 HyperParameters 建立動態 LSTM 模型

    定義 7 個維度的超參數搜尋空間:
    1. time_steps: 時間窗口 (20, 40, 60, 80)
    2. num_layers: LSTM 層數 (2, 3, 4)
    3. units: 每層單元數 (32, 64, 96, 128)
    4. dropout: Dropout 比率 (0.1 - 0.4)
    5. learning_rate: 學習率 (0.001, 0.0005, 0.0001)
    6. batch_size: 批次大小 (32, 64, 128)
    7. feature_set_id: 特徵集選擇 (Set A, Set B, Set C)

    Args:
        hp: Keras Tuner HyperParameters 物件

    Returns:
        keras.Model: 已編譯的 LSTM 模型

    Note:
        此函式會被 Keras Tuner 自動呼叫，每次試驗都會建立新模型
    """
    if keras is None:
        raise ImportError("TensorFlow/Keras 未安裝")

    # 1. 特徵集選擇（影響 n_features）
    feature_set_id = hp.Choice("feature_set_id", ["Set A", "Set B", "Set C"])
    n_features = get_n_features_for_feature_set(feature_set_id)

    # 2. 時間窗口
    time_steps = hp.Choice("time_steps", [20, 40, 60, 80])

    # 3. LSTM 層數
    num_layers = hp.Int("num_layers", min_value=2, max_value=4, step=1)

    # 4. 每層單元數（動態產生）
    units_per_layer = []
    for i in range(num_layers):
        units = hp.Choice(f"units_layer_{i+1}", [32, 64, 96, 128])
        units_per_layer.append(units)

    # 5. Dropout 比率
    dropout_rate = hp.Float("dropout_rate", min_value=0.1, max_value=0.4, step=0.1)

    # 6. 學習率
    learning_rate = hp.Choice("learning_rate", [0.001, 0.0005, 0.0001])

    # 7. 批次大小（稍後在 fit 時使用）
    # batch_size = hp.Choice("batch_size", [32, 64, 128])

    # 建立模型
    model = build_dynamic_lstm_model(
        time_steps=time_steps,
        n_features=n_features,
        n_classes=5,
        num_layers=num_layers,
        units_per_layer=units_per_layer,
        dropout_rate=dropout_rate,
        learning_rate=learning_rate,
    )

    logging.debug(f"Trial 模型建立完成:")
    logging.debug(f"  - 特徵集: {feature_set_id} ({n_features} 個特徵)")
    logging.debug(f"  - 時間窗口: {time_steps}")
    logging.debug(f"  - 層數: {num_layers}")
    logging.debug(f"  - 單元數: {units_per_layer}")
    logging.debug(f"  - Dropout: {dropout_rate}")
    logging.debug(f"  - 學習率: {learning_rate}")

    return model


def run_tuning(
    tuner: "kt.Tuner",
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs_per_trial: int = 50,
    callbacks: list = None,
    verbose: int = 1,
) -> Tuple["kt.Tuner", dict]:
    """
    執行超參數調整

    Args:
        tuner: Keras Tuner 實例
        X_train: 訓練集輸入
        y_train: 訓練集目標
        X_val: 驗證集輸入
        y_val: 驗證集目標
        epochs_per_trial: 每次試驗的訓練週期數
        callbacks: 額外的 Keras Callbacks（預設使用 EarlyStopping）
        verbose: 訓練輸出詳細程度

    Returns:
        Tuple[kt.Tuner, dict]: (完成調整的 Tuner, 調整統計資訊)

    Example:
        >>> tuner, stats = run_tuning(
        ...     tuner, X_train, y_train, X_val, y_val,
        ...     epochs_per_trial=50
        ... )
    """
    if keras is None or kt is None:
        raise ImportError("TensorFlow/Keras 或 Keras Tuner 未安裝")

    # 預設 Callbacks: Early Stopping
    if callbacks is None:
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=5, restore_best_weights=True, verbose=0
            )
        ]

    logging.info("=" * 60)
    logging.info("開始超參數調整")
    logging.info(f"訓練集: X {X_train.shape}, y {y_train.shape}")
    logging.info(f"驗證集: X {X_val.shape}, y {y_val.shape}")
    logging.info(f"每次試驗訓練週期: {epochs_per_trial}")
    logging.info(f"最大試驗次數: {tuner.oracle.max_trials}")
    logging.info("=" * 60)

    # 記錄開始時間
    start_time = time.time()

    # 執行搜尋（注意：batch_size 從 hp 取得）
    tuner.search(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs_per_trial,
        callbacks=callbacks,
        verbose=verbose,
    )

    # 計算總執行時間
    total_time = time.time() - start_time

    # 取得最佳試驗
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    # 統計資訊
    num_layers = best_hps.get("num_layers")
    stats = {
        "total_time": total_time,
        "total_trials": len(tuner.oracle.trials),
        "best_hyperparameters": {
            "feature_set_id": best_hps.get("feature_set_id"),
            "time_steps": best_hps.get("time_steps"),
            "num_layers": num_layers,
            "units_per_layer": [
                best_hps.get(f"units_layer_{i+1}")
                for i in range(num_layers)
            ],
            "dropout_rate": best_hps.get("dropout_rate"),
            "learning_rate": best_hps.get("learning_rate"),
            "batch_size": best_hps.values.get("batch_size", 32),
        },
    }

    # 取得最佳試驗的驗證損失與準確度
    best_trial = tuner.oracle.get_best_trials(num_trials=1)[0]
    stats["best_val_loss"] = best_trial.score
    stats["best_val_accuracy"] = best_trial.metrics.get_best_value("val_accuracy")

    logging.info("=" * 60)
    logging.info("超參數調整完成")
    logging.info(f"總試驗次數: {stats['total_trials']}")
    logging.info(f"總執行時間: {total_time / 3600:.2f} 小時")
    logging.info(f"最佳驗證損失: {stats['best_val_loss']:.4f}")
    logging.info(f"最佳驗證準確度: {stats['best_val_accuracy']:.2%}")
    logging.info("最佳超參數:")
    for key, value in stats["best_hyperparameters"].items():
        logging.info(f"  - {key}: {value}")
    logging.info("=" * 60)

    return tuner, stats


def get_best_model(
    tuner: "kt.Tuner", retrain: bool = False, X_train=None, y_train=None, **kwargs
) -> "keras.Model":
    """
    取得最佳模型

    Args:
        tuner: 完成調整的 Keras Tuner 實例
        retrain: 是否重新訓練最佳模型（使用完整訓練集與更多 epochs）
        X_train: 訓練集輸入（若 retrain=True 則必填）
        y_train: 訓練集目標（若 retrain=True 則必填）
        **kwargs: 傳遞給 model.fit() 的額外參數

    Returns:
        keras.Model: 最佳模型

    Example:
        >>> best_model = get_best_model(tuner)
        >>> # 或重新訓練
        >>> best_model = get_best_model(
        ...     tuner, retrain=True,
        ...     X_train=X_train, y_train=y_train,
        ...     epochs=100, validation_data=(X_val, y_val)
        ... )
    """
    if kt is None:
        raise ImportError("Keras Tuner 未安裝")

    if retrain:
        if X_train is None or y_train is None:
            raise ValueError("retrain=True 時必須提供 X_train 和 y_train")

        # 取得最佳超參數並重新訓練
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        model = tuner.hypermodel.build(best_hps)

        logging.info("使用最佳超參數重新訓練模型...")
        model.fit(X_train, y_train, **kwargs)

        logging.info("模型重新訓練完成")
    else:
        # 直接取得調整過程中的最佳模型
        model = tuner.get_best_models(num_models=1)[0]
        logging.info("取得調整過程中的最佳模型")

    return model


def save_tuner_results(
    tuner: "kt.Tuner", stats: dict, output_file: str = "logs/tuning_results.txt"
) -> None:
    """
    儲存超參數調整結果至檔案

    Args:
        tuner: 完成調整的 Keras Tuner 實例
        stats: run_tuning 回傳的統計資訊
        output_file: 輸出檔案路徑

    Example:
        >>> save_tuner_results(tuner, stats, "logs/tuning_results.txt")
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("超參數調整結果\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"總試驗次數: {stats['total_trials']}\n")
        f.write(f"總執行時間: {stats['total_time'] / 3600:.2f} 小時\n")
        f.write(f"最佳驗證損失: {stats['best_val_loss']:.4f}\n")
        f.write(f"最佳驗證準確度: {stats['best_val_accuracy']:.2%}\n\n")

        f.write("最佳超參數:\n")
        for key, value in stats["best_hyperparameters"].items():
            f.write(f"  - {key}: {value}\n")

        f.write("\n" + "=" * 60 + "\n")
        f.write("前 5 名試驗結果\n")
        f.write("=" * 60 + "\n\n")

        # 取得前 5 名試驗
        best_trials = tuner.oracle.get_best_trials(num_trials=5)
        for i, trial in enumerate(best_trials, 1):
            f.write(f"Trial {i}:\n")
            f.write(f"  - ID: {trial.trial_id}\n")
            f.write(f"  - 驗證損失: {trial.score:.4f}\n")
            f.write(f"  - 驗證準確度: {trial.metrics.get_best_value('val_accuracy'):.2%}\n")
            f.write(f"  - 超參數: {trial.hyperparameters.values}\n\n")

    logging.info(f"調整結果已儲存至: {output_file}")


if __name__ == "__main__":
    # 測試 Tuner 建立
    logging.basicConfig(level=logging.INFO)

    print("測試 Keras Tuner 整合")
    print("=" * 60)

    try:
        tuner = create_tuner(tuner_type="random", max_trials=5)
        print(f"✅ Tuner 建立成功")
        print(f"Tuner 類型: {type(tuner).__name__}")
        print(f"搜尋空間已定義")

        # 顯示搜尋空間
        print("\n超參數搜尋空間:")
        print("  - feature_set_id: Set A, Set B, Set C")
        print("  - time_steps: 20, 40, 60, 80")
        print("  - num_layers: 2, 3, 4")
        print("  - units: 32, 64, 96, 128")
        print("  - dropout_rate: 0.1 - 0.4")
        print("  - learning_rate: 0.001, 0.0005, 0.0001")
        print("  - batch_size: 32, 64, 128")

    except Exception as e:
        print(f"❌ 錯誤: {e}")
