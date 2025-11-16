"""
訓練日誌記錄工具

提供統一的日誌記錄介面，支援檔案輸出與控制台輸出。
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str,
    log_dir: str = "logs/training_logs",
    level: int = logging.INFO,
    console_output: bool = True,
) -> logging.Logger:
    """
    設置日誌記錄器

    Args:
        name: Logger 名稱
        log_dir: 日誌檔案儲存目錄
        level: 日誌級別 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        console_output: 是否同時輸出到控制台

    Returns:
        logging.Logger: 設置好的 logger 實例

    Example:
        >>> logger = setup_logger("training", log_dir="logs/training_logs")
        >>> logger.info("開始訓練模型")
    """
    # 建立 logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 清除現有的 handlers（避免重複）
    if logger.hasHandlers():
        logger.handlers.clear()

    # 建立日誌目錄
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # 建立檔案 handler
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_path / f"{name}_{timestamp}.log"

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(level)

    # 建立格式化器
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(formatter)

    # 加入 file handler
    logger.addHandler(file_handler)

    # 建立控制台 handler（若需要）
    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    logger.info(f"日誌記錄器已初始化: {log_file}")

    return logger


def log_training_start(
    logger: logging.Logger,
    model_name: str,
    feature_set: str,
    time_steps: int,
    epochs: int,
    batch_size: int,
) -> None:
    """
    記錄訓練開始資訊

    Args:
        logger: Logger 實例
        model_name: 模型名稱
        feature_set: 特徵集 ID
        time_steps: 時間窗口大小
        epochs: 訓練週期數
        batch_size: 批次大小
    """
    logger.info("=" * 60)
    logger.info("開始訓練模型")
    logger.info(f"模型名稱: {model_name}")
    logger.info(f"特徵集: {feature_set}")
    logger.info(f"時間窗口: {time_steps}")
    logger.info(f"訓練週期: {epochs}")
    logger.info(f"批次大小: {batch_size}")
    logger.info("=" * 60)


def log_training_complete(
    logger: logging.Logger,
    val_loss: float,
    val_accuracy: float,
    training_time: float,
    model_path: str,
) -> None:
    """
    記錄訓練完成資訊

    Args:
        logger: Logger 實例
        val_loss: 驗證損失
        val_accuracy: 驗證準確度
        training_time: 訓練時間（秒）
        model_path: 模型儲存路徑
    """
    logger.info("=" * 60)
    logger.info("訓練完成")
    logger.info(f"最佳驗證損失: {val_loss:.4f}")
    logger.info(f"最佳驗證準確度: {val_accuracy:.2%}")
    logger.info(f"訓練時間: {training_time / 60:.1f} 分鐘")
    logger.info(f"模型已儲存至: {model_path}")
    logger.info("=" * 60)


def log_dataset_info(
    logger: logging.Logger,
    train_size: int,
    val_size: int,
    test_size: int,
    n_features: int,
) -> None:
    """
    記錄資料集資訊

    Args:
        logger: Logger 實例
        train_size: 訓練集大小
        val_size: 驗證集大小
        test_size: 測試集大小
        n_features: 特徵數量
    """
    logger.info("資料集資訊:")
    logger.info(f"  訓練集: {train_size} 筆")
    logger.info(f"  驗證集: {val_size} 筆")
    logger.info(f"  測試集: {test_size} 筆")
    logger.info(f"  特徵數: {n_features} 個")


def log_hyperparameter_trial(
    logger: logging.Logger,
    trial_id: str,
    hyperparameters: dict,
    val_loss: float,
    val_accuracy: float,
) -> None:
    """
    記錄超參數試驗結果

    Args:
        logger: Logger 實例
        trial_id: 試驗 ID
        hyperparameters: 超參數字典
        val_loss: 驗證損失
        val_accuracy: 驗證準確度
    """
    logger.info(f"Trial {trial_id}:")
    logger.info(f"  超參數: {hyperparameters}")
    logger.info(f"  驗證損失: {val_loss:.4f}")
    logger.info(f"  驗證準確度: {val_accuracy:.2%}")


def log_prediction(
    logger: logging.Logger,
    input_date: str,
    predicted_class: int,
    confidence: float,
    prediction_date: Optional[str] = None,
) -> None:
    """
    記錄預測結果

    Args:
        logger: Logger 實例
        input_date: 輸入日期
        predicted_class: 預測類別
        confidence: 信心度
        prediction_date: 預測日期（選用）
    """
    logger.info("=" * 60)
    logger.info("預測結果")
    logger.info(f"輸入日期: {input_date}")
    if prediction_date:
        logger.info(f"預測日期: {prediction_date}")
    logger.info(f"預測類別: {predicted_class}")
    logger.info(f"信心度: {confidence:.2%}")
    logger.info("=" * 60)


if __name__ == "__main__":
    # 測試 logger 設置
    logger = setup_logger("test", log_dir="logs/test")
    logger.info("這是一條測試訊息")
    logger.warning("這是一條警告訊息")
    logger.error("這是一條錯誤訊息")

    # 測試訓練日誌
    log_training_start(
        logger,
        model_name="baseline_lstm",
        feature_set="Set A",
        time_steps=60,
        epochs=100,
        batch_size=32,
    )
