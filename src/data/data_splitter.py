"""
資料集分割模組

提供時間序列資料的訓練/驗證/測試集分割功能，確保無資料洩漏。
"""

import logging
from typing import Tuple

import numpy as np
import pandas as pd


def split_time_series(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    時間序列分割（按時間順序）

    Args:
        df: 完整的 DataFrame（必須已按日期排序）
        train_ratio: 訓練集比例
        val_ratio: 驗證集比例
        test_ratio: 測試集比例

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: (訓練集, 驗證集, 測試集)

    Raises:
        ValueError: 比例總和不為 1

    Example:
        >>> train_df, val_df, test_df = split_time_series(df)
        >>> print(f"訓練集: {len(train_df)}, 驗證集: {len(val_df)}, 測試集: {len(test_df)}")
    """
    # 驗證比例
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError(
            f"比例總和必須為 1，目前為 {train_ratio + val_ratio + test_ratio}"
        )

    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    # 分割資料
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    # 記錄分割資訊
    logging.info("=" * 60)
    logging.info("資料集分割完成（時間序列分割）")
    logging.info(f"訓練集: {len(train_df)} 筆 ({train_ratio:.1%})")
    logging.info(f"  日期範圍: {train_df['date'].min()} ~ {train_df['date'].max()}")
    logging.info(f"驗證集: {len(val_df)} 筆 ({val_ratio:.1%})")
    logging.info(f"  日期範圍: {val_df['date'].min()} ~ {val_df['date'].max()}")
    logging.info(f"測試集: {len(test_df)} 筆 ({test_ratio:.1%})")
    logging.info(f"  日期範圍: {test_df['date'].min()} ~ {test_df['date'].max()}")
    logging.info("=" * 60)

    # 驗證無資料洩漏
    if train_df["date"].max() >= val_df["date"].min():
        logging.warning("⚠️ 警告: 訓練集與驗證集日期可能有重疊")
    if val_df["date"].max() >= test_df["date"].min():
        logging.warning("⚠️ 警告: 驗證集與測試集日期可能有重疊")

    return train_df, val_df, test_df


def create_sequences(
    features: np.ndarray, target: np.ndarray, time_steps: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    建立時間序列窗口（滑動窗口）

    Args:
        features: 特徵矩陣，形狀 (n_samples, n_features)
        target: 目標變數，形狀 (n_samples, n_classes) 或 (n_samples,)
        time_steps: 時間窗口大小（例如 60 表示使用前 60 天預測）

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - X: 形狀 (n_sequences, time_steps, n_features)
            - y: 形狀 (n_sequences, n_classes) 或 (n_sequences,)

    Example:
        >>> X, y = create_sequences(features, target, time_steps=60)
        >>> print(f"X shape: {X.shape}, y shape: {y.shape}")
        X shape: (6700, 60, 14), y shape: (6700, 5)
    """
    if len(features) != len(target):
        raise ValueError(
            f"特徵與目標長度不一致: {len(features)} vs {len(target)}"
        )

    if len(features) < time_steps:
        raise ValueError(
            f"資料長度 {len(features)} 小於時間窗口 {time_steps}"
        )

    X, y = [], []

    for i in range(len(features) - time_steps):
        X.append(features[i : i + time_steps])
        y.append(target[i + time_steps])

    X = np.array(X)
    y = np.array(y)

    logging.info(f"建立序列完成: X shape {X.shape}, y shape {y.shape}")

    return X, y


def prepare_training_data(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list,
    target_column: str,
    time_steps: int,
) -> dict:
    """
    準備訓練資料（特徵提取 + 序列建立）

    Args:
        train_df: 訓練集 DataFrame
        val_df: 驗證集 DataFrame
        test_df: 測試集 DataFrame
        feature_columns: 特徵欄位清單
        target_column: 目標變數欄位名稱
        time_steps: 時間窗口大小

    Returns:
        dict: 包含 X_train, y_train, X_val, y_val, X_test, y_test

    Example:
        >>> data = prepare_training_data(
        ...     train_df, val_df, test_df,
        ...     feature_columns=['close', 'volume'],
        ...     target_column='target',
        ...     time_steps=60
        ... )
        >>> X_train = data['X_train']
    """
    # 提取特徵與目標
    train_features = train_df[feature_columns].values
    train_target = train_df[target_column].values

    val_features = val_df[feature_columns].values
    val_target = val_df[target_column].values

    test_features = test_df[feature_columns].values
    test_target = test_df[target_column].values

    # 建立時間序列
    X_train, y_train = create_sequences(train_features, train_target, time_steps)
    X_val, y_val = create_sequences(val_features, val_target, time_steps)
    X_test, y_test = create_sequences(test_features, test_target, time_steps)

    logging.info("=" * 60)
    logging.info("訓練資料準備完成")
    logging.info(f"訓練集: X {X_train.shape}, y {y_train.shape}")
    logging.info(f"驗證集: X {X_val.shape}, y {y_val.shape}")
    logging.info(f"測試集: X {X_test.shape}, y {y_test.shape}")
    logging.info("=" * 60)

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
    }


if __name__ == "__main__":
    # 測試資料分割
    logging.basicConfig(level=logging.INFO)

    # 建立模擬資料
    n_samples = 1000
    dates = pd.date_range("2020-01-01", periods=n_samples, freq="D")
    df = pd.DataFrame(
        {
            "date": dates,
            "close": np.random.randn(n_samples),
            "volume": np.random.randn(n_samples),
        }
    )

    # 測試分割
    train_df, val_df, test_df = split_time_series(df)

    # 測試序列建立
    features = df[["close", "volume"]].values
    target = np.random.randint(0, 5, size=n_samples)
    X, y = create_sequences(features, target, time_steps=60)

    print(f"\n序列形狀: X {X.shape}, y {y.shape}")
