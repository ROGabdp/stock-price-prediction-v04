"""
特徵縮放工具

提供特徵標準化與縮放功能，支援 StandardScaler 和 MinMaxScaler。
"""

import logging
from typing import Tuple, Optional
import pickle
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def create_scaler(scaler_type: str = "standard"):
    """
    建立縮放器實例

    Args:
        scaler_type: 縮放器類型 ("standard" 或 "minmax")

    Returns:
        Scaler: StandardScaler 或 MinMaxScaler 實例

    Raises:
        ValueError: 未知的縮放器類型

    Example:
        >>> scaler = create_scaler("standard")
    """
    if scaler_type.lower() == "standard":
        return StandardScaler()
    elif scaler_type.lower() == "minmax":
        return MinMaxScaler()
    else:
        raise ValueError(
            f"未知的縮放器類型: {scaler_type}，可用選項: 'standard', 'minmax'"
        )


def fit_scaler(scaler, data: np.ndarray):
    """
    訓練縮放器（僅在訓練集上 fit）

    Args:
        scaler: StandardScaler 或 MinMaxScaler 實例
        data: 訓練資料，形狀 (n_samples, n_features)

    Returns:
        Scaler: 已訓練的縮放器

    Example:
        >>> scaler = create_scaler("standard")
        >>> scaler = fit_scaler(scaler, train_data)
    """
    if data.ndim != 2:
        raise ValueError(f"資料必須是 2D 陣列，目前形狀: {data.shape}")

    scaler.fit(data)
    logging.info(f"縮放器已訓練，資料形狀: {data.shape}")

    return scaler


def transform_features(scaler, data: np.ndarray) -> np.ndarray:
    """
    使用已訓練的縮放器轉換資料

    Args:
        scaler: 已訓練的縮放器
        data: 待轉換資料，形狀 (n_samples, n_features)

    Returns:
        np.ndarray: 轉換後的資料

    Example:
        >>> train_scaled = transform_features(scaler, train_data)
        >>> val_scaled = transform_features(scaler, val_data)
    """
    if data.ndim != 2:
        raise ValueError(f"資料必須是 2D 陣列，目前形狀: {data.shape}")

    scaled_data = scaler.transform(data)
    logging.info(f"特徵縮放完成，形狀: {scaled_data.shape}")

    return scaled_data


def fit_transform_features(
    scaler, train_data: np.ndarray, val_data: Optional[np.ndarray] = None, test_data: Optional[np.ndarray] = None
) -> Tuple:
    """
    訓練縮放器並轉換訓練/驗證/測試集

    Args:
        scaler: 縮放器實例
        train_data: 訓練資料
        val_data: 驗證資料（選用）
        test_data: 測試資料（選用）

    Returns:
        Tuple: (已訓練的 scaler, 縮放後的訓練集, 縮放後的驗證集, 縮放後的測試集)

    Example:
        >>> scaler, train_scaled, val_scaled, test_scaled = fit_transform_features(
        ...     scaler, train_data, val_data, test_data
        ... )
    """
    # 在訓練集上 fit
    scaler = fit_scaler(scaler, train_data)

    # 轉換訓練集
    train_scaled = transform_features(scaler, train_data)

    # 轉換驗證集（若提供）
    val_scaled = None
    if val_data is not None:
        val_scaled = transform_features(scaler, val_data)

    # 轉換測試集（若提供）
    test_scaled = None
    if test_data is not None:
        test_scaled = transform_features(scaler, test_data)

    logging.info("=" * 60)
    logging.info("特徵縮放完成")
    logging.info(f"縮放器類型: {scaler.__class__.__name__}")
    logging.info(f"訓練集: {train_scaled.shape}")
    if val_scaled is not None:
        logging.info(f"驗證集: {val_scaled.shape}")
    if test_scaled is not None:
        logging.info(f"測試集: {test_scaled.shape}")
    logging.info("=" * 60)

    return scaler, train_scaled, val_scaled, test_scaled


def save_scaler(scaler, file_path: str) -> None:
    """
    儲存縮放器至檔案

    Args:
        scaler: 已訓練的縮放器
        file_path: 儲存路徑

    Example:
        >>> save_scaler(scaler, "models/scaler.pkl")
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, "wb") as f:
        pickle.dump(scaler, f)

    logging.info(f"縮放器已儲存至: {file_path}")


def load_scaler(file_path: str):
    """
    從檔案載入縮放器

    Args:
        file_path: 縮放器檔案路徑

    Returns:
        Scaler: 載入的縮放器實例

    Raises:
        FileNotFoundError: 檔案不存在

    Example:
        >>> scaler = load_scaler("models/scaler.pkl")
    """
    if not Path(file_path).exists():
        raise FileNotFoundError(f"找不到縮放器檔案: {file_path}")

    with open(file_path, "rb") as f:
        scaler = pickle.load(f)

    logging.info(f"縮放器已載入: {file_path}")

    return scaler


def inverse_transform_features(scaler, scaled_data: np.ndarray) -> np.ndarray:
    """
    反向轉換（將縮放後的資料還原為原始尺度）

    Args:
        scaler: 已訓練的縮放器
        scaled_data: 縮放後的資料

    Returns:
        np.ndarray: 還原後的資料

    Example:
        >>> original_data = inverse_transform_features(scaler, scaled_data)
    """
    if scaled_data.ndim != 2:
        raise ValueError(f"資料必須是 2D 陣列，目前形狀: {scaled_data.shape}")

    original_data = scaler.inverse_transform(scaled_data)
    logging.info(f"反向轉換完成，形狀: {original_data.shape}")

    return original_data


if __name__ == "__main__":
    # 測試縮放器
    logging.basicConfig(level=logging.INFO)

    # 建立模擬資料
    np.random.seed(42)
    train_data = np.random.randn(1000, 10)
    val_data = np.random.randn(200, 10)
    test_data = np.random.randn(200, 10)

    # 測試 StandardScaler
    print("\n測試 StandardScaler:")
    scaler = create_scaler("standard")
    scaler, train_scaled, val_scaled, test_scaled = fit_transform_features(
        scaler, train_data, val_data, test_data
    )

    print(f"\n訓練集縮放後統計:")
    print(f"  平均值: {train_scaled.mean(axis=0)[:5]}")  # 應接近 0
    print(f"  標準差: {train_scaled.std(axis=0)[:5]}")  # 應接近 1

    # 測試儲存與載入
    save_scaler(scaler, "models/test_scaler.pkl")
    loaded_scaler = load_scaler("models/test_scaler.pkl")
    print(f"\n縮放器載入成功: {loaded_scaler.__class__.__name__}")

    # 清理測試檔案
    Path("models/test_scaler.pkl").unlink()
