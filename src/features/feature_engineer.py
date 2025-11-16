"""
特徵工程模組

負責從原始股價資料中工程化特徵，並計算目標變數（5 類 One-Hot 編碼）。
"""

import logging
from typing import Tuple

import pandas as pd
import numpy as np

from .feature_sets import get_feature_names


def engineer_features(df: pd.DataFrame, feature_set_id: str) -> pd.DataFrame:
    """
    執行特徵工程（根據特徵集配置）

    Args:
        df: 原始 DataFrame（包含 OHLCV 與技術指標）
        feature_set_id: 特徵集 ID ("Set A", "Set B", "Set C")

    Returns:
        pd.DataFrame: 包含工程化特徵的 DataFrame

    Example:
        >>> df_features = engineer_features(df, "Set A")
        >>> print(df_features.columns)
    """
    df = df.copy()

    logging.info(f"開始特徵工程，使用特徵集: {feature_set_id}")

    # 1. 計算價格變化率（返回率）
    df["open_return"] = df["open"].pct_change()
    df["high_return"] = df["high"].pct_change()
    df["low_return"] = df["low"].pct_change()
    df["close_return"] = df["close"].pct_change()

    # 2. 計算 Volume 變化率
    df["volume_change"] = df["volume"].pct_change()

    # 3. 計算 SMA 比例與斜率
    df["sma20_sma60_ratio"] = df["SMA20"] / df["SMA60"]
    df["sma20_slope"] = df["SMA20"].pct_change()

    # 4. 計算法人籌碼變化率
    df["net_buy_sell_change"] = df["net buy sell"].pct_change()

    # 5. 處理無限值與缺失值
    df = df.replace([np.inf, -np.inf], np.nan)

    # 6. 移除前幾列（因為 pct_change 會產生 NaN）
    initial_rows = len(df)
    df = df.dropna()
    rows_removed = initial_rows - len(df)
    if rows_removed > 0:
        logging.info(f"移除 {rows_removed} 筆包含 NaN 的資料列")

    # 7. 取得特徵集定義的特徵欄位
    feature_columns = get_feature_names(feature_set_id)

    # 8. 驗證所有特徵欄位都存在
    missing_features = set(feature_columns) - set(df.columns)
    if missing_features:
        raise ValueError(f"缺少特徵欄位: {missing_features}")

    logging.info(f"✅ 特徵工程完成，特徵數: {len(feature_columns)}")

    return df


def calculate_target_variable(df: pd.DataFrame, look_ahead_days: int = 20) -> pd.DataFrame:
    """
    計算目標變數：未來 N 天的漲跌幅區間（5 類分類）

    漲跌幅區間定義:
        - 類別 0: R < -5.0% (極度下跌)
        - 類別 1: -5.0% ≤ R < -2.5% (溫和下跌)
        - 類別 2: -2.5% ≤ R ≤ +2.5% (區間震盪)
        - 類別 3: +2.5% < R ≤ +5.0% (溫和上漲)
        - 類別 4: R > +5.0% (極度上漲)

    Args:
        df: DataFrame（必須包含 'close' 欄位）
        look_ahead_days: 預測未來幾天（預設 20 個交易日）

    Returns:
        pd.DataFrame: 新增 'target_return' 與 'target_class' 欄位

    Example:
        >>> df_with_target = calculate_target_variable(df, look_ahead_days=20)
        >>> print(df_with_target['target_class'].value_counts())
    """
    df = df.copy()

    # 計算未來 N 天的收盤價變化率
    df["future_close"] = df["close"].shift(-look_ahead_days)
    df["target_return"] = (df["future_close"] - df["close"]) / df["close"]

    # 定義類別邊界
    def classify_return(return_value):
        if pd.isna(return_value):
            return np.nan
        elif return_value < -0.05:
            return 0  # 極度下跌
        elif return_value < -0.025:
            return 1  # 溫和下跌
        elif return_value <= 0.025:
            return 2  # 區間震盪
        elif return_value <= 0.05:
            return 3  # 溫和上漲
        else:
            return 4  # 極度上漲

    df["target_class"] = df["target_return"].apply(classify_return)

    # 移除最後 N 筆（因為沒有未來資料）
    initial_rows = len(df)
    df = df.dropna(subset=["target_class"])
    rows_removed = initial_rows - len(df)

    logging.info(f"目標變數計算完成，移除最後 {rows_removed} 筆（無未來資料）")

    # 顯示類別分佈
    class_counts = df["target_class"].value_counts().sort_index()
    logging.info("目標類別分佈:")
    class_names = ["極度下跌", "溫和下跌", "區間震盪", "溫和上漲", "極度上漲"]
    for cls, count in class_counts.items():
        logging.info(f"  類別 {int(cls)} ({class_names[int(cls)]}): {count} 筆 ({count/len(df):.1%})")

    return df


def create_one_hot_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    將目標類別轉換為 One-Hot 編碼

    Args:
        df: DataFrame（必須包含 'target_class' 欄位）

    Returns:
        Tuple[pd.DataFrame, np.ndarray]:
            - df: 原始 DataFrame
            - target_onehot: One-Hot 編碼的目標變數，形狀 (n_samples, 5)

    Example:
        >>> df, target_onehot = create_one_hot_target(df)
        >>> print(target_onehot.shape)
        (6700, 5)
    """
    if "target_class" not in df.columns:
        raise ValueError("DataFrame 必須包含 'target_class' 欄位")

    # 取得類別
    target_classes = df["target_class"].values.astype(int)

    # 建立 One-Hot 編碼
    n_samples = len(target_classes)
    n_classes = 5
    target_onehot = np.zeros((n_samples, n_classes))

    for i, cls in enumerate(target_classes):
        target_onehot[i, cls] = 1

    logging.info(f"One-Hot 編碼完成，形狀: {target_onehot.shape}")

    return df, target_onehot


def get_feature_set_config(feature_set_id: str) -> dict:
    """
    取得特徵集配置資訊

    Args:
        feature_set_id: 特徵集 ID

    Returns:
        dict: 特徵集配置

    Example:
        >>> config = get_feature_set_config("Set A")
        >>> print(config['name'])
        動能型
    """
    from .feature_sets import get_feature_set

    return get_feature_set(feature_set_id)


def prepare_features_and_target(
    df: pd.DataFrame, feature_set_id: str, look_ahead_days: int = 20
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    完整的特徵工程與目標變數準備流程

    Args:
        df: 原始 DataFrame
        feature_set_id: 特徵集 ID
        look_ahead_days: 預測未來幾天

    Returns:
        Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
            - features: 特徵矩陣，形狀 (n_samples, n_features)
            - target_onehot: One-Hot 編碼的目標變數，形狀 (n_samples, 5)
            - df_processed: 處理後的 DataFrame

    Example:
        >>> features, target, df_processed = prepare_features_and_target(df, "Set A")
        >>> print(f"特徵形狀: {features.shape}, 目標形狀: {target.shape}")
    """
    logging.info("=" * 60)
    logging.info("開始特徵工程與目標變數準備")

    # 1. 特徵工程
    df_features = engineer_features(df, feature_set_id)

    # 2. 計算目標變數
    df_with_target = calculate_target_variable(df_features, look_ahead_days)

    # 3. 轉換為 One-Hot 編碼
    df_processed, target_onehot = create_one_hot_target(df_with_target)

    # 4. 提取特徵矩陣
    feature_columns = get_feature_names(feature_set_id)
    features = df_processed[feature_columns].values

    logging.info(f"✅ 特徵與目標準備完成")
    logging.info(f"   特徵形狀: {features.shape}")
    logging.info(f"   目標形狀: {target_onehot.shape}")
    logging.info("=" * 60)

    return features, target_onehot, df_processed


if __name__ == "__main__":
    # 測試特徵工程
    logging.basicConfig(level=logging.INFO)

    from ..data.data_loader import load_csv_data

    try:
        # 載入資料
        df = load_csv_data("19980601-20251111-converted.csv")

        # 測試特徵工程
        features, target, df_processed = prepare_features_and_target(df, "Set A")

        print(f"\n特徵形狀: {features.shape}")
        print(f"目標形狀: {target.shape}")
        print(f"\n前 5 筆目標變數:\n{target[:5]}")

    except Exception as e:
        print(f"錯誤: {str(e)}")
