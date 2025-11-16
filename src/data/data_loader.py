"""
資料載入模組

負責從 CSV 檔案載入台股歷史資料，並進行資料驗證與清理。
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import numpy as np


# 定義必要的欄位
REQUIRED_COLUMNS = [
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "SMA5",
    "SMA10",
    "SMA20",
    "SMA60",
    "SMA120",
    "SMA240",
    "MA5",
    "MA10",
    "DIF12-26",
    "MACD9",
    "OSC",
    "K(9,3)",
    "D(9,3)",
    "net buy sell",
    "cumulative net buy sell",
    "buy",
    "sell",
]


def load_csv_data(file_path: str) -> pd.DataFrame:
    """
    載入 CSV 格式的台股歷史資料

    Args:
        file_path: CSV 檔案路徑

    Returns:
        pd.DataFrame: 載入的資料 DataFrame

    Raises:
        FileNotFoundError: 檔案不存在
        ValueError: 資料格式錯誤或驗證失敗

    Example:
        >>> df = load_csv_data("19980601-20251111-converted.csv")
        >>> print(f"載入 {len(df)} 筆交易資料")
        載入 6823 筆交易資料
    """
    # 檢查檔案是否存在
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"找不到資料檔案: {file_path}")

    logging.info(f"正在載入資料: {file_path}")

    try:
        # 載入 CSV
        df = pd.read_csv(file_path)

        logging.info(f"✅ 成功載入 {len(df)} 筆原始資料")

        # 驗證資料格式
        validate_schema(df)

        # 資料清理與預處理
        df = preprocess_data(df)

        logging.info(f"✅ 資料預處理完成，剩餘 {len(df)} 筆有效資料")

        return df

    except Exception as e:
        logging.error(f"❌ 資料載入失敗: {str(e)}")
        raise


def validate_schema(df: pd.DataFrame) -> None:
    """
    驗證 DataFrame 的欄位結構

    Args:
        df: 待驗證的 DataFrame

    Raises:
        ValueError: 欄位驗證失敗

    Example:
        >>> validate_schema(df)  # 若驗證通過，不會拋出例外
    """
    # 檢查必要欄位
    missing_columns = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing_columns:
        raise ValueError(f"缺少必要欄位: {missing_columns}")

    logging.info(f"✅ 欄位驗證通過（共 {len(df.columns)} 個欄位）")

    # 檢查資料型別
    numeric_columns = [col for col in REQUIRED_COLUMNS if col != "date"]
    for col in numeric_columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            logging.warning(f"⚠️ 欄位 '{col}' 不是數值型別，嘗試轉換")
            try:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            except Exception as e:
                raise ValueError(f"欄位 '{col}' 無法轉換為數值: {str(e)}")


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    預處理資料：處理日期、缺失值、異常值

    Args:
        df: 原始 DataFrame

    Returns:
        pd.DataFrame: 預處理後的 DataFrame

    Example:
        >>> df_clean = preprocess_data(df_raw)
    """
    df = df.copy()

    # 1. 處理日期欄位
    try:
        df["date"] = pd.to_datetime(df["date"])
    except Exception as e:
        raise ValueError(f"日期欄位轉換失敗: {str(e)}")

    # 2. 按日期排序
    df = df.sort_values("date").reset_index(drop=True)
    logging.info(f"資料日期範圍: {df['date'].min()} 至 {df['date'].max()}")

    # 3. 檢查缺失值
    missing_counts = df[REQUIRED_COLUMNS].isnull().sum()
    if missing_counts.any():
        logging.warning(f"⚠️ 發現缺失值:\n{missing_counts[missing_counts > 0]}")

        # 移除包含缺失值的列
        df = df.dropna(subset=REQUIRED_COLUMNS)
        logging.info(f"已移除包含缺失值的列，剩餘 {len(df)} 筆")

    # 4. 驗證數值範圍
    if (df["open"] <= 0).any() or (df["close"] <= 0).any():
        logging.warning("⚠️ 發現非正數的價格資料，將移除")
        df = df[(df["open"] > 0) & (df["close"] > 0)]

    if (df["high"] < df["low"]).any():
        logging.warning("⚠️ 發現 high < low 的異常資料，將移除")
        df = df[df["high"] >= df["low"]]

    if (df["volume"] < 0).any():
        logging.warning("⚠️ 發現負數成交量，將移除")
        df = df[df["volume"] >= 0]

    # 5. 驗證 KD 指標範圍 (應在 0-1 之間)
    kd_valid = (df["K(9,3)"] >= 0) & (df["K(9,3)"] <= 1) & (df["D(9,3)"] >= 0) & (df["D(9,3)"] <= 1)
    if not kd_valid.all():
        invalid_count = (~kd_valid).sum()
        logging.warning(f"⚠️ 發現 {invalid_count} 筆 KD 指標超出範圍 [0, 1]，將移除")
        df = df[kd_valid]

    return df


def get_data_info(df: pd.DataFrame) -> dict:
    """
    取得資料集的統計資訊

    Args:
        df: DataFrame

    Returns:
        dict: 資料統計資訊

    Example:
        >>> info = get_data_info(df)
        >>> print(f"資料筆數: {info['count']}")
    """
    return {
        "count": len(df),
        "date_range": {
            "start": df["date"].min(),
            "end": df["date"].max(),
        },
        "columns": list(df.columns),
        "missing_values": df.isnull().sum().to_dict(),
        "numeric_summary": df.describe().to_dict(),
    }


def split_by_date(
    df: pd.DataFrame, split_date: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    根據日期分割資料集

    Args:
        df: DataFrame
        split_date: 分割日期 (字串格式，如 "2024-01-01")

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (分割前的資料, 分割後的資料)

    Example:
        >>> before, after = split_by_date(df, "2024-01-01")
    """
    split_datetime = pd.to_datetime(split_date)
    before = df[df["date"] < split_datetime].copy()
    after = df[df["date"] >= split_datetime].copy()

    logging.info(f"以 {split_date} 分割: 前 {len(before)} 筆, 後 {len(after)} 筆")

    return before, after


if __name__ == "__main__":
    # 測試資料載入
    logging.basicConfig(level=logging.INFO)

    try:
        df = load_csv_data("19980601-20251111-converted.csv")
        print(f"\n載入成功: {len(df)} 筆資料")
        print(f"日期範圍: {df['date'].min()} ~ {df['date'].max()}")
        print(f"\n前 5 筆資料:\n{df.head()}")

        # 顯示資料資訊
        info = get_data_info(df)
        print(f"\n資料統計: {info['count']} 筆")

    except Exception as e:
        print(f"錯誤: {str(e)}")
