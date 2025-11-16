"""
歷史驗證模組

此模組提供查詢實際市場資料並計算預測準確度的功能
"""

from typing import Dict, Any, Optional
import pandas as pd
import logging

# 從預測模組導入分類映射常數,確保100%一致性
from src.prediction.predictor import CLASS_MAPPING


def calculate_actual_class_5(actual_change_pct: float) -> int:
    """
    根據實際漲跌幅計算5分類類別

    Args:
        actual_change_pct: 實際漲跌幅百分比 (如 0.032 表示 +3.2%)

    Returns:
        int: 5分類類別 (0-4)

    Example:
        >>> calculate_actual_class_5(0.035)  # +3.5%
        3  # 溫和上漲
    """
    # 根據 CLASS_MAPPING 的範圍定義進行分類
    # Class 0: R < -5.0%
    # Class 1: -5.0% ≤ R < -2.5%
    # Class 2: -2.5% ≤ R ≤ +2.5%
    # Class 3: +2.5% < R ≤ +5.0%
    # Class 4: R > +5.0%

    if actual_change_pct < -0.05:  # < -5.0%
        return 0
    elif -0.05 <= actual_change_pct < -0.025:  # -5.0% ≤ R < -2.5%
        return 1
    elif -0.025 <= actual_change_pct <= 0.025:  # -2.5% ≤ R ≤ +2.5%
        return 2
    elif 0.025 < actual_change_pct <= 0.05:  # +2.5% < R ≤ +5.0%
        return 3
    else:  # > +5.0%
        return 4


def map_5class_to_3class(class_5: int) -> str:
    """
    將5分類索引映射為3分類標籤

    Args:
        class_5: 5分類索引 (0-4)

    Returns:
        str: 3分類標籤 ("看跌"/"震盪"/"看漲")

    Raises:
        ValueError: 若 class_5 不在 0-4 範圍

    Example:
        >>> map_5class_to_3class(0)
        '看跌'
        >>> map_5class_to_3class(3)
        '看漲'
    """
    if class_5 not in range(5):
        raise ValueError(f"class_5 必須在 0-4 範圍內,實際為 {class_5}")

    # 映射規則: [0,1]→看跌, [2]→震盪, [3,4]→看漲
    if class_5 in [0, 1]:
        return "看跌"
    elif class_5 == 2:
        return "震盪"
    else:  # class_5 in [3, 4]
        return "看漲"


def get_actual_data(
    df_raw: pd.DataFrame,
    input_date: str,
    prediction_date: str,
    predicted_class_5: int,
    predicted_class_3: str,
) -> Optional[Dict[str, Any]]:
    """
    查詢預測日期的實際市場資料並計算驗證結果

    Args:
        df_raw: 原始歷史資料 DataFrame (包含 'date' 和 'close' 欄位)
        input_date: 輸入日期 "YYYY-MM-DD"
        prediction_date: 預測日期 "YYYY-MM-DD"
        predicted_class_5: 5分類預測類別 (0-4)
        predicted_class_3: 3分類預測類別 ("看跌"/"震盪"/"看漲")

    Returns:
        Optional[Dict]: 若預測日期存在於歷史資料,回傳 ActualDataDict,否則 None

    Example:
        >>> actual = get_actual_data(df, "2024-02-15", "2024-03-15", 3, "看漲")
        >>> print(actual["is_correct_5class"])
        True
    """
    try:
        # 確保 date 欄位為 datetime 型別
        if not pd.api.types.is_datetime64_any_dtype(df_raw["date"]):
            df_raw["date"] = pd.to_datetime(df_raw["date"])

        # 查詢預測日期的實際資料
        prediction_rows = df_raw[df_raw["date"] == pd.to_datetime(prediction_date)]

        if len(prediction_rows) == 0:
            # 預測日期不存在於歷史資料
            logging.debug(f"預測日期 {prediction_date} 不存在於歷史資料中")
            return None

        # 取得實際收盤價
        actual_close = float(prediction_rows.iloc[0]["close"])

        # 查詢輸入日期的收盤價以計算漲跌幅
        input_rows = df_raw[df_raw["date"] == pd.to_datetime(input_date)]
        if len(input_rows) == 0:
            logging.warning(
                f"輸入日期 {input_date} 不存在於歷史資料中,無法計算實際漲跌幅"
            )
            return None

        input_close = float(input_rows.iloc[0]["close"])

        # 計算實際漲跌幅 (百分比形式)
        actual_change_pct = (actual_close - input_close) / input_close

        # 計算實際類別
        actual_class_5 = calculate_actual_class_5(actual_change_pct)
        actual_class_3 = map_5class_to_3class(actual_class_5)

        # 比對預測正確性
        is_correct_5class = predicted_class_5 == actual_class_5
        is_correct_3class = predicted_class_3 == actual_class_3

        # 組裝結果字典
        result = {
            "actual_close": actual_close,
            "actual_change_pct": actual_change_pct,
            "actual_class_5": actual_class_5,
            "actual_class_3": actual_class_3,
            "is_correct_5class": is_correct_5class,
            "is_correct_3class": is_correct_3class,
        }

        logging.debug(
            f"實際資料查詢成功: {prediction_date} 收盤 {actual_close:.2f} ({actual_change_pct:+.2%})"
        )
        return result

    except KeyError as e:
        logging.warning(f"DataFrame 缺少必要欄位: {e}")
        return None
    except Exception as e:
        logging.error(f"查詢實際資料時發生錯誤: {e}")
        return None
