"""
3分類聚合模組

此模組提供將5分類機率向量聚合為3分類(看跌/震盪/看漲)的功能
"""

from typing import Dict
import numpy as np


def aggregate_to_3_categories(prob_5class: np.ndarray) -> Dict[str, float]:
    """
    將5分類機率向量聚合為3分類機率字典

    Args:
        prob_5class (np.ndarray): 長度為5的numpy陣列,順序為:
            [0] 極度下跌機率
            [1] 溫和下跌機率
            [2] 區間震盪機率
            [3] 溫和上漲機率
            [4] 極度上漲機率

    Returns:
        Dict[str, float]: 3分類機率字典,鍵為 "看跌", "震盪", "看漲"

    Raises:
        TypeError: 若 prob_5class 不是 numpy.ndarray 型別
        ValueError: 若 prob_5class 形狀不為 (5,)
        AssertionError: 若機率總和不為 1.0 (容忍誤差 1e-10)

    Example:
        >>> prob_5 = np.array([0.05, 0.15, 0.25, 0.35, 0.20])
        >>> result = aggregate_to_3_categories(prob_5)
        >>> print(result)
        {'看跌': 0.20, '震盪': 0.25, '看漲': 0.55}
    """
    # 型別檢查
    if not isinstance(prob_5class, np.ndarray):
        raise TypeError("prob_5class 必須是 numpy.ndarray 型別")

    # 形狀檢查
    if prob_5class.shape != (5,):
        raise ValueError(f"prob_5class 形狀必須為 (5,),實際為 {prob_5class.shape}")

    # 機率總和檢查
    total_5class = np.sum(prob_5class)
    assert (
        abs(total_5class - 1.0) < 1e-6
    ), f"5分類機率總和必須為1.0,實際為 {total_5class}"

    # 聚合計算
    bearish = float(prob_5class[0] + prob_5class[1])  # 極度下跌 + 溫和下跌
    neutral = float(prob_5class[2])  # 震盪
    bullish = float(prob_5class[3] + prob_5class[4])  # 溫和上漲 + 極度上漲

    # 驗證3分類總和
    total_3class = bearish + neutral + bullish
    assert (
        abs(total_3class - 1.0) < 1e-6
    ), f"3分類機率總和必須為1.0,實際為 {total_3class}"

    return {"看跌": bearish, "震盪": neutral, "看漲": bullish}


def get_predicted_class_3(prob_3class: Dict[str, float]) -> str:
    """
    根據3分類機率字典,回傳機率最高的類別

    Args:
        prob_3class (Dict[str, float]): 3分類機率字典,鍵為 "看跌", "震盪", "看漲"

    Returns:
        str: 機率最高的類別名稱 ("看跌", "震盪", "看漲" 之一)

    Raises:
        ValueError: 若字典缺少必要的鍵
        ValueError: 若所有機率值都為 0.0

    Example:
        >>> prob_3 = {"看跌": 0.20, "震盪": 0.25, "看漲": 0.55}
        >>> get_predicted_class_3(prob_3)
        '看漲'
    """
    # 檢查必要鍵
    required_keys = {"看跌", "震盪", "看漲"}
    if not required_keys.issubset(prob_3class.keys()):
        raise ValueError("prob_3class 必須包含 '看跌', '震盪', '看漲' 三個鍵")

    # 檢查是否所有機率都為0
    if all(prob_3class[key] == 0.0 for key in required_keys):
        raise ValueError("所有3分類機率不可同時為0")

    # 找出機率最高的類別
    # 優先級: 看跌 > 震盪 > 看漲 (當機率相同時)
    max_prob = -1.0
    predicted_class = None

    for category in ["看跌", "震盪", "看漲"]:
        if prob_3class[category] > max_prob:
            max_prob = prob_3class[category]
            predicted_class = category

    return predicted_class
