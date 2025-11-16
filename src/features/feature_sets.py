"""
特徵集定義配置

定義三種特徵集（Set A: 動能型, Set B: 震盪型, Set C: 全特徵集）
"""

from typing import List, Dict


# 特徵集定義
FEATURE_SETS: Dict[str, Dict] = {
    "Set A": {
        "name": "動能型",
        "description": "專注於價格動能與法人籌碼的特徵集",
        "features": [
            # 價格變化率 (4 個)
            "open_return",
            "high_return",
            "low_return",
            "close_return",
            # Volume 變化率 (1 個)
            "volume_change",
            # SMA 比例與斜率 (2 個)
            "sma20_sma60_ratio",
            "sma20_slope",
            # MACD 指標 (2 個)
            "DIF12-26",
            "MACD9",
            # 法人籌碼 (3 個) - 使用原始欄位名稱
            "net buy sell",
            "cumulative net buy sell",
            "net_buy_sell_change",
        ],
        "exclude": [
            "K(9,3)",
            "D(9,3)",
            "OSC",
        ],
    },
    "Set B": {
        "name": "震盪型",
        "description": "專注於震盪指標與法人籌碼的特徵集",
        "features": [
            # 價格變化率 (4 個)
            "open_return",
            "high_return",
            "low_return",
            "close_return",
            # Volume 變化率 (1 個)
            "volume_change",
            # SMA 比例與斜率 (2 個)
            "sma20_sma60_ratio",
            "sma20_slope",
            # KD 指標 (2 個)
            "K(9,3)",
            "D(9,3)",
            # 法人籌碼 (3 個) - 使用原始欄位名稱
            "net buy sell",
            "cumulative net buy sell",
            "net_buy_sell_change",
        ],
        "exclude": [
            "DIF12-26",
            "MACD9",
        ],
    },
    "Set C": {
        "name": "全特徵集",
        "description": "包含所有處理過的特徵",
        "features": [
            # 價格變化率 (4 個)
            "open_return",
            "high_return",
            "low_return",
            "close_return",
            # Volume 變化率 (1 個)
            "volume_change",
            # SMA 比例與斜率 (2 個)
            "sma20_sma60_ratio",
            "sma20_slope",
            # MACD 指標 (3 個)
            "DIF12-26",
            "MACD9",
            "OSC",
            # KD 指標 (2 個)
            "K(9,3)",
            "D(9,3)",
            # 法人籌碼 (3 個) - 使用原始欄位名稱
            "net buy sell",
            "cumulative net buy sell",
            "net_buy_sell_change",
        ],
        "exclude": [],
    },
}


def get_feature_set(feature_set_id: str) -> Dict:
    """
    取得指定特徵集的配置

    Args:
        feature_set_id: 特徵集 ID ("Set A", "Set B", "Set C")

    Returns:
        Dict: 特徵集配置

    Raises:
        ValueError: 特徵集 ID 不存在

    Example:
        >>> config = get_feature_set("Set A")
        >>> print(config['name'])
        動能型
    """
    if feature_set_id not in FEATURE_SETS:
        raise ValueError(
            f"未知的特徵集 ID: {feature_set_id}，"
            f"可用選項: {list(FEATURE_SETS.keys())}"
        )

    return FEATURE_SETS[feature_set_id]


def get_feature_names(feature_set_id: str) -> List[str]:
    """
    取得特徵集的特徵名稱清單

    Args:
        feature_set_id: 特徵集 ID

    Returns:
        List[str]: 特徵名稱清單

    Example:
        >>> features = get_feature_names("Set A")
        >>> print(f"Set A 包含 {len(features)} 個特徵")
        Set A 包含 12 個特徵
    """
    config = get_feature_set(feature_set_id)
    return config["features"]


def list_all_feature_sets() -> List[str]:
    """
    列出所有可用的特徵集 ID

    Returns:
        List[str]: 特徵集 ID 清單

    Example:
        >>> print(list_all_feature_sets())
        ['Set A', 'Set B', 'Set C']
    """
    return list(FEATURE_SETS.keys())


def get_feature_set_info(feature_set_id: str) -> str:
    """
    取得特徵集的詳細資訊（格式化字串）

    Args:
        feature_set_id: 特徵集 ID

    Returns:
        str: 格式化的特徵集資訊

    Example:
        >>> print(get_feature_set_info("Set A"))
        Set A (動能型)
        描述: 專注於價格動能與法人籌碼的特徵集
        特徵數量: 12
        ...
    """
    config = get_feature_set(feature_set_id)

    info = f"{feature_set_id} ({config['name']})\n"
    info += f"描述: {config['description']}\n"
    info += f"特徵數量: {len(config['features'])}\n"
    info += f"特徵清單:\n"
    for i, feature in enumerate(config["features"], 1):
        info += f"  {i}. {feature}\n"

    if config["exclude"]:
        info += f"排除特徵: {', '.join(config['exclude'])}\n"

    return info


if __name__ == "__main__":
    # 測試特徵集定義
    print("所有可用的特徵集:")
    for fs_id in list_all_feature_sets():
        print(f"\n{get_feature_set_info(fs_id)}")

    # 測試特徵名稱提取
    print("\n" + "=" * 60)
    print("Set A 特徵清單:")
    for feature in get_feature_names("Set A"):
        print(f"  - {feature}")
