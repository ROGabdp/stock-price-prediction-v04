"""
特徵集選擇邏輯

根據超參數調整過程中的 hp.Choice 動態載入不同的特徵集（Set A/B/C）
"""

import logging
from typing import Tuple
import pandas as pd
import numpy as np

from src.features.feature_sets import (
    get_feature_set,
    get_feature_names,
    list_all_feature_sets,
)


def select_feature_set(
    df: pd.DataFrame, feature_set_id: str
) -> Tuple[pd.DataFrame, list]:
    """
    根據特徵集 ID 選擇對應的特徵欄位

    Args:
        df: 包含所有特徵的 DataFrame
        feature_set_id: 特徵集 ID ("Set A", "Set B", "Set C")

    Returns:
        Tuple[pd.DataFrame, list]: (選中的特徵 DataFrame, 特徵名稱清單)

    Raises:
        ValueError: 特徵集 ID 不存在或特徵欄位缺失

    Example:
        >>> df_features, feature_names = select_feature_set(df, "Set A")
        >>> print(f"選擇了 {len(feature_names)} 個特徵")
        選擇了 12 個特徵
    """
    # 取得特徵集配置
    config = get_feature_set(feature_set_id)
    feature_names = config["features"]

    # 檢查所需特徵是否存在於 DataFrame
    missing_features = [f for f in feature_names if f not in df.columns]
    if missing_features:
        raise ValueError(
            f"特徵集 {feature_set_id} 所需的特徵在 DataFrame 中缺失: "
            f"{missing_features}\n"
            f"可用欄位: {df.columns.tolist()}"
        )

    # 選擇特徵
    df_selected = df[feature_names].copy()

    logging.info(f"特徵集選擇完成:")
    logging.info(f"  - 特徵集: {feature_set_id} ({config['name']})")
    logging.info(f"  - 特徵數量: {len(feature_names)}")
    logging.info(f"  - 特徵形狀: {df_selected.shape}")

    return df_selected, feature_names


def get_feature_set_for_tuning(feature_set_id: str) -> dict:
    """
    取得用於超參數調整的特徵集資訊

    此函式提供完整的特徵集配置，供 Keras Tuner 在試驗中使用

    Args:
        feature_set_id: 特徵集 ID

    Returns:
        dict: 包含特徵集資訊的字典
            {
                'id': 特徵集 ID,
                'name': 特徵集名稱,
                'features': 特徵清單,
                'n_features': 特徵數量
            }

    Example:
        >>> info = get_feature_set_for_tuning("Set B")
        >>> print(f"{info['name']}: {info['n_features']} 個特徵")
        震盪型: 12 個特徵
    """
    config = get_feature_set(feature_set_id)

    return {
        "id": feature_set_id,
        "name": config["name"],
        "description": config["description"],
        "features": config["features"],
        "n_features": len(config["features"]),
    }


def validate_feature_set_compatibility(
    df: pd.DataFrame, feature_set_ids: list = None
) -> dict:
    """
    驗證 DataFrame 與多個特徵集的相容性

    檢查 DataFrame 是否包含所有特徵集所需的欄位

    Args:
        df: 要驗證的 DataFrame
        feature_set_ids: 要檢查的特徵集 ID 清單（預設檢查全部）

    Returns:
        dict: 驗證結果字典
            {
                'Set A': {'compatible': True, 'missing': []},
                'Set B': {'compatible': False, 'missing': ['K(9,3)']},
                ...
            }

    Example:
        >>> results = validate_feature_set_compatibility(df)
        >>> for fs_id, result in results.items():
        ...     if result['compatible']:
        ...         print(f"{fs_id}: ✅ 相容")
    """
    if feature_set_ids is None:
        feature_set_ids = list_all_feature_sets()

    results = {}

    for fs_id in feature_set_ids:
        feature_names = get_feature_names(fs_id)
        missing = [f for f in feature_names if f not in df.columns]

        results[fs_id] = {
            "compatible": len(missing) == 0,
            "missing": missing,
            "required_features": feature_names,
            "n_features": len(feature_names),
        }

    return results


def get_n_features_for_feature_set(feature_set_id: str) -> int:
    """
    取得特徵集的特徵數量

    此函式用於在建立模型前確定輸入維度

    Args:
        feature_set_id: 特徵集 ID

    Returns:
        int: 特徵數量

    Example:
        >>> n_features = get_n_features_for_feature_set("Set C")
        >>> print(f"Set C 有 {n_features} 個特徵")
        Set C 有 16 個特徵
    """
    feature_names = get_feature_names(feature_set_id)
    return len(feature_names)


if __name__ == "__main__":
    # 測試特徵集選擇邏輯
    logging.basicConfig(level=logging.INFO)

    # 建立測試 DataFrame
    test_data = {}
    for fs_id in list_all_feature_sets():
        features = get_feature_names(fs_id)
        for feature in features:
            if feature not in test_data:
                test_data[feature] = np.random.randn(100)

    df_test = pd.DataFrame(test_data)

    print("測試 DataFrame 建立完成")
    print(f"形狀: {df_test.shape}")
    print(f"欄位: {df_test.columns.tolist()}\n")

    # 測試每個特徵集
    for fs_id in list_all_feature_sets():
        print(f"\n{'=' * 60}")
        print(f"測試特徵集: {fs_id}")
        print("=" * 60)

        try:
            df_selected, feature_names = select_feature_set(df_test, fs_id)
            print(f"✅ 成功選擇 {len(feature_names)} 個特徵")
            print(f"形狀: {df_selected.shape}")

            # 測試取得特徵集資訊
            info = get_feature_set_for_tuning(fs_id)
            print(f"特徵集名稱: {info['name']}")
            print(f"特徵數量: {info['n_features']}")

        except Exception as e:
            print(f"❌ 錯誤: {e}")

    # 測試相容性驗證
    print(f"\n{'=' * 60}")
    print("相容性驗證")
    print("=" * 60)

    validation_results = validate_feature_set_compatibility(df_test)
    for fs_id, result in validation_results.items():
        status = "✅ 相容" if result["compatible"] else "❌ 不相容"
        print(f"{fs_id}: {status}")
        if not result["compatible"]:
            print(f"  缺失特徵: {result['missing']}")
