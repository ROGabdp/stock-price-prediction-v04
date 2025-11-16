"""
模型配置管理模組

負責保存和載入模型訓練時的超參數配置
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime


def save_model_config(
    model_path: str,
    feature_set_id: str,
    time_steps: int,
    additional_params: Optional[Dict] = None,
) -> str:
    """
    保存模型配置到 JSON 文件

    配置文件會保存在與模型相同的目錄,文件名為 {model_name}_config.json

    Args:
        model_path: 模型文件路徑 (例如: models/baseline_model.h5)
        feature_set_id: 特徵集 ID (Set A, Set B, Set C)
        time_steps: 時間窗口大小
        additional_params: 額外的參數字典 (選用)

    Returns:
        str: 配置文件路徑

    Example:
        >>> config_path = save_model_config(
        ...     "models/baseline_model.h5",
        ...     "Set A",
        ...     60,
        ...     {"epochs": 100, "batch_size": 32}
        ... )
    """
    model_path_obj = Path(model_path)

    # 配置文件路徑: 將 .h5 替換為 _config.json
    config_path = model_path_obj.with_suffix("").with_suffix(".json")
    # 如果原文件名已經有 .h5,則保持原樣
    if not str(config_path).endswith("_config.json"):
        config_path = Path(str(model_path_obj.with_suffix("")) + "_config.json")

    # 組裝配置字典
    config = {
        "model_file": str(model_path),
        "feature_set_id": feature_set_id,
        "time_steps": time_steps,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    # 添加額外參數
    if additional_params:
        config.update(additional_params)

    # 保存為 JSON
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    logging.info(f"✅ 模型配置已保存: {config_path}")
    logging.info(f"   - 特徵集: {feature_set_id}")
    logging.info(f"   - 時間窗口: {time_steps}")

    return str(config_path)


def load_model_config(model_path: str) -> Optional[Dict]:
    """
    載入模型配置

    Args:
        model_path: 模型文件路徑

    Returns:
        Dict or None: 配置字典,如果配置文件不存在則返回 None

    Example:
        >>> config = load_model_config("models/baseline_model.h5")
        >>> if config:
        ...     print(f"特徵集: {config['feature_set_id']}")
        ...     print(f"時間窗口: {config['time_steps']}")
    """
    model_path_obj = Path(model_path)

    # 嘗試找到配置文件
    config_path = Path(str(model_path_obj.with_suffix("")) + "_config.json")

    if not config_path.exists():
        logging.warning(f"⚠️ 找不到模型配置文件: {config_path}")
        return None

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        logging.info(f"✅ 模型配置已載入: {config_path}")
        logging.info(f"   - 特徵集: {config.get('feature_set_id', 'N/A')}")
        logging.info(f"   - 時間窗口: {config.get('time_steps', 'N/A')}")

        return config

    except Exception as e:
        logging.error(f"❌ 載入模型配置失敗: {e}")
        return None


def get_config_path(model_path: str) -> str:
    """
    取得模型對應的配置文件路徑

    Args:
        model_path: 模型文件路徑

    Returns:
        str: 配置文件路徑

    Example:
        >>> config_path = get_config_path("models/baseline_model.h5")
        >>> print(config_path)
        models/baseline_model_config.json
    """
    model_path_obj = Path(model_path)
    return str(Path(str(model_path_obj.with_suffix("")) + "_config.json"))


if __name__ == "__main__":
    # 測試模組
    logging.basicConfig(level=logging.INFO)

    print("測試模型配置管理")
    print("=" * 80)

    # 測試保存配置
    test_model_path = "models/test_model.h5"
    config_path = save_model_config(
        test_model_path,
        feature_set_id="Set A",
        time_steps=60,
        additional_params={
            "epochs": 100,
            "batch_size": 32,
            "learning_rate": 0.001,
        },
    )

    print(f"\n配置文件路徑: {config_path}")

    # 測試載入配置
    config = load_model_config(test_model_path)
    if config:
        print("\n載入的配置:")
        for key, value in config.items():
            print(f"  {key}: {value}")

    # 測試取得配置路徑
    expected_path = get_config_path(test_model_path)
    print(f"\n預期配置路徑: {expected_path}")
