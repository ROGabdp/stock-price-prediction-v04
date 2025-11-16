"""
GPU 可用性檢查與配置模組

此模組負責檢測 TensorFlow GPU 可用性，並配置 GPU 記憶體管理策略。
若 GPU 不可用，系統將自動降級使用 CPU 訓練並記錄警告訊息。
"""

import logging
from typing import Tuple

try:
    import tensorflow as tf
except ImportError:
    tf = None


def setup_gpu() -> Tuple[bool, str]:
    """
    設置 GPU 環境並檢查可用性

    Returns:
        Tuple[bool, str]: (GPU 是否可用, 狀態訊息)

    Example:
        >>> gpu_available, message = setup_gpu()
        >>> print(message)
        ✅ GPU 可用: 1 個 GPU
    """
    if tf is None:
        message = "⚠️ TensorFlow 未安裝，無法使用 GPU"
        logging.warning(message)
        return False, message

    try:
        gpus = tf.config.list_physical_devices("GPU")

        if gpus:
            try:
                # 允許 GPU 記憶體動態增長，避免佔用所有 VRAM
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)

                # 記錄 GPU 資訊
                gpu_names = [gpu.name for gpu in gpus]
                message = f"✅ GPU 可用: {len(gpus)} 個 GPU ({', '.join(gpu_names)})"
                logging.info(message)
                return True, message

            except RuntimeError as e:
                message = f"⚠️ GPU 配置失敗: {str(e)}，將使用 CPU 訓練"
                logging.warning(message)
                return False, message
        else:
            message = "⚠️ 未偵測到 GPU，使用 CPU 訓練（訓練速度會較慢）"
            logging.warning(message)
            return False, message

    except Exception as e:
        message = f"⚠️ GPU 檢查時發生錯誤: {str(e)}，將使用 CPU 訓練"
        logging.error(message)
        return False, message


def get_gpu_info() -> dict:
    """
    取得 GPU 詳細資訊

    Returns:
        dict: GPU 資訊字典，包含數量、名稱、記憶體資訊等

    Example:
        >>> info = get_gpu_info()
        >>> print(f"GPU 數量: {info['count']}")
        GPU 數量: 1
    """
    if tf is None:
        return {"count": 0, "available": False, "error": "TensorFlow 未安裝"}

    try:
        gpus = tf.config.list_physical_devices("GPU")
        if not gpus:
            return {"count": 0, "available": False}

        gpu_info = {
            "count": len(gpus),
            "available": True,
            "devices": [],
        }

        for i, gpu in enumerate(gpus):
            device_info = {
                "id": i,
                "name": gpu.name,
                "type": gpu.device_type,
            }
            gpu_info["devices"].append(device_info)

        return gpu_info

    except Exception as e:
        return {"count": 0, "available": False, "error": str(e)}


def check_gpu_memory_usage() -> dict:
    """
    檢查 GPU 記憶體使用狀況（僅在訓練時有效）

    Returns:
        dict: 記憶體使用資訊

    Note:
        此函式需在模型訓練過程中呼叫才能取得有效的記憶體使用資訊
    """
    if tf is None:
        return {"error": "TensorFlow 未安裝"}

    try:
        gpus = tf.config.list_physical_devices("GPU")
        if not gpus:
            return {"available": False}

        # TensorFlow 的記憶體統計需在訓練過程中才準確
        # 這裡僅提供基本檢查
        return {
            "available": True,
            "gpu_count": len(gpus),
            "note": "詳細記憶體使用需在訓練時監控",
        }

    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    # 測試 GPU 設置
    logging.basicConfig(level=logging.INFO)
    gpu_available, message = setup_gpu()
    print(message)

    # 顯示 GPU 詳細資訊
    info = get_gpu_info()
    print(f"\nGPU 詳細資訊: {info}")
