"""
預測模組

實作模型載入、日期預測邏輯、預測結果格式化等功能
"""

import logging
from pathlib import Path
from typing import Tuple, Dict, Optional
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

try:
    from tensorflow import keras
except ImportError:
    keras = None

from src.data.data_loader import load_csv_data
from src.features.feature_engineer import engineer_features, get_feature_names
from src.features.scalers import create_scaler, fit_scaler, transform_features
from src.data.data_splitter import create_sequences


# 類別映射（預測類別 → 漲跌幅區間）
CLASS_MAPPING = {
    0: {"label": "極度下跌", "range": "R < -5.0%", "min_change": None, "max_change": -0.05},
    1: {"label": "溫和下跌", "range": "-5.0% ≤ R < -2.5%", "min_change": -0.05, "max_change": -0.025},
    2: {"label": "區間震盪", "range": "-2.5% ≤ R ≤ +2.5%", "min_change": -0.025, "max_change": 0.025},
    3: {"label": "溫和上漲", "range": "+2.5% < R ≤ +5.0%", "min_change": 0.025, "max_change": 0.05},
    4: {"label": "極度上漲", "range": "R > +5.0%", "min_change": 0.05, "max_change": None},
}


def load_trained_model(model_file: str) -> "keras.Model":
    """
    載入訓練好的 LSTM 模型

    Args:
        model_file: 模型檔案路徑 (.h5 格式)

    Returns:
        keras.Model: 載入的模型

    Raises:
        FileNotFoundError: 模型檔案不存在
        ImportError: TensorFlow/Keras 未安裝

    Example:
        >>> model = load_trained_model("models/best_tuned_model.h5")
    """
    if keras is None:
        raise ImportError("TensorFlow/Keras 未安裝")

    model_path = Path(model_file)
    if not model_path.exists():
        raise FileNotFoundError(f"找不到模型檔案: {model_file}")

    model = keras.models.load_model(model_file)
    logging.info(f"✅ 模型已載入: {model_file}")
    logging.info(f"模型輸入形狀: {model.input_shape}")
    logging.info(f"模型輸出形狀: {model.output_shape}")

    return model


def predict_for_date(
    model: "keras.Model",
    df_raw: pd.DataFrame,
    input_date: str,
    feature_set_id: str = "Set A",
    time_steps: int = 60,
) -> Dict:
    """
    根據指定日期預測未來 20 個交易日的漲跌幅

    此函式會:
    1. 驗證輸入日期是否存在於歷史資料中
    2. 載入輸入日期前 time_steps 天的資料
    3. 進行特徵工程與縮放
    4. 執行模型預測
    5. 計算預測日期（input_date + 20 交易日）
    6. 查詢實際收盤價（若存在）

    Args:
        model: 訓練好的 LSTM 模型
        df_raw: 原始歷史資料 DataFrame
        input_date: 輸入日期（格式: "YYYY-MM-DD" 或 "YYYY/M/D"）
        feature_set_id: 特徵集 ID ("Set A", "Set B", "Set C")
        time_steps: 時間窗口大小（必須與模型訓練時一致）

    Returns:
        Dict: 預測結果字典，包含:
            - input_date: 輸入日期
            - prediction_date: 預測日期
            - probability_vector: 5 類機率向量
            - predicted_class: 預測類別 (0-4)
            - confidence: 預測信心度
            - predicted_price_range: 預測收盤價區間 {"min": float, "max": float}
            - actual_close_price: 實際收盤價（若存在則為 float，否則為 None）
            - timestamp: 預測時間戳記

    Raises:
        ValueError: 輸入日期不存在或資料不足

    Example:
        >>> result = predict_for_date(
        ...     model, df_raw, "2024-01-15",
        ...     feature_set_id="Set B", time_steps=60
        ... )
        >>> print(f"預測類別: {result['predicted_class']}")
        >>> print(f"信心度: {result['confidence']:.2%}")
    """
    # 解析輸入日期
    try:
        input_dt = pd.to_datetime(input_date)
    except Exception as e:
        raise ValueError(f"無效的日期格式: {input_date}，錯誤: {e}")

    # 檢查日期是否存在於歷史資料中
    df_raw["date"] = pd.to_datetime(df_raw["date"])
    if input_dt not in df_raw["date"].values:
        raise ValueError(
            f"輸入日期 {input_date} 不存在於歷史資料中\n"
            f"可用日期範圍: {df_raw['date'].min()} 至 {df_raw['date'].max()}"
        )

    # 找到輸入日期在 DataFrame 中的位置
    input_idx = df_raw[df_raw["date"] == input_dt].index[0]

    # 檢查是否有足夠的歷史資料(包含緩衝)
    buffer_days = 10
    if input_idx < (time_steps + buffer_days):
        raise ValueError(
            f"輸入日期 {input_date} 之前的資料不足 {time_steps + buffer_days} 天\n"
            f"當前位置: {input_idx}，需要: {time_steps + buffer_days}"
        )

    # 載入更多歷史資料以應對特徵工程時的資料損失
    # 特徵工程會損失一些行,所以需要多載入一些資料
    buffer_days = 10  # 額外緩衝天數
    df_history = df_raw.iloc[input_idx - time_steps - buffer_days + 1 : input_idx + 1].copy()

    logging.info(f"載入歷史資料: {len(df_history)} 天 ({df_history['date'].min()} 至 {df_history['date'].max()})")

    # 特徵工程
    df_features = engineer_features(df_history, feature_set_id=feature_set_id)

    # 提取特徵矩陣
    feature_columns = get_feature_names(feature_set_id)
    X_features = df_features[feature_columns].values

    logging.info(f"特徵工程後資料長度: {len(X_features)}")

    # 只取最後 time_steps 行（確保至少有 time_steps 行）
    if len(X_features) < time_steps:
        raise ValueError(
            f"特徵工程後資料不足: {len(X_features)} < {time_steps}\n"
            f"請選擇更晚的日期或減少 time_steps"
        )

    X_features = X_features[-time_steps:]  # 取最後 time_steps 行

    # 特徵縮放
    scaler = create_scaler()
    scaler = fit_scaler(scaler, X_features)
    X_scaled = transform_features(scaler, X_features)

    # 直接使用縮放後的特徵作為輸入（不需要 create_sequences）
    # 因為我們已經有正確長度的序列了
    X_input = X_scaled.reshape(1, time_steps, -1)  # 形狀: (1, time_steps, n_features)

    logging.info(f"輸入序列形狀: {X_input.shape}")

    # 執行預測
    y_pred_prob = model.predict(X_input, verbose=0)  # 形狀: (1, 5)

    # 提取預測結果
    probability_vector = y_pred_prob[0]  # 形狀: (5,)
    predicted_class = int(np.argmax(probability_vector))
    confidence = float(np.max(probability_vector))

    logging.info(f"預測完成:")
    logging.info(f"  - 預測類別: {predicted_class} ({CLASS_MAPPING[predicted_class]['label']})")
    logging.info(f"  - 信心度: {confidence:.2%}")
    logging.info(f"  - 機率向量: {probability_vector}")

    # 計算預測日期（輸入日期 + 20 個交易日）
    # 找到輸入日期後的第 20 個交易日
    future_dates = df_raw[df_raw["date"] > input_dt]["date"]
    if len(future_dates) >= 20:
        prediction_dt = future_dates.iloc[19]  # 第 20 個交易日（index 從 0 開始）
        prediction_date_str = prediction_dt.strftime("%Y-%m-%d")
    else:
        # 若資料不足 20 個交易日，則估算日期（假設每週 5 個交易日）
        estimated_days = 20 * 7 // 5  # 約 28 天
        prediction_dt = input_dt + timedelta(days=estimated_days)
        prediction_date_str = prediction_dt.strftime("%Y-%m-%d") + " (估算)"
        logging.warning(f"歷史資料不足 20 個交易日，預測日期為估算值")

    # 查詢輸入日期的收盤價（用於計算預測價格區間）
    input_close_price = float(df_raw.loc[input_idx, "close"])

    # 計算預測收盤價區間
    class_info = CLASS_MAPPING[predicted_class]
    if class_info["min_change"] is None:
        # 極度下跌（R < -5%）
        predicted_min_price = 0
        predicted_max_price = input_close_price * (1 + class_info["max_change"])
    elif class_info["max_change"] is None:
        # 極度上漲（R > 5%）
        predicted_min_price = input_close_price * (1 + class_info["min_change"])
        predicted_max_price = input_close_price * 2  # 假設上限為 100% 漲幅
    else:
        # 其他類別
        predicted_min_price = input_close_price * (1 + class_info["min_change"])
        predicted_max_price = input_close_price * (1 + class_info["max_change"])

    # 查詢實際收盤價（若存在）
    actual_close_price = None
    if prediction_dt in df_raw["date"].values:
        actual_idx = df_raw[df_raw["date"] == prediction_dt].index[0]
        actual_close_price = float(df_raw.loc[actual_idx, "close"])
        logging.info(f"實際收盤價: {actual_close_price:.2f} 元")
    else:
        logging.warning(f"預測日期 {prediction_date_str} 的實際資料不存在")

    # 組裝預測結果
    result = {
        "input_date": input_dt.strftime("%Y-%m-%d"),
        "input_close_price": input_close_price,
        "prediction_date": prediction_date_str,
        "probability_vector": probability_vector.tolist(),
        "predicted_class": predicted_class,
        "predicted_class_label": class_info["label"],
        "predicted_class_range": class_info["range"],
        "confidence": confidence,
        "predicted_price_range": {
            "min": predicted_min_price,
            "max": predicted_max_price,
        },
        "actual_close_price": actual_close_price,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    return result


def format_prediction_result(result: Dict) -> str:
    """
    格式化預測結果為易讀的文字輸出

    Args:
        result: predict_for_date 回傳的預測結果字典

    Returns:
        str: 格式化的預測結果文字

    Example:
        >>> formatted = format_prediction_result(result)
        >>> print(formatted)
    """
    lines = []
    lines.append("=" * 80)
    lines.append("台股 20 日預測結果")
    lines.append("=" * 80)
    lines.append(f"輸入日期: {result['input_date']}")
    lines.append(f"輸入收盤價: {result['input_close_price']:.2f} 元")
    lines.append(f"預測日期: {result['prediction_date']} (20 個交易日後)")
    lines.append("")

    lines.append("預測結果:")
    lines.append(f"  類別: {result['predicted_class']} ({result['predicted_class_label']})")
    lines.append(f"  漲跌幅區間: {result['predicted_class_range']}")
    lines.append(f"  信心度: {result['confidence']:.2%}")
    lines.append(f"  預測收盤價區間: {result['predicted_price_range']['min']:.2f} - {result['predicted_price_range']['max']:.2f} 元")
    lines.append("")

    # 實際收盤價比較
    if result["actual_close_price"] is not None:
        actual_price = result["actual_close_price"]
        min_price = result["predicted_price_range"]["min"]
        max_price = result["predicted_price_range"]["max"]

        in_range = min_price <= actual_price <= max_price
        status = "✅ 落在預測區間內" if in_range else "❌ 超出預測區間"

        lines.append(f"實際收盤價: {actual_price:.2f} 元 {status}")

        # 計算實際漲跌幅
        actual_change = (actual_price - result["input_close_price"]) / result["input_close_price"]
        lines.append(f"實際漲跌幅: {actual_change:+.2%}")
    else:
        lines.append("實際收盤價: 資料尚未公布")

    lines.append("")
    lines.append("機率分佈:")
    for i, (class_id, class_info) in enumerate(CLASS_MAPPING.items()):
        prob = result["probability_vector"][i]
        bar_length = int(prob * 50)  # 最大 50 個字元
        bar = "█" * bar_length
        marker = " ← 預測" if i == result["predicted_class"] else ""
        lines.append(f"  {class_info['label']} ({class_info['range']}): {prob:.1%} {bar}{marker}")

    lines.append("")
    lines.append(f"預測時間: {result['timestamp']}")
    lines.append("=" * 80)

    return "\n".join(lines)


if __name__ == "__main__":
    # 測試預測模組
    logging.basicConfig(level=logging.INFO)

    print("測試預測模組")
    print("=" * 80)

    # 測試類別映射
    print("\n類別映射:")
    for class_id, class_info in CLASS_MAPPING.items():
        print(f"  {class_id}: {class_info['label']} ({class_info['range']})")

    # 測試格式化（使用範例結果）
    example_result = {
        "input_date": "2024-01-15",
        "input_close_price": 15800.0,
        "prediction_date": "2024-02-14",
        "probability_vector": [0.023, 0.087, 0.124, 0.685, 0.081],
        "predicted_class": 3,
        "predicted_class_label": "溫和上漲",
        "predicted_class_range": "+2.5% < R ≤ +5.0%",
        "confidence": 0.685,
        "predicted_price_range": {"min": 16195.0, "max": 16590.0},
        "actual_close_price": 16120.0,
        "timestamp": "2025-11-15 14:35:21",
    }

    print("\n格式化範例:")
    formatted = format_prediction_result(example_result)
    print(formatted)
