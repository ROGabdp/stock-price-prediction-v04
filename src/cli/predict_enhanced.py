"""
增強版股價預測 CLI

此CLI工具提供3分類聚合、5分類詳細視圖、以及歷史驗證功能
"""

import argparse
import sys
import os
import time
import logging
from typing import List, Dict, Any, Optional
import numpy as np

# 加入專案根目錄至 sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# 導入現有模組
from src.prediction.predictor import load_trained_model, predict_for_date
from src.data.data_loader import load_csv_data
from src.utils.model_config import load_model_config

# 導入新增的視覺化模組
from src.visualization.aggregator import (
    aggregate_to_3_categories,
    get_predicted_class_3,
)
from src.visualization.validator import get_actual_data
from src.visualization.plotter import plot_dual_view, get_default_config


def parse_arguments() -> argparse.Namespace:
    """
    解析命令列參數

    Returns:
        argparse.Namespace: 解析後的參數
    """
    parser = argparse.ArgumentParser(
        description="增強版股價預測工具 - 提供3分類聚合、5分類詳細視圖、以及歷史驗證功能",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用範例:
  # 基本用法: 預測單一日期
  python src/cli/predict_enhanced.py \\
      --model-file models/best_tuned_model_20251116_153022.h5 \\
      --data-file 19980601-20251111-converted.csv \\
      --input-date "2024-01-15"

  # 批次預測多個日期
  python src/cli/predict_enhanced.py \\
      -m models/best_tuned_model_20251116_153022.h5 \\
      -d 19980601-20251111-converted.csv \\
      -i "2024-01-15" "2024-02-20" "2024-03-10"

  # 僅文字輸出 (不生成圖片)
  python src/cli/predict_enhanced.py \\
      -m models/best_tuned_model_20251116_153022.h5 \\
      -d 19980601-20251111-converted.csv \\
      -i "2024-01-15" \\
      --no-plot
        """,
    )

    parser.add_argument(
        "--model-file", "-m", type=str, required=True, help="訓練好的模型檔案路徑 (.h5)"
    )

    parser.add_argument(
        "--data-file", "-d", type=str, required=True, help="歷史資料 CSV 檔案路徑"
    )

    parser.add_argument(
        "--input-date",
        "-i",
        type=str,
        nargs="+",
        required=True,
        help="輸入日期 (格式: YYYY-MM-DD,可指定多個)",
    )

    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="outputs",
        help="視覺化圖片輸出目錄 (預設: outputs)",
    )

    parser.add_argument(
        "--feature-set",
        "-f",
        type=str,
        default=None,
        help="特徵集 ID (Set A/B/C,預設從模型配置載入)",
    )

    parser.add_argument(
        "--time-steps",
        "-t",
        type=int,
        default=None,
        help="時間窗口大小 (20/40/60/80,預設從模型配置載入)",
    )

    parser.add_argument(
        "--no-plot", action="store_true", help="僅輸出文字結果,不生成視覺化圖片"
    )

    parser.add_argument("--verbose", "-v", action="store_true", help="顯示詳細日誌")

    return parser.parse_args()


def print_section_header(title: str, width: int = 80) -> None:
    """印出區段標題"""
    print("=" * width)
    print(title.center(width))
    print("=" * width)


def print_subsection_header(title: str, width: int = 80) -> None:
    """印出子區段標題"""
    print("-" * width)
    print(title)
    print("-" * width)


def print_3class_results(
    prob_3class: Dict[str, float], predicted_class_3: str, confidence: float
) -> None:
    """
    印出3分類文字結果

    Args:
        prob_3class: 3分類機率字典
        predicted_class_3: 預測類別
        confidence: 預測信心度 (針對3分類)
    """
    print("\n【3分類聚合預測】")

    for category in ["看跌", "震盪", "看漲"]:
        prob = prob_3class[category]
        bar_length = int(prob * 40)  # 最大40個字元
        bar = "█" * bar_length
        marker = (
            f"← (預測類別,信心度 {confidence*100:.1f}%)"
            if category == predicted_class_3
            else ""
        )
        print(f"  {category}: {prob*100:5.1f}% {bar}{marker}")


def print_5class_results(
    prob_5class: np.ndarray, predicted_class_5: int, class_labels: List[str]
) -> None:
    """
    印出5分類文字結果

    Args:
        prob_5class: 5分類機率向量
        predicted_class_5: 預測類別索引
        class_labels: 類別標籤列表
    """
    print("\n【5分類詳細預測】")

    for i, (prob, label) in enumerate(zip(prob_5class, class_labels)):
        bar_length = int(prob * 40)
        bar = "█" * bar_length
        marker = (
            f"← (預測類別,信心度 {prob*100:.1f}%)" if i == predicted_class_5 else ""
        )
        print(f"  {label:8s}: {prob*100:5.1f}% {bar}{marker}")


def print_actual_results(
    actual_data: Optional[Dict[str, Any]], prediction_date: str
) -> None:
    """
    印出實際結果

    Args:
        actual_data: 實際資料字典 (若存在)
        prediction_date: 預測日期
    """
    print("\n【實際結果】")

    if actual_data is None:
        print(f"  預測日期: {prediction_date} (未來日期或資料不存在)")
        print("  狀態: 待驗證 ⏳")
        print("  說明: 預測日期尚未發生或不在歷史資料中,無法進行驗證")
    else:
        actual_close = actual_data["actual_close"]
        actual_change_pct = actual_data["actual_change_pct"]
        actual_class_5_label = [
            "極度下跌",
            "溫和下跌",
            "區間震盪",
            "溫和上漲",
            "極度上漲",
        ][actual_data["actual_class_5"]]
        actual_class_3 = actual_data["actual_class_3"]
        is_correct_5 = actual_data["is_correct_5class"]
        is_correct_3 = actual_data["is_correct_3class"]

        print(f"  實際收盤價: {actual_close:,.0f} ({actual_change_pct:+.2%})")
        print(f"  實際類別 (5分類): {actual_class_5_label}")
        print(f"  實際類別 (3分類): {actual_class_3}")
        print(
            f"  {'✓' if is_correct_5 else '✗'} 5分類預測: {'正確' if is_correct_5 else '錯誤'}"
        )
        print(
            f"  {'✓' if is_correct_3 else '✗'} 3分類預測: {'正確' if is_correct_3 else '錯誤'}"
        )


def process_single_prediction(
    model,
    df_raw,
    feature_set_id: str,
    time_steps: int,
    input_date: str,
    output_dir: str,
    no_plot: bool,
    config: Dict[str, Any],
    index: int,
    total: int,
) -> Dict[str, Any]:
    """
    處理單一日期的預測

    Args:
        model: 載入的模型
        df_raw: 原始資料 DataFrame
        feature_set_id: 特徵集 ID
        time_steps: 時間窗口大小
        input_date: 輸入日期
        output_dir: 輸出目錄
        no_plot: 是否跳過繪圖
        config: 視覺化配置
        index: 當前索引 (批次預測用)
        total: 總數 (批次預測用)

    Returns:
        Dict: 預測結果統計
    """
    from src.prediction.predictor import CLASS_MAPPING

    print(f"\n預測 {index}/{total}: {input_date}", end="")

    # 執行預測 (呼叫現有的 predict_for_date)
    try:
        result = predict_for_date(
            model, df_raw, input_date, feature_set_id, time_steps
        )
    except Exception as e:
        print(f" → 預測失敗: {e}")
        return {"status": "failed", "error": str(e)}

    # 取得預測結果
    prob_5class = np.array(result["probability_vector"])  # 確保轉換為 numpy array
    predicted_class_5 = result["predicted_class"]
    prediction_date = result["prediction_date"].replace(" (估算)", "")  # 移除估算標記

    print(f" → {prediction_date}")
    print_subsection_header("")

    # 聚合為3分類
    prob_3class = aggregate_to_3_categories(prob_5class)
    predicted_class_3 = get_predicted_class_3(prob_3class)
    confidence_3class = prob_3class[predicted_class_3]

    # 查詢實際資料
    actual_data = get_actual_data(
        df_raw, input_date, prediction_date, predicted_class_5, predicted_class_3
    )

    # 印出文字結果
    class_labels = [CLASS_MAPPING[i]["label"] for i in range(5)]
    print_3class_results(prob_3class, predicted_class_3, confidence_3class)
    print_5class_results(prob_5class, predicted_class_5, class_labels)
    print_actual_results(actual_data, prediction_date)

    # 繪製視覺化 (若未禁用)
    if not no_plot:
        # 組裝 PredictionResultDict
        prediction_result = {
            "input_date": input_date,
            "prediction_date": prediction_date,
            "prob_5class": prob_5class,
            "prob_3class": prob_3class,
            "predicted_class_5": predicted_class_5,
            "predicted_class_3": predicted_class_3,
            "confidence": result["confidence"],
            "actual_data": actual_data,
        }

        # 生成輸出路徑
        filename = f"prediction_{input_date}_{prediction_date}.png"
        output_path = os.path.join(output_dir, filename)

        try:
            saved_path = plot_dual_view(prediction_result, output_path, config)
            print(f"\n視覺化圖表已儲存至: {saved_path}")
        except Exception as e:
            logging.error(f"視覺化生成失敗: {e}")
            print(f"\n視覺化生成失敗: {e}")

    # 回傳統計資訊
    stats = {
        "status": "success",
        "has_actual": actual_data is not None,
        "is_correct_5": actual_data["is_correct_5class"] if actual_data else None,
        "is_correct_3": actual_data["is_correct_3class"] if actual_data else None,
    }

    return stats


def main() -> int:
    """
    CLI 主程式入口點

    Returns:
        int: 退出碼 (0=成功, 非0=失敗)
    """
    # 解析參數
    args = parse_arguments()

    # 設定日誌
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="[%(levelname)s] %(message)s")

    # 記錄開始時間
    start_time = time.time()

    # 印出標題
    print_section_header("增強版股價預測結果")

    # 驗證檔案存在
    if not os.path.exists(args.model_file):
        print(f"錯誤: 找不到模型檔案 {args.model_file}")
        return 1

    if not os.path.exists(args.data_file):
        print(f"錯誤: 找不到資料檔案 {args.data_file}")
        return 1

    # 建立輸出目錄
    if not args.no_plot:
        os.makedirs(args.output_dir, exist_ok=True)

    # 印出配置資訊
    print(f"模型檔案: {args.model_file}")
    print(f"資料檔案: {args.data_file}")
    if args.feature_set:
        print(f"特徵集: {args.feature_set}")
    if args.time_steps:
        print(f"時間窗口: {args.time_steps}")
    print("")

    # 載入模型
    logging.info("載入模型中...")
    try:
        model = load_trained_model(args.model_file)
        logging.info(f"模型載入成功")
    except Exception as e:
        print(f"錯誤: 模型載入失敗 - {e}")
        return 2

    # 載入資料
    logging.info("載入歷史資料中...")
    try:
        df_raw = load_csv_data(args.data_file)
        logging.info(f"資料載入成功 (共 {len(df_raw)} 筆)")
        logging.info(f"資料範圍: {df_raw['date'].min()} 至 {df_raw['date'].max()}")
    except Exception as e:
        print(f"錯誤: 資料載入失敗 - {e}")
        return 3

    # 載入模型配置 (若存在)
    feature_set_id = args.feature_set
    time_steps = args.time_steps

    if feature_set_id is None or time_steps is None:
        try:
            model_config = load_model_config(args.model_file)
            if feature_set_id is None:
                feature_set_id = model_config.get("feature_set_id", "Set A")
            if time_steps is None:
                time_steps = model_config.get("time_steps", 60)
            logging.info(
                f"從配置檔載入: feature_set={feature_set_id}, time_steps={time_steps}"
            )
        except Exception as e:
            logging.warning(f"無法載入模型配置,使用預設值: {e}")
            feature_set_id = feature_set_id or "Set A"
            time_steps = time_steps or 60

    # 取得視覺化配置
    viz_config = get_default_config()

    # 處理每個輸入日期
    input_dates = args.input_date
    total_count = len(input_dates)
    results = []

    for idx, input_date in enumerate(input_dates, 1):
        stats = process_single_prediction(
            model,
            df_raw,
            feature_set_id,
            time_steps,
            input_date,
            args.output_dir,
            args.no_plot,
            viz_config,
            idx,
            total_count,
        )
        results.append(stats)

    # 印出批次摘要 (若有多個預測)
    if total_count > 1:
        print_section_header("批次預測摘要")

        success_count = sum(1 for r in results if r["status"] == "success")
        verifiable_count = sum(1 for r in results if r["has_actual"])
        correct_5_count = sum(1 for r in results if r.get("is_correct_5"))
        correct_3_count = sum(1 for r in results if r.get("is_correct_3"))

        print(f"總預測數: {total_count}")
        print(f"成功預測: {success_count}")
        if verifiable_count > 0:
            accuracy_5 = correct_5_count / verifiable_count * 100
            accuracy_3 = correct_3_count / verifiable_count * 100
            print(
                f"  - 可驗證: {verifiable_count} (5分類正確: {correct_5_count}, 準確率: {accuracy_5:.1f}%)"
            )
            print(
                f"             (3分類正確: {correct_3_count}, 準確率: {accuracy_3:.1f}%)"
            )
        pending_count = total_count - verifiable_count
        if pending_count > 0:
            print(f"  - 待驗證: {pending_count} (未來日期或資料不存在)")

    # 計算執行時間
    elapsed_time = time.time() - start_time
    avg_time = elapsed_time / total_count if total_count > 0 else 0

    if total_count > 1:
        print(f"執行總時間: {elapsed_time:.2f} 秒")
        print(f"平均每筆: {avg_time:.2f} 秒")
    else:
        print(f"\n執行時間: {elapsed_time:.2f} 秒")

    print_section_header("")

    return 0


if __name__ == "__main__":
    sys.exit(main())
