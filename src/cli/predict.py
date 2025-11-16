"""
預測 CLI 指令

根據指定日期預測未來 20 個交易日的漲跌幅區間與收盤價
"""

import argparse
import logging
import sys
from pathlib import Path
import time

# 確保可以 import src 模組
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.data_loader import load_csv_data
from src.prediction.predictor import (
    load_trained_model,
    predict_for_date,
    format_prediction_result,
)
from src.utils.logger import setup_logger
from src.utils.model_config import load_model_config


def main():
    """
    預測主程式

    執行流程:
    1. 載入訓練好的模型
    2. 載入歷史資料
    3. 驗證輸入日期
    4. 執行預測
    5. 格式化並輸出預測結果
    """
    # 解析命令列參數
    parser = argparse.ArgumentParser(
        description="LSTM 台股預測系統 - 預測",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用範例:
  # 基本用法（使用預設特徵集 Set A 與時間窗口 60）
  python src/cli/predict.py \\
      --model-file models/best_tuned_model.h5 \\
      --data-file 19980601-20251111-converted.csv \\
      --input-date "2024-01-15"

  # 指定特徵集與時間窗口
  python src/cli/predict.py \\
      --model-file models/baseline_model.h5 \\
      --data-file data.csv \\
      --input-date "2024-01-15" \\
      --feature-set "Set B" \\
      --time-steps 80

  # 批次預測多個日期
  python src/cli/predict.py \\
      --model-file models/best_tuned_model.h5 \\
      --data-file data.csv \\
      --input-date "2024-01-15" "2024-02-20" "2024-03-10"
        """,
    )

    parser.add_argument(
        "--model-file",
        type=str,
        required=True,
        help="訓練好的模型檔案路徑 (例如: models/best_tuned_model.h5)",
    )

    parser.add_argument(
        "--data-file",
        type=str,
        required=True,
        help="歷史資料 CSV 檔案路徑 (例如: 19980601-20251111-converted.csv)",
    )

    parser.add_argument(
        "--input-date",
        type=str,
        nargs="+",
        required=True,
        help='輸入日期（格式: "YYYY-MM-DD" 或 "YYYY/M/D"），可指定多個日期',
    )

    parser.add_argument(
        "--feature-set",
        type=str,
        default="Set A",
        choices=["Set A", "Set B", "Set C"],
        help='特徵集 ID（預設: "Set A"，必須與訓練時一致）',
    )

    parser.add_argument(
        "--time-steps",
        type=int,
        default=60,
        choices=[20, 40, 60, 80],
        help="時間窗口大小（預設: 60，必須與訓練時一致）",
    )

    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="儲存預測結果至檔案（選用，若不指定則僅輸出至控制台）",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="顯示詳細日誌（預設: False）",
    )

    args = parser.parse_args()

    # 設置日誌
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logger(name="predict", log_dir="logs/prediction_logs", level=log_level)

    logger.info("=" * 80)
    logger.info("LSTM 台股價格預測系統 - 預測")
    logger.info("=" * 80)
    logger.info(f"模型檔案: {args.model_file}")
    logger.info(f"資料檔案: {args.data_file}")
    logger.info(f"輸入日期: {args.input_date}")
    logger.info(f"特徵集: {args.feature_set}")
    logger.info(f"時間窗口: {args.time_steps}")
    logger.info("=" * 80)

    # 載入模型
    logger.info("\n步驟 1/4: 載入訓練好的模型")
    try:
        model = load_trained_model(args.model_file)
        logger.info(f"✅ 模型載入成功")
    except Exception as e:
        logger.error(f"❌ 模型載入失敗: {e}")
        sys.exit(1)

    # 嘗試載入模型配置
    model_config = load_model_config(args.model_file)
    if model_config:
        # 如果配置存在,使用配置中的參數(除非用戶明確指定)
        # 檢查用戶是否使用了默認值
        parser_defaults = {
            "feature_set": "Set A",
            "time_steps": 60,
        }

        # 如果用戶沒有明確指定 feature_set,使用配置中的值
        if args.feature_set == parser_defaults["feature_set"]:
            args.feature_set = model_config.get("feature_set_id", args.feature_set)
            logger.info(f"📋 使用模型配置中的特徵集: {args.feature_set}")

        # 如果用戶沒有明確指定 time_steps,使用配置中的值
        if args.time_steps == parser_defaults["time_steps"]:
            args.time_steps = model_config.get("time_steps", args.time_steps)
            logger.info(f"📋 使用模型配置中的時間窗口: {args.time_steps}")
    else:
        logger.warning("⚠️ 未找到模型配置文件,使用命令行參數")
        logger.warning(f"   請確認 --feature-set 和 --time-steps 與模型訓練時一致")
        logger.warning(f"   當前使用: feature_set={args.feature_set}, time_steps={args.time_steps}")

    # 載入歷史資料
    logger.info("\n步驟 2/4: 載入歷史資料")
    try:
        df_raw = load_csv_data(args.data_file)
        logger.info(f"✅ 資料載入成功: {len(df_raw)} 筆交易日")
        logger.info(f"資料範圍: {df_raw['date'].min()} 至 {df_raw['date'].max()}")
    except Exception as e:
        logger.error(f"❌ 資料載入失敗: {e}")
        sys.exit(1)

    # 執行預測（支援批次預測）
    logger.info(f"\n步驟 3/4: 執行預測（共 {len(args.input_date)} 個日期）")
    results = []

    for i, input_date in enumerate(args.input_date, 1):
        logger.info(f"\n預測 {i}/{len(args.input_date)}: {input_date}")
        logger.info("-" * 80)

        try:
            start_time = time.time()

            result = predict_for_date(
                model=model,
                df_raw=df_raw,
                input_date=input_date,
                feature_set_id=args.feature_set,
                time_steps=args.time_steps,
            )

            elapsed_time = time.time() - start_time
            result["elapsed_time"] = elapsed_time

            results.append(result)

            logger.info(f"✅ 預測完成（耗時: {elapsed_time:.2f} 秒）")

        except Exception as e:
            logger.error(f"❌ 預測失敗: {e}")
            # 繼續預測下一個日期
            continue

    # 輸出預測結果
    logger.info("\n步驟 4/4: 輸出預測結果")
    logger.info("=" * 80)

    if not results:
        logger.error("所有預測均失敗，無結果可輸出")
        sys.exit(1)

    # 格式化並輸出每個預測結果
    formatted_results = []
    for i, result in enumerate(results, 1):
        formatted = format_prediction_result(result)
        formatted_results.append(formatted)

        # 輸出至控制台
        if i == 1:
            print()  # 空行分隔
        print(formatted)

        # 顯示執行時間
        logger.info(f"預測 {i} 完成: {result['input_date']} → {result['prediction_date']}")
        logger.info(f"  - 預測類別: {result['predicted_class']} ({result['predicted_class_label']})")
        logger.info(f"  - 信心度: {result['confidence']:.2%}")
        logger.info(f"  - 執行時間: {result['elapsed_time']:.2f} 秒")

    # 儲存至檔案（選用）
    if args.output_file:
        try:
            output_path = Path(args.output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(args.output_file, "w", encoding="utf-8") as f:
                f.write("\n\n".join(formatted_results))

            logger.info(f"\n✅ 預測結果已儲存至: {args.output_file}")
        except Exception as e:
            logger.error(f"❌ 儲存失敗: {e}")

    # 完成
    logger.info("\n" + "=" * 80)
    logger.info(f"預測完成！共處理 {len(results)}/{len(args.input_date)} 個日期")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
