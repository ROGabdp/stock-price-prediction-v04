"""
超參數調整 CLI 指令

執行 Keras Tuner 超參數調整，自動搜尋最佳模型配置（含特徵集選擇）
"""

import argparse
import logging
import sys
from pathlib import Path

# 確保可以 import src 模組
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.data_loader import load_csv_data
from src.data.data_splitter import create_sequences
from src.features.feature_engineer import prepare_features_and_target
from src.features.scalers import create_scaler, fit_scaler, transform_features
from src.tuning.hyperparameter_tuner import (
    create_tuner,
    run_tuning,
    get_best_model,
    save_tuner_results,
)
from src.tuning.feature_set_selector import select_feature_set
from src.utils.gpu_checker import setup_gpu
from src.utils.logger import setup_logger
from src.utils.model_config import save_model_config


def main():
    """
    超參數調整主程式

    執行流程:
    1. 載入歷史資料
    2. 對每個特徵集進行特徵工程
    3. 資料集分割（訓練/驗證/測試）
    4. 建立 Keras Tuner
    5. 執行超參數調整（自動選擇特徵集）
    6. 儲存最佳模型與調整結果
    """
    # 解析命令列參數
    parser = argparse.ArgumentParser(
        description="LSTM 台股預測系統 - 超參數調整",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用範例:
  # 基本用法（50 次試驗）
  python src/cli/tune.py \\
      --data-file 19980601-20251111-converted.csv \\
      --max-trials 50 \\
      --epochs-per-trial 50

  # 指定 Tuner 類型與輸出目錄
  python src/cli/tune.py \\
      --data-file data.csv \\
      --tuner-type bayesian \\
      --max-trials 100 \\
      --output-dir models/tuned/

  # 快速測試（僅 5 次試驗）
  python src/cli/tune.py \\
      --data-file data.csv \\
      --max-trials 5 \\
      --epochs-per-trial 10
        """,
    )

    parser.add_argument(
        "--data-file",
        type=str,
        required=True,
        help="歷史資料 CSV 檔案路徑 (例如: 19980601-20251111-converted.csv)",
    )

    parser.add_argument(
        "--max-trials",
        type=int,
        default=50,
        help="最大試驗次數（預設: 50）",
    )

    parser.add_argument(
        "--epochs-per-trial",
        type=int,
        default=50,
        help="每次試驗的訓練週期數（預設: 50）",
    )

    parser.add_argument(
        "--tuner-type",
        type=str,
        default="random",
        choices=["random", "bayesian", "hyperband"],
        help='Tuner 類型（預設: "random"）',
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="模型儲存目錄（預設: models/）",
    )

    parser.add_argument(
        "--val-split",
        type=float,
        default=0.15,
        help="驗證集比例（預設: 0.15）",
    )

    parser.add_argument(
        "--test-split",
        type=float,
        default=0.15,
        help="測試集比例（預設: 0.15）",
    )

    parser.add_argument(
        "--project-name",
        type=str,
        default="lstm_stock_tuning",
        help='Tuner 專案名稱（預設: "lstm_stock_tuning"）',
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="覆寫現有的調整紀錄（預設: False，繼續先前的調整）",
    )

    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="訓練輸出詳細程度（0=靜默, 1=進度條, 2=每個 epoch 一行，預設: 1）",
    )

    args = parser.parse_args()

    # 設置日誌
    logger = setup_logger(name="tune", log_dir="logs/tuning_logs", level=logging.INFO)

    logger.info("=" * 80)
    logger.info("LSTM 台股價格預測系統 - 超參數調整")
    logger.info("=" * 80)
    logger.info(f"資料檔案: {args.data_file}")
    logger.info(f"Tuner 類型: {args.tuner_type}")
    logger.info(f"最大試驗次數: {args.max_trials}")
    logger.info(f"每次試驗訓練週期: {args.epochs_per_trial}")
    logger.info(f"輸出目錄: {args.output_dir}")
    logger.info("=" * 80)

    # 檢查 GPU 可用性
    logger.info("\n步驟 1/7: 檢查 GPU 可用性")
    gpu_available = setup_gpu()

    # 載入資料
    logger.info("\n步驟 2/7: 載入歷史資料")
    try:
        df_raw = load_csv_data(args.data_file)
        logger.info(f"✅ 資料載入成功: {len(df_raw)} 筆交易日")
    except Exception as e:
        logger.error(f"❌ 資料載入失敗: {e}")
        sys.exit(1)

    # 特徵工程（使用 Set C 全特徵集，Tuner 會自動選擇子集）
    logger.info("\n步驟 3/7: 特徵工程與目標變數準備（Set C: 全特徵集）")
    try:
        features, target, df_processed = prepare_features_and_target(df_raw, feature_set_id="Set C")
        logger.info(f"✅ 特徵工程完成: {features.shape[1]} 個特徵")
        logger.info(f"✅ 目標變數準備完成: {target.shape}")
    except Exception as e:
        logger.error(f"❌ 特徵工程失敗: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

    # 資料集分割 (簡單按比例分割特徵和目標)
    logger.info("\n步驟 4/7: 資料集分割")
    try:
        # 計算分割點
        n_samples = len(features)
        train_end = int(n_samples * (1 - args.val_split - args.test_split))
        val_end = int(n_samples * (1 - args.test_split))

        # 分割特徵和目標
        features_train = features[:train_end]
        target_train = target[:train_end]
        features_val = features[train_end:val_end]
        target_val = target[train_end:val_end]
        features_test = features[val_end:]
        target_test = target[val_end:]

        # 注意：time_steps 將由 Tuner 動態決定，這裡使用最大值 80 建立序列
        # 實際使用時 Tuner 會根據選定的 time_steps 重新切割
        time_steps_max = 80
        X_train, y_train = create_sequences(features_train, target_train, time_steps_max)
        X_val, y_val = create_sequences(features_val, target_val, time_steps_max)
        X_test, y_test = create_sequences(features_test, target_test, time_steps_max)

        logger.info(f"✅ 資料集分割完成:")
        logger.info(f"  - 訓練集: X {X_train.shape}, y {y_train.shape}")
        logger.info(f"  - 驗證集: X {X_val.shape}, y {y_val.shape}")
        logger.info(f"  - 測試集: X {X_test.shape}, y {y_test.shape}")
    except Exception as e:
        logger.error(f"❌ 資料集分割失敗: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

    # 建立 Keras Tuner
    logger.info("\n步驟 5/7: 建立 Keras Tuner")
    try:
        tuner = create_tuner(
            tuner_type=args.tuner_type,
            max_trials=args.max_trials,
            project_name=args.project_name,
            directory="logs/tuning_logs",
            overwrite=args.overwrite,
        )
        logger.info("✅ Keras Tuner 建立完成")
        logger.info("\n超參數搜尋空間:")
        logger.info("  - 特徵集: Set A, Set B, Set C")
        logger.info("  - 時間窗口: 20, 40, 60, 80")
        logger.info("  - LSTM 層數: 2, 3, 4")
        logger.info("  - 每層單元數: 32, 64, 96, 128")
        logger.info("  - Dropout: 0.1 - 0.4")
        logger.info("  - 學習率: 0.001, 0.0005, 0.0001")
        logger.info("  - 批次大小: 32, 64, 128")
    except Exception as e:
        logger.error(f"❌ Tuner 建立失敗: {e}")
        sys.exit(1)

    # 執行超參數調整
    logger.info("\n步驟 6/7: 執行超參數調整")
    logger.info(f"此過程可能需要數小時，請耐心等待...")
    logger.info("=" * 80)

    try:
        tuner, stats = run_tuning(
            tuner=tuner,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            epochs_per_trial=args.epochs_per_trial,
            verbose=args.verbose,
        )
        logger.info("✅ 超參數調整完成")
    except Exception as e:
        logger.error(f"❌ 超參數調整失敗: {e}")
        sys.exit(1)

    # 儲存最佳模型與結果
    logger.info("\n步驟 7/7: 儲存最佳模型與結果")
    try:
        # 取得最佳模型和超參數
        best_model = get_best_model(tuner)
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

        # 儲存最佳模型
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        model_file = output_path / "best_tuned_model.h5"
        best_model.save(str(model_file))
        logger.info(f"✅ 最佳模型已儲存: {model_file}")

        # 保存模型配置
        save_model_config(
            model_path=str(model_file),
            feature_set_id=best_hps.get("feature_set_id"),
            time_steps=best_hps.get("time_steps"),
            additional_params={
                "num_layers": best_hps.get("num_layers"),
                "dropout_rate": best_hps.get("dropout_rate"),
                "learning_rate": best_hps.get("learning_rate"),
                "batch_size": best_hps.values.get("batch_size", 32),
                "epochs_trained": args.epochs_per_trial,
                "tuner_type": args.tuner_type,
                "max_trials": args.max_trials,
            },
        )
        logger.info(f"✅ 模型配置已儲存: {model_file.with_suffix('')}_config.json")

        # 儲存調整結果
        results_file = "logs/tuning_results.txt"
        save_tuner_results(tuner, stats, results_file)
        logger.info(f"✅ 調整結果已儲存: {results_file}")

        # 評估最佳模型
        logger.info("\n最佳模型評估（測試集）:")
        test_loss, test_accuracy = best_model.evaluate(
            X_test, y_test, verbose=0
        )
        logger.info(f"  - 測試損失: {test_loss:.4f}")
        logger.info(f"  - 測試準確度: {test_accuracy:.2%}")

    except Exception as e:
        logger.error(f"❌ 儲存失敗: {e}")
        sys.exit(1)

    # 完成
    logger.info("\n" + "=" * 80)
    logger.info("超參數調整完成！")
    logger.info("=" * 80)
    logger.info(f"最佳模型: {model_file}")
    logger.info(f"調整結果: {results_file}")
    logger.info(f"試驗紀錄: logs/tuning_logs/{args.project_name}/")
    logger.info("\n最佳超參數:")
    for key, value in stats["best_hyperparameters"].items():
        logger.info(f"  - {key}: {value}")
    logger.info(f"\n最佳驗證損失: {stats['best_val_loss']:.4f}")
    logger.info(f"最佳驗證準確度: {stats['best_val_accuracy']:.2%}")
    logger.info(f"測試準確度: {test_accuracy:.2%}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
