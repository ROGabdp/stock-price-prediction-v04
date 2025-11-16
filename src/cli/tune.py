"""
超參數調整 CLI 指令

執行 Keras Tuner 超參數調整，自動搜尋最佳模型配置（含特徵集選擇）
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime

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
    compare_with_baseline,
)
from src.tuning.feature_set_selector import select_feature_set
from src.utils.gpu_checker import setup_gpu
from src.utils.logger import setup_logger
from src.utils.model_config import save_model_config, load_model_config


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

    parser.add_argument(
        "--baseline-model",
        type=str,
        default=None,
        help="基準模型路徑（.h5 檔案），用於比較調整後的模型是否有改善",
    )

    parser.add_argument(
        "--tuned-model-name",
        type=str,
        default="best_tuned_model",
        help="調整後模型的名稱（預設: best_tuned_model，會自動加上時間戳記）",
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

    # 檢查是否存在先前的試驗數據
    tuning_dir = Path("logs/tuning_logs") / args.project_name
    if tuning_dir.exists() and not args.overwrite:
        # 計算已有的試驗數量
        trial_dirs = list(tuning_dir.glob("trial_*"))
        num_existing_trials = len(trial_dirs)

        if num_existing_trials > 0:
            logger.warning("⚠️" + "=" * 78)
            logger.warning(f"⚠️ 偵測到先前的調整紀錄: {tuning_dir}")
            logger.warning(f"⚠️ 已有 {num_existing_trials} 個試驗數據")
            logger.warning("⚠️")
            logger.warning("⚠️ Keras Tuner 將會：")
            logger.warning("⚠️   1. 重複使用這些試驗結果（不會重新訓練）")
            logger.warning("⚠️   2. 如果 max_trials 更大，只訓練額外的試驗")
            logger.warning("⚠️   3. 這可能導致「瞬間完成」的情況")
            logger.warning("⚠️")
            logger.warning("⚠️ 如果要完全重新訓練，請使用以下任一方式：")
            logger.warning("⚠️   - 加上 --overwrite 參數")
            logger.warning("⚠️   - 使用不同的 --project-name")
            logger.warning(f"⚠️   - 手動刪除目錄: {tuning_dir}")
            logger.warning("⚠️" + "=" * 78)
            logger.warning("")

            # 給用戶 5 秒鐘時間閱讀警告
            import time
            for i in range(5, 0, -1):
                logger.warning(f"⚠️ 將在 {i} 秒後繼續使用現有數據...")
                time.sleep(1)
            logger.warning("")

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

        # 生成帶時間戳記的模型名稱
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tuned_model_name_with_timestamp = f"{args.tuned_model_name}_{timestamp}"

        # 儲存最佳模型
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        model_file = output_path / f"{tuned_model_name_with_timestamp}.h5"
        best_model.save(str(model_file))
        logger.info(f"✅ 最佳調整模型已儲存: {model_file}")

        # 載入基準模型資訊（如果有提供）
        baseline_model_path = None
        baseline_model_info = None
        if args.baseline_model:
            baseline_model_path = args.baseline_model
            if Path(baseline_model_path).exists():
                # 嘗試載入基準模型的配置
                # 配置檔案路徑：models/baseline_model_20251116_144357.h5 → models/baseline_model_20251116_144357_config.json
                baseline_config_path = str(Path(baseline_model_path).with_suffix('')) + "_config.json"
                if Path(baseline_config_path).exists():
                    baseline_model_info = load_model_config(baseline_model_path)
                    logger.info(f"📊 基準模型配置已載入: {baseline_config_path}")
                logger.info(f"📊 基準模型: {baseline_model_path}")
            else:
                logger.warning(f"⚠️ 基準模型檔案不存在: {baseline_model_path}")
                baseline_model_path = None

        # 保存模型配置（包含基準模型資訊）
        config_file = model_file.with_suffix('') / f"_config.json"
        # 正確的配置檔案路徑
        config_file_path = str(model_file).replace('.h5', '_config.json')

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
                "baseline_model": baseline_model_path,  # 記錄基準模型路徑
                "tuning_timestamp": timestamp,
                "model_type": "tuned",
            },
        )
        logger.info(f"✅ 模型配置已儲存: {config_file_path}")

        # 儲存調整結果（帶時間戳記）
        results_file = f"logs/tuning_logs/tuning_results_{timestamp}.txt"
        save_tuner_results(tuner, stats, results_file, baseline_model_path)
        logger.info(f"✅ 調整結果已儲存: {results_file}")

        # 評估最佳模型
        logger.info("\n最佳調整模型評估（測試集）:")
        test_loss, test_accuracy = best_model.evaluate(
            X_test, y_test, verbose=0
        )
        logger.info(f"  - 測試損失: {test_loss:.4f}")
        logger.info(f"  - 測試準確度: {test_accuracy:.2%}")

        # 與基準模型比較（如果有提供）
        comparison_result = None
        if baseline_model_path:
            logger.info("\n" + "=" * 80)
            logger.info("與基準模型比較")
            logger.info("=" * 80)
            try:
                comparison_result = compare_with_baseline(
                    tuned_model=best_model,
                    baseline_model_path=baseline_model_path,
                    X_test=X_test,
                    y_test=y_test,
                    logger=logger,
                )

                # 儲存比較報告
                comparison_file = f"logs/tuning_logs/comparison_report_{timestamp}.txt"
                Path(comparison_file).parent.mkdir(parents=True, exist_ok=True)
                with open(comparison_file, "w", encoding="utf-8") as f:
                    f.write("=" * 80 + "\n")
                    f.write("模型比較報告\n")
                    f.write("=" * 80 + "\n\n")
                    f.write(f"調整時間: {timestamp}\n")
                    f.write(f"基準模型: {baseline_model_path}\n")
                    f.write(f"調整模型: {model_file}\n\n")
                    f.write("=" * 80 + "\n")
                    f.write("效能比較（測試集）\n")
                    f.write("=" * 80 + "\n\n")
                    f.write(f"基準模型:\n")
                    f.write(f"  - 測試損失: {comparison_result['baseline_test_loss']:.4f}\n")
                    f.write(f"  - 測試準確度: {comparison_result['baseline_test_accuracy']:.2%}\n\n")
                    f.write(f"調整模型:\n")
                    f.write(f"  - 測試損失: {comparison_result['tuned_test_loss']:.4f}\n")
                    f.write(f"  - 測試準確度: {comparison_result['tuned_test_accuracy']:.2%}\n\n")
                    f.write("=" * 80 + "\n")
                    f.write("改善幅度\n")
                    f.write("=" * 80 + "\n\n")
                    f.write(f"損失改善: {comparison_result['loss_improvement']:.4f} ({comparison_result['loss_improvement_percent']:.2f}%)\n")
                    f.write(f"準確度改善: {comparison_result['accuracy_improvement']:.4f} ({comparison_result['accuracy_improvement_percent']:.2f}%)\n\n")

                    if comparison_result['is_better']:
                        f.write("✅ 調整模型優於基準模型\n")
                    else:
                        f.write("❌ 調整模型未優於基準模型\n")

                logger.info(f"✅ 比較報告已儲存: {comparison_file}")

            except Exception as e:
                logger.error(f"❌ 基準模型比較失敗: {e}")
                import traceback
                logger.error(traceback.format_exc())

    except Exception as e:
        logger.error(f"❌ 儲存失敗: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

    # 完成
    logger.info("\n" + "=" * 80)
    logger.info("超參數調整完成！")
    logger.info("=" * 80)
    logger.info(f"調整模型: {model_file}")
    logger.info(f"調整結果: {results_file}")
    logger.info(f"試驗紀錄: logs/tuning_logs/{args.project_name}/")

    if baseline_model_path:
        logger.info(f"\n基準模型: {baseline_model_path}")
        if comparison_result:
            logger.info(f"比較報告: logs/tuning_logs/comparison_report_{timestamp}.txt")

    logger.info("\n最佳超參數:")
    for key, value in stats["best_hyperparameters"].items():
        logger.info(f"  - {key}: {value}")

    logger.info(f"\n效能指標:")
    logger.info(f"  驗證損失: {stats['best_val_loss']:.4f}")
    logger.info(f"  驗證準確度: {stats['best_val_accuracy']:.2%}")
    logger.info(f"  測試準確度: {test_accuracy:.2%}")

    if comparison_result:
        logger.info(f"\n與基準模型比較:")
        logger.info(f"  基準模型測試準確度: {comparison_result['baseline_test_accuracy']:.2%}")
        logger.info(f"  調整模型測試準確度: {comparison_result['tuned_test_accuracy']:.2%}")
        logger.info(f"  準確度改善: {comparison_result['accuracy_improvement_percent']:+.2f}%")

        if comparison_result['is_better']:
            logger.info("  ✅ 調整模型優於基準模型")
        else:
            logger.info("  ⚠️ 調整模型未優於基準模型")

    logger.info("=" * 80)


if __name__ == "__main__":
    main()
