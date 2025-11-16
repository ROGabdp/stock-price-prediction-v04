"""
訓練 CLI 指令

提供命令列介面來訓練 LSTM 模型。

使用範例:
    python src/cli/train.py \
        --data-file 19980601-20251111-converted.csv \
        --feature-set "Set A" \
        --time-steps 60 \
        --epochs 100 \
        --batch-size 32 \
        --output-dir models/
"""

import argparse
import logging
import sys
from pathlib import Path

# 加入專案根目錄至 Python 路徑
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.gpu_checker import setup_gpu
from src.utils.logger import setup_logger, log_training_start, log_training_complete, log_dataset_info
from src.utils.model_config import save_model_config
from src.data.data_loader import load_csv_data
from src.data.data_splitter import split_time_series, create_sequences
from src.features.feature_engineer import prepare_features_and_target
from src.features.scalers import create_scaler, fit_transform_features, save_scaler
from src.models.lstm_baseline import build_lstm_model
from src.models.model_builder import train_model, evaluate_model


def parse_arguments():
    """解析命令列參數"""
    parser = argparse.ArgumentParser(
        description="訓練 LSTM 台股預測模型",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--data-file",
        type=str,
        required=True,
        help="CSV 資料檔案路徑",
    )

    parser.add_argument(
        "--feature-set",
        type=str,
        default="Set A",
        choices=["Set A", "Set B", "Set C"],
        help="特徵集選擇",
    )

    parser.add_argument(
        "--time-steps",
        type=int,
        default=60,
        choices=[20, 40, 60, 80],
        help="時間窗口大小（天數）",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="訓練週期數",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        choices=[32, 64, 128],
        help="批次大小",
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="學習率",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="模型儲存目錄",
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default="baseline_model",
        help="模型名稱",
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early Stopping 耐心值",
    )

    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="訓練輸出詳細程度",
    )

    return parser.parse_args()


def main():
    """主程式"""
    # 解析參數
    args = parse_arguments()

    # 設置 logger
    logger = setup_logger("training", log_dir="logs/training_logs")

    logger.info("=" * 80)
    logger.info("LSTM 台股預測模型 - 訓練程序")
    logger.info("=" * 80)

    try:
        # 1. GPU 設置
        logger.info("\n步驟 1/8: 檢查 GPU 可用性")
        gpu_available, gpu_message = setup_gpu()
        logger.info(gpu_message)

        # 2. 載入資料
        logger.info("\n步驟 2/8: 載入資料")
        df = load_csv_data(args.data_file)
        logger.info(f"✅ 載入 {len(df)} 筆交易資料")
        logger.info(f"   日期範圍: {df['date'].min()} ~ {df['date'].max()}")

        # 3. 特徵工程與目標變數準備
        logger.info("\n步驟 3/8: 特徵工程與目標變數準備")
        features, target_onehot, df_processed = prepare_features_and_target(
            df, feature_set_id=args.feature_set, look_ahead_days=20
        )
        logger.info(f"✅ 特徵集: {args.feature_set}")
        logger.info(f"   特徵形狀: {features.shape}")
        logger.info(f"   目標形狀: {target_onehot.shape}")

        # 4. 資料集分割
        logger.info("\n步驟 4/8: 資料集分割（時間序列分割）")
        train_df, val_df, test_df = split_time_series(
            df_processed, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
        )

        # 5. 特徵縮放
        logger.info("\n步驟 5/8: 特徵縮放（StandardScaler）")
        from src.features.feature_sets import get_feature_names
        feature_columns = get_feature_names(args.feature_set)

        train_features = train_df[feature_columns].values
        val_features = val_df[feature_columns].values
        test_features = test_df[feature_columns].values

        scaler = create_scaler("standard")
        scaler, train_scaled, val_scaled, test_scaled = fit_transform_features(
            scaler, train_features, val_features, test_features
        )

        # 暫時儲存 scaler（稍後會用帶時間戳記的名稱更新）
        # scaler 會在訓練後取得實際的模型名稱後重新儲存
        temp_scaler = scaler  # 保留 scaler 物件供稍後使用

        # 6. 建立時間序列（滑動窗口）
        logger.info("\n步驟 6/8: 建立時間序列")
        train_target = train_df["target_class"].values
        val_target = val_df["target_class"].values
        test_target = test_df["target_class"].values

        # 轉換為 One-Hot
        from sklearn.preprocessing import OneHotEncoder
        encoder = OneHotEncoder(sparse_output=False, categories=[range(5)])
        train_target_onehot = encoder.fit_transform(train_target.reshape(-1, 1))
        val_target_onehot = encoder.transform(val_target.reshape(-1, 1))
        test_target_onehot = encoder.transform(test_target.reshape(-1, 1))

        X_train, y_train = create_sequences(train_scaled, train_target_onehot, args.time_steps)
        X_val, y_val = create_sequences(val_scaled, val_target_onehot, args.time_steps)
        X_test, y_test = create_sequences(test_scaled, test_target_onehot, args.time_steps)

        log_dataset_info(logger, len(X_train), len(X_val), len(X_test), features.shape[1])

        # 7. 建構模型
        logger.info("\n步驟 7/8: 建構 LSTM 模型")
        model = build_lstm_model(
            time_steps=args.time_steps,
            n_features=features.shape[1],
            n_classes=5,
            learning_rate=args.learning_rate,
        )

        # 記錄訓練開始
        log_training_start(
            logger,
            model_name=args.model_name,
            feature_set=args.feature_set,
            time_steps=args.time_steps,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )

        # 8. 訓練模型
        logger.info("\n步驟 8/8: 訓練模型")
        model, history = train_model(
            model,
            X_train,
            y_train,
            X_val,
            y_val,
            epochs=args.epochs,
            batch_size=args.batch_size,
            output_dir=args.output_dir,
            model_name=args.model_name,
            patience=args.patience,
            verbose=args.verbose,
        )

        # 評估測試集
        logger.info("\n評估測試集表現")
        test_loss, test_accuracy = evaluate_model(model, X_test, y_test)

        # 記錄訓練完成
        best_val_loss = min(history["val_loss"])
        best_val_accuracy = max(history["val_accuracy"])
        training_time = history["training_time"]
        model_name_with_timestamp = history["model_name"]  # 從 history 取得實際的模型名稱

        # 使用相同的時間戳記儲存 scaler
        scaler_path = Path(args.output_dir) / f"{model_name_with_timestamp}_scaler.pkl"
        save_scaler(temp_scaler, str(scaler_path))

        # 儲存模型配置（使用帶時間戳記的名稱）
        model_file_path = f"{args.output_dir}/{model_name_with_timestamp}.h5"
        save_model_config(
            model_path=model_file_path,
            feature_set_id=args.feature_set,
            time_steps=args.time_steps,
            additional_params={
                "learning_rate": args.learning_rate,
                "batch_size": args.batch_size,
                "epochs_trained": args.epochs,
                "patience": args.patience,
                "model_type": "baseline",
            },
        )
        logger.info(f"✅ 模型配置已儲存: {args.output_dir}/{model_name_with_timestamp}_config.json")

        log_training_complete(
            logger,
            val_loss=best_val_loss,
            val_accuracy=best_val_accuracy,
            training_time=training_time,
            model_path=model_file_path,
        )

        logger.info("\n" + "=" * 80)
        logger.info("✅ 訓練流程完成！")
        logger.info(f"   最佳驗證準確度: {best_val_accuracy:.2%}")
        logger.info(f"   測試準確度: {test_accuracy:.2%}")
        logger.info(f"   模型檔案: {args.output_dir}/{model_name_with_timestamp}.h5")
        logger.info(f"   配置檔案: {args.output_dir}/{model_name_with_timestamp}_config.json")
        logger.info(f"   縮放器檔案: {scaler_path}")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"\n❌ 訓練失敗: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
