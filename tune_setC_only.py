"""
專門針對 Set C 進行超參數調整

使用方式：
    python tune_setC_only.py
"""

import sys
from pathlib import Path

# 加入專案路徑
sys.path.insert(0, str(Path(__file__).parent))

from src.tuning.hyperparameter_tuner import create_tuner, run_tuning, get_best_model, save_tuner_results, compare_with_baseline
from src.utils.gpu_checker import setup_gpu
from src.utils.logger import setup_logger
from src.utils.model_config import save_model_config
from src.data.data_loader import load_csv_data
from src.data.data_splitter import split_time_series, create_sequences
from src.features.feature_engineer import prepare_features_and_target
from src.features.feature_sets import get_feature_names
from src.features.scalers import create_scaler, fit_scaler, transform_features
from datetime import datetime
import logging

try:
    from tensorflow import keras
    import keras_tuner as kt
except ImportError:
    keras = None
    kt = None


def build_model_setC_only(hp):
    """固定使用 Set C，只調整其他超參數"""
    if keras is None:
        raise ImportError("TensorFlow/Keras 未安裝")

    from src.tuning.feature_set_selector import get_n_features_for_feature_set
    from src.models.model_builder import build_dynamic_lstm_model

    # 1. 固定使用 Set C
    feature_set_id = "Set C"
    n_features = get_n_features_for_feature_set(feature_set_id)

    # 將 feature_set_id 存入 hp，雖然不搜尋但要記錄
    hp.Fixed("feature_set_id", "Set C")

    # 2. 時間窗口（保留搜尋）
    time_steps = hp.Choice("time_steps", [20, 40, 60, 80])

    # 3. LSTM 層數（保留搜尋）
    num_layers = hp.Int("num_layers", min_value=2, max_value=4, step=1)

    # 4. 每層單元數（保留搜尋）
    units_per_layer = []
    for i in range(num_layers):
        units = hp.Choice(f"units_layer_{i+1}", [32, 64, 96, 128])
        units_per_layer.append(units)

    # 5. Dropout 比率（擴大搜尋範圍）
    dropout_rate = hp.Float("dropout_rate", min_value=0.1, max_value=0.5, step=0.05)

    # 6. 學習率（保留搜尋）
    learning_rate = hp.Choice("learning_rate", [0.001, 0.0005, 0.0001])

    # 建立模型
    model = build_dynamic_lstm_model(
        time_steps=time_steps,
        n_features=n_features,
        n_classes=5,
        num_layers=num_layers,
        units_per_layer=units_per_layer,
        dropout_rate=dropout_rate,
        learning_rate=learning_rate,
    )

    logging.debug(f"Trial 模型建立完成 (Set C 固定):")
    logging.debug(f"  - 特徵集: Set C ({n_features} 個特徵)")
    logging.debug(f"  - 時間窗口: {time_steps}")
    logging.debug(f"  - 層數: {num_layers}")
    logging.debug(f"  - 單元數: {units_per_layer}")
    logging.debug(f"  - Dropout: {dropout_rate}")
    logging.debug(f"  - 學習率: {learning_rate}")

    return model


def main():
    """主程式"""
    # 設定參數
    DATA_FILE = "19980601-20251111-converted.csv"
    BASELINE_MODEL = "models/best_tuned_model.h5"  # 使用你目前最好的模型作參考
    MAX_TRIALS = 100
    EPOCHS_PER_TRIAL = 100
    PROJECT_NAME = "lstm_stock_tuning_setC_only"
    TUNED_MODEL_NAME = "best_tuned_setC_optimized"

    # 設置 logger
    logger = setup_logger("tuning_setC", log_dir="logs/tuning_logs")

    logger.info("=" * 80)
    logger.info("Set C 專用超參數調整")
    logger.info("=" * 80)
    logger.info(f"資料檔案: {DATA_FILE}")
    logger.info(f"參考模型: {BASELINE_MODEL}")
    logger.info(f"最大試驗次數: {MAX_TRIALS}")
    logger.info(f"每次試驗週期: {EPOCHS_PER_TRIAL}")
    logger.info("固定特徵集: Set C")
    logger.info("=" * 80)

    # 1. GPU 設置
    logger.info("\n步驟 1/7: 檢查 GPU 可用性")
    gpu_available, gpu_message = setup_gpu()
    logger.info(gpu_message)

    # 2. 載入資料
    logger.info("\n步驟 2/7: 載入資料")
    df = load_csv_data(DATA_FILE)
    logger.info(f"✅ 載入 {len(df)} 筆交易資料")

    # 3. 特徵工程（固定使用 Set C）
    logger.info("\n步驟 3/7: 特徵工程（固定 Set C）")
    features, target_onehot, df_processed = prepare_features_and_target(
        df, feature_set_id="Set C", look_ahead_days=20
    )
    logger.info(f"✅ 特徵集: Set C")
    logger.info(f"   特徵形狀: {features.shape}")
    logger.info(f"   目標形狀: {target_onehot.shape}")

    # 4. 資料集分割
    logger.info("\n步驟 4/7: 資料集分割")
    train_df, val_df, test_df = split_time_series(
        df_processed, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
    )

    # 5. 特徵縮放
    logger.info("\n步驟 5/7: 特徵縮放")
    feature_columns = get_feature_names("Set C")

    train_features = train_df[feature_columns].values
    val_features = val_df[feature_columns].values
    test_features = test_df[feature_columns].values

    scaler = create_scaler("standard")
    scaler = fit_scaler(scaler, train_features)
    train_scaled = transform_features(scaler, train_features)
    val_scaled = transform_features(scaler, val_features)
    test_scaled = transform_features(scaler, test_features)

    # 建立序列
    from sklearn.preprocessing import OneHotEncoder
    encoder = OneHotEncoder(sparse_output=False, categories=[range(5)])

    train_target = train_df["target_class"].values
    val_target = val_df["target_class"].values
    test_target = test_df["target_class"].values

    train_target_onehot = encoder.fit_transform(train_target.reshape(-1, 1))
    val_target_onehot = encoder.transform(val_target.reshape(-1, 1))
    test_target_onehot = encoder.transform(test_target.reshape(-1, 1))

    # 注意：這裡需要根據不同的 time_steps 建立序列
    # 為了簡化，我們先用 60（可以在調整時動態處理）
    # 實際上 Keras Tuner 會為每個 time_steps 重新準備資料

    # 6. 建立自訂 Tuner
    logger.info("\n步驟 6/7: 建立 Keras Tuner（Set C 專用）")

    # 檢查是否存在先前的試驗
    tuning_dir = Path("logs/tuning_logs") / PROJECT_NAME
    existing_trials = 0
    if tuning_dir.exists():
        trial_dirs = list(tuning_dir.glob("trial_*"))
        existing_trials = len(trial_dirs)
        if existing_trials > 0:
            logger.info(f"⚠️ 偵測到先前的調整紀錄: {tuning_dir}")
            logger.info(f"⚠️ 已有 {existing_trials} 個試驗數據")
            logger.info(f"⚠️ 將繼續調整至 {MAX_TRIALS} 個試驗")
            logger.info(f"⚠️ 預計新增: {max(0, MAX_TRIALS - existing_trials)} 個試驗")
            logger.info("")

    tuner = kt.BayesianOptimization(
        hypermodel=build_model_setC_only,
        objective=kt.Objective("val_loss", direction="min"),
        max_trials=MAX_TRIALS,
        directory="logs/tuning_logs",
        project_name=PROJECT_NAME,
        overwrite=False,  # ⚠️ 改成 False 以保留舊試驗並繼續
    )

    logger.info("✅ Keras Tuner 建立完成")
    logger.info("\n超參數搜尋空間:")
    logger.info("  - 特徵集: Set C（固定）")
    logger.info("  - 時間窗口: 20, 40, 60, 80")
    logger.info("  - LSTM 層數: 2, 3, 4")
    logger.info("  - 每層單元數: 32, 64, 96, 128")
    logger.info("  - Dropout: 0.1 - 0.5（步長 0.05）")
    logger.info("  - 學習率: 0.001, 0.0005, 0.0001")

    # 7. 執行調整（使用固定的 time_steps=60 進行示範）
    logger.info("\n步驟 7/7: 執行超參數調整")
    logger.info("⚠️ 注意：此版本使用固定 time_steps=60 的序列")
    logger.info("   如需支援動態 time_steps，需要更複雜的實作")

    # 使用 time_steps=60 建立序列
    X_train, y_train = create_sequences(train_scaled, train_target_onehot, 60)
    X_val, y_val = create_sequences(val_scaled, val_target_onehot, 60)
    X_test, y_test = create_sequences(test_scaled, test_target_onehot, 60)

    logger.info(f"訓練集: {X_train.shape}")
    logger.info(f"驗證集: {X_val.shape}")
    logger.info(f"測試集: {X_test.shape}")

    # 執行搜尋
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True, verbose=0
        )
    ]

    import time
    start_time = time.time()

    tuner.search(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS_PER_TRIAL,
        callbacks=callbacks,
        verbose=1,
    )

    total_time = time.time() - start_time

    # 取得最佳模型
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    best_model = tuner.get_best_models(num_models=1)[0]

    # 統計資訊
    best_trial = tuner.oracle.get_best_trials(num_trials=1)[0]
    stats = {
        "total_time": total_time,
        "total_trials": len(tuner.oracle.trials),
        "best_hyperparameters": {
            "feature_set_id": "Set C",
            "time_steps": best_hps.get("time_steps"),
            "num_layers": best_hps.get("num_layers"),
            "units_per_layer": [
                best_hps.get(f"units_layer_{i+1}")
                for i in range(best_hps.get("num_layers"))
            ],
            "dropout_rate": best_hps.get("dropout_rate"),
            "learning_rate": best_hps.get("learning_rate"),
            "batch_size": 32,
        },
        "best_val_loss": best_trial.score,
        "best_val_accuracy": best_trial.metrics.get_best_value("val_accuracy"),
    }

    logger.info("\n" + "=" * 80)
    logger.info("超參數調整完成")
    logger.info(f"總試驗次數: {stats['total_trials']}")
    logger.info(f"總執行時間: {total_time / 3600:.2f} 小時")
    logger.info(f"最佳驗證損失: {stats['best_val_loss']:.4f}")
    logger.info(f"最佳驗證準確度: {stats['best_val_accuracy']:.2%}")
    logger.info("=" * 80)

    # 儲存模型
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name_with_timestamp = f"{TUNED_MODEL_NAME}_{timestamp}"

    output_path = Path("models")
    output_path.mkdir(parents=True, exist_ok=True)
    model_file = output_path / f"{model_name_with_timestamp}.h5"
    best_model.save(str(model_file))
    logger.info(f"✅ 最佳模型已儲存: {model_file}")

    # 儲存配置
    config_file_path = str(model_file).replace('.h5', '_config.json')
    save_model_config(
        model_path=str(model_file),
        feature_set_id="Set C",
        time_steps=best_hps.get("time_steps"),
        additional_params={
            "num_layers": best_hps.get("num_layers"),
            "dropout_rate": best_hps.get("dropout_rate"),
            "learning_rate": best_hps.get("learning_rate"),
            "batch_size": 32,
            "epochs_trained": EPOCHS_PER_TRIAL,
            "tuner_type": "bayesian",
            "max_trials": MAX_TRIALS,
            "baseline_model": BASELINE_MODEL,
            "tuning_timestamp": timestamp,
            "model_type": "tuned_setC_only",
        },
    )
    logger.info(f"✅ 模型配置已儲存: {config_file_path}")

    # 儲存調整結果
    results_file = f"logs/tuning_logs/tuning_results_setC_{timestamp}.txt"
    save_tuner_results(tuner, stats, results_file, BASELINE_MODEL)
    logger.info(f"✅ 調整結果已儲存: {results_file}")

    # 評估並比較
    test_loss, test_accuracy = best_model.evaluate(X_test, y_test, verbose=0)
    logger.info(f"\n測試集效能:")
    logger.info(f"  測試損失: {test_loss:.4f}")
    logger.info(f"  測試準確度: {test_accuracy:.2%}")

    # 與參考模型比較
    if Path(BASELINE_MODEL).exists():
        logger.info("\n與參考模型比較...")
        try:
            comparison_result = compare_with_baseline(
                tuned_model=best_model,
                baseline_model_path=BASELINE_MODEL,
                X_test=X_test,
                y_test=y_test,
                logger=logger,
            )

            # 儲存比較報告
            comparison_file = f"logs/tuning_logs/comparison_report_setC_{timestamp}.txt"
            with open(comparison_file, "w", encoding="utf-8") as f:
                f.write("=" * 80 + "\n")
                f.write("Set C 優化模型比較報告\n")
                f.write("=" * 80 + "\n\n")
                f.write(f"調整時間: {timestamp}\n")
                f.write(f"參考模型: {BASELINE_MODEL}\n")
                f.write(f"優化模型: {model_file}\n\n")
                f.write("=" * 80 + "\n")
                f.write("效能比較（測試集）\n")
                f.write("=" * 80 + "\n\n")
                f.write(f"參考模型:\n")
                f.write(f"  - 測試損失: {comparison_result['baseline_test_loss']:.4f}\n")
                f.write(f"  - 測試準確度: {comparison_result['baseline_test_accuracy']:.2%}\n\n")
                f.write(f"優化模型:\n")
                f.write(f"  - 測試損失: {comparison_result['tuned_test_loss']:.4f}\n")
                f.write(f"  - 測試準確度: {comparison_result['tuned_test_accuracy']:.2%}\n\n")
                f.write("=" * 80 + "\n")
                f.write("改善幅度\n")
                f.write("=" * 80 + "\n\n")
                f.write(f"損失改善: {comparison_result['loss_improvement']:.4f} ({comparison_result['loss_improvement_percent']:.2f}%)\n")
                f.write(f"準確度改善: {comparison_result['accuracy_improvement']:.4f} ({comparison_result['accuracy_improvement_percent']:.2f}%)\n\n")

                if comparison_result['is_better']:
                    f.write("✅ 優化模型優於參考模型\n")
                else:
                    f.write("⚠️ 優化模型未優於參考模型\n")

            logger.info(f"✅ 比較報告已儲存: {comparison_file}")

        except Exception as e:
            logger.error(f"❌ 模型比較失敗: {e}")

    logger.info("\n" + "=" * 80)
    logger.info("✅ Set C 專用超參數調整完成！")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
