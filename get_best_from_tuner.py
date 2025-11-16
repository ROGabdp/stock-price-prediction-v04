"""
從 Keras Tuner 中斷的調整中取得目前最佳模型

使用方式：
    python get_best_from_tuner.py
"""

import sys
from pathlib import Path
from datetime import datetime

# 加入專案路徑
sys.path.insert(0, str(Path(__file__).parent))

try:
    from tensorflow import keras
    import keras_tuner as kt
except ImportError:
    print("❌ 錯誤：需要安裝 tensorflow 和 keras-tuner")
    sys.exit(1)

from src.utils.logger import setup_logger
from src.utils.model_config import save_model_config


def build_model_setC_only(hp):
    """與 tune_setC_only.py 中相同的模型建構函數"""
    from src.tuning.feature_set_selector import get_n_features_for_feature_set
    from src.models.model_builder import build_dynamic_lstm_model

    feature_set_id = "Set C"
    n_features = get_n_features_for_feature_set(feature_set_id)
    hp.Fixed("feature_set_id", "Set C")

    time_steps = hp.Choice("time_steps", [20, 40, 60, 80])
    num_layers = hp.Int("num_layers", min_value=2, max_value=4, step=1)

    units_per_layer = []
    for i in range(num_layers):
        units = hp.Choice(f"units_layer_{i+1}", [32, 64, 96, 128])
        units_per_layer.append(units)

    dropout_rate = hp.Float("dropout_rate", min_value=0.1, max_value=0.5, step=0.05)
    learning_rate = hp.Choice("learning_rate", [0.001, 0.0005, 0.0001])

    model = build_dynamic_lstm_model(
        time_steps=time_steps,
        n_features=n_features,
        n_classes=5,
        num_layers=num_layers,
        units_per_layer=units_per_layer,
        dropout_rate=dropout_rate,
        learning_rate=learning_rate,
    )
    return model


def main():
    """主程式"""
    # 設定參數（與 tune_setC_only.py 相同）
    PROJECT_NAME = "lstm_stock_tuning_setC_only"
    DIRECTORY = "logs/tuning_logs"

    logger = setup_logger("get_best", log_dir="logs")

    print("=" * 80)
    print("從 Keras Tuner 取得目前最佳模型")
    print("=" * 80)
    print(f"專案名稱: {PROJECT_NAME}")
    print(f"目錄: {DIRECTORY}")
    print()

    # 檢查是否存在試驗資料
    tuning_dir = Path(DIRECTORY) / PROJECT_NAME
    if not tuning_dir.exists():
        print(f"❌ 錯誤：找不到調整紀錄")
        print(f"   路徑: {tuning_dir}")
        print()
        print("請確認：")
        print("  1. 是否已執行過 tune_setC_only.py")
        print("  2. PROJECT_NAME 是否正確")
        sys.exit(1)

    # 計算已完成的試驗數量
    trial_dirs = list(tuning_dir.glob("trial_*"))
    num_trials = len(trial_dirs)

    if num_trials == 0:
        print(f"❌ 錯誤：沒有找到任何試驗數據")
        sys.exit(1)

    print(f"✅ 找到 {num_trials} 個試驗數據")
    print()

    # 載入 Tuner
    print("正在載入 Keras Tuner...")
    try:
        tuner = kt.BayesianOptimization(
            hypermodel=build_model_setC_only,
            objective=kt.Objective("val_loss", direction="min"),
            max_trials=num_trials,  # 使用已有的試驗數量
            directory=DIRECTORY,
            project_name=PROJECT_NAME,
            overwrite=False,  # 不覆蓋
        )
        print("✅ Tuner 載入成功")
        print()
    except Exception as e:
        print(f"❌ Tuner 載入失敗: {e}")
        sys.exit(1)

    # 取得最佳試驗
    print("正在分析試驗結果...")
    try:
        best_trials = tuner.oracle.get_best_trials(num_trials=10)

        if len(best_trials) == 0:
            print("❌ 錯誤：沒有找到完整的試驗結果")
            sys.exit(1)

        print(f"✅ 找到 {len(best_trials)} 個完整的試驗")
        print()

        # 顯示前 5 名
        print("=" * 80)
        print("前 5 名試驗結果")
        print("=" * 80)
        for i, trial in enumerate(best_trials[:5], 1):
            val_acc = trial.metrics.get_best_value("val_accuracy")
            print(f"\nRank {i} - Trial {trial.trial_id}:")
            print(f"  驗證損失: {trial.score:.4f}")
            print(f"  驗證準確度: {val_acc:.2%}")
            print(f"  超參數:")
            for key, value in trial.hyperparameters.values.items():
                print(f"    - {key}: {value}")

        print()
        print("=" * 80)

        # 取得最佳模型
        best_trial = best_trials[0]
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

        print("\n正在建立最佳模型...")
        best_model = tuner.hypermodel.build(best_hps)
        print("✅ 模型建立成功")
        print()

        # 儲存模型
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"best_tuned_setC_partial_{timestamp}"

        output_path = Path("models")
        output_path.mkdir(parents=True, exist_ok=True)
        model_file = output_path / f"{model_name}.h5"

        print(f"正在儲存模型至: {model_file}")
        best_model.save(str(model_file))
        print("✅ 模型已儲存")
        print()

        # 儲存配置
        config_file_path = str(model_file).replace('.h5', '_config.json')

        num_layers = best_hps.get("num_layers")
        save_model_config(
            model_path=str(model_file),
            feature_set_id="Set C",
            time_steps=best_hps.get("time_steps"),
            additional_params={
                "num_layers": num_layers,
                "units_per_layer": [
                    best_hps.get(f"units_layer_{i+1}")
                    for i in range(num_layers)
                ],
                "dropout_rate": best_hps.get("dropout_rate"),
                "learning_rate": best_hps.get("learning_rate"),
                "batch_size": 32,
                "tuner_type": "bayesian",
                "total_trials_completed": num_trials,
                "best_trial_id": best_trial.trial_id,
                "best_val_loss": best_trial.score,
                "best_val_accuracy": best_trial.metrics.get_best_value("val_accuracy"),
                "model_type": "tuned_setC_partial",
                "note": f"從 {num_trials} 個試驗中提取的最佳模型（調整未完成）"
            },
        )
        print(f"✅ 配置已儲存: {config_file_path}")
        print()

        # 摘要
        print("=" * 80)
        print("摘要")
        print("=" * 80)
        print(f"已完成試驗數: {num_trials}")
        print(f"最佳試驗 ID: {best_trial.trial_id}")
        print(f"最佳驗證損失: {best_trial.score:.4f}")
        print(f"最佳驗證準確度: {best_trial.metrics.get_best_value('val_accuracy'):.2%}")
        print()
        print("最佳超參數:")
        print(f"  - 特徵集: Set C")
        print(f"  - 時間窗口: {best_hps.get('time_steps')}")
        print(f"  - LSTM 層數: {best_hps.get('num_layers')}")
        print(f"  - 每層單元數: {[best_hps.get(f'units_layer_{i+1}') for i in range(num_layers)]}")
        print(f"  - Dropout: {best_hps.get('dropout_rate')}")
        print(f"  - 學習率: {best_hps.get('learning_rate')}")
        print()
        print("輸出檔案:")
        print(f"  - 模型: {model_file}")
        print(f"  - 配置: {config_file_path}")
        print()
        print("⚠️ 注意：這是從未完成的調整中提取的模型")
        print("   如果想要更好的結果，可以繼續執行 tune_setC_only.py")
        print("=" * 80)

    except Exception as e:
        print(f"❌ 錯誤: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
