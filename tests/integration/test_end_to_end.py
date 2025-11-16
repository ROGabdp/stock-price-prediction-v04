"""
端到端整合測試：完整工作流程

測試訓練 → 調參 → 預測的完整工作流程
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile

try:
    from tensorflow import keras
    import keras_tuner as kt

    TENSORFLOW_AVAILABLE = True
    KERAS_TUNER_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    KERAS_TUNER_AVAILABLE = False

from src.data.data_loader import load_csv_data
from src.features.feature_engineer import engineer_features
from src.data.data_splitter import split_time_series


@pytest.fixture
def e2e_test_data():
    """建立端到端測試用資料"""
    temp_dir = tempfile.mkdtemp()
    csv_file = Path(temp_dir) / "e2e_test_data.csv"

    # 建立較大的測試資料集（用於完整流程測試）
    n_days = 300
    dates = pd.date_range("2024-01-01", periods=n_days)

    np.random.seed(42)
    base_price = 15000
    price_changes = np.random.randn(n_days).cumsum() * 50
    closes = base_price + price_changes

    df = pd.DataFrame(
        {
            "date": dates.strftime("%Y/%m/%d"),
            "open": closes + np.random.uniform(-50, 50, n_days),
            "high": closes + np.random.uniform(50, 150, n_days),
            "low": closes - np.random.uniform(50, 150, n_days),
            "close": closes,
            "volume": np.random.uniform(1000, 2000, n_days),
            "SMA5": closes + np.random.uniform(-20, 20, n_days),
            "SMA10": closes + np.random.uniform(-30, 30, n_days),
            "SMA20": closes + np.random.uniform(-40, 40, n_days),
            "SMA60": closes + np.random.uniform(-50, 50, n_days),
            "SMA120": closes + np.random.uniform(-60, 60, n_days),
            "SMA240": closes + np.random.uniform(-70, 70, n_days),
            "MA5": closes + np.random.uniform(-20, 20, n_days),
            "MA10": closes + np.random.uniform(-30, 30, n_days),
            "DIF12-26": np.random.uniform(-100, 100, n_days),
            "MACD9": np.random.uniform(-100, 100, n_days),
            "OSC": np.random.uniform(-50, 50, n_days),
            "K(9,3)": np.random.uniform(0, 1, n_days),
            "D(9,3)": np.random.uniform(0, 1, n_days),
            "net buy sell": np.random.uniform(-1e6, 1e6, n_days),
            "cumulative net buy sell": np.random.uniform(-1e7, 1e7, n_days),
            "buy": np.random.uniform(0, 5e6, n_days),
            "sell": np.random.uniform(0, 5e6, n_days),
        }
    )

    df.to_csv(csv_file, index=False)

    return {
        "csv_file": str(csv_file),
        "temp_dir": temp_dir,
    }


@pytest.mark.skipif(not TENSORFLOW_AVAILABLE, reason="TensorFlow 未安裝")
class TestEndToEndWorkflow:
    """測試端到端工作流程"""

    def test_train_evaluate_predict_workflow(self, e2e_test_data):
        """測試訓練 → 評估 → 預測完整流程"""
        from src.models.lstm_baseline import build_lstm_model
        from src.models.model_builder import train_model, evaluate_model
        from src.prediction.predictor import predict_for_date, format_prediction_result

        csv_file = e2e_test_data["csv_file"]
        temp_dir = e2e_test_data["temp_dir"]

        # 步驟 1: 載入資料
        df_raw = load_csv_data(csv_file)
        assert len(df_raw) == 300

        # 步驟 2: 特徵工程
        df_features, target = engineer_features(df_raw, feature_set_id="Set A")
        assert df_features is not None

        # 步驟 3: 資料分割
        result = split_time_series(
            df_features, target, time_steps=60, val_ratio=0.15, test_ratio=0.15
        )

        X_train = result["X_train"]
        y_train = result["y_train"]
        X_val = result["X_val"]
        y_val = result["y_val"]
        X_test = result["X_test"]
        y_test = result["y_test"]

        # 步驟 4: 建立模型
        model = build_lstm_model(
            time_steps=X_train.shape[1], n_features=X_train.shape[2]
        )

        # 步驟 5: 訓練模型
        trained_model, history = train_model(
            model,
            X_train,
            y_train,
            X_val,
            y_val,
            epochs=2,
            batch_size=16,
            output_dir=temp_dir,
            model_name="e2e_test_model",
            verbose=0,
        )

        assert trained_model is not None
        assert "loss" in history

        # 步驟 6: 評估模型
        test_loss, test_acc = evaluate_model(trained_model, X_test, y_test)
        assert test_loss > 0
        assert 0 <= test_acc <= 1

        # 步驟 7: 儲存模型
        model_file = Path(temp_dir) / "e2e_test_model.h5"
        assert model_file.exists()

        # 步驟 8: 載入模型並預測
        from src.prediction.predictor import load_trained_model

        loaded_model = load_trained_model(str(model_file))

        prediction_result = predict_for_date(
            model=loaded_model,
            df_raw=df_raw,
            input_date="2024-08-01",
            feature_set_id="Set A",
            time_steps=60,
        )

        assert prediction_result is not None
        assert "predicted_class" in prediction_result

        # 步驟 9: 格式化輸出
        formatted = format_prediction_result(prediction_result)
        assert isinstance(formatted, str)
        assert len(formatted) > 0


@pytest.mark.skipif(
    not (TENSORFLOW_AVAILABLE and KERAS_TUNER_AVAILABLE),
    reason="TensorFlow 或 Keras Tuner 未安裝",
)
class TestTuningWorkflow:
    """測試超參數調整工作流程"""

    def test_tuning_to_prediction_workflow(self, e2e_test_data):
        """測試調參 → 預測完整流程（簡化版）"""
        from src.tuning.hyperparameter_tuner import create_tuner, run_tuning, get_best_model
        from src.prediction.predictor import predict_for_date

        csv_file = e2e_test_data["csv_file"]
        temp_dir = e2e_test_data["temp_dir"]

        # 載入與處理資料
        df_raw = load_csv_data(csv_file)
        df_features, target = engineer_features(df_raw, feature_set_id="Set C")

        result = split_time_series(
            df_features, target, time_steps=60, val_ratio=0.15, test_ratio=0.15
        )

        X_train = result["X_train"]
        y_train = result["y_train"]
        X_val = result["X_val"]
        y_val = result["y_val"]

        # 建立 Tuner（僅 2 次試驗用於測試）
        tuner = create_tuner(
            tuner_type="random",
            max_trials=2,
            project_name="e2e_test_tuning",
            directory=temp_dir,
            overwrite=True,
        )

        assert tuner is not None

        # 執行調參（非常少的 epoch）
        tuner, stats = run_tuning(
            tuner,
            X_train,
            y_train,
            X_val,
            y_val,
            epochs_per_trial=2,
            verbose=0,
        )

        assert stats is not None
        assert "best_val_loss" in stats

        # 取得最佳模型
        best_model = get_best_model(tuner)
        assert best_model is not None

        # 儲存最佳模型
        best_model_file = Path(temp_dir) / "best_tuned_model.h5"
        best_model.save(str(best_model_file))

        # 使用最佳模型進行預測
        from src.prediction.predictor import load_trained_model

        loaded_model = load_trained_model(str(best_model_file))

        # 預測（注意：需使用調參時的特徵集）
        prediction_result = predict_for_date(
            model=loaded_model,
            df_raw=df_raw,
            input_date="2024-08-01",
            feature_set_id="Set C",  # 使用 Set C
            time_steps=60,
        )

        assert prediction_result is not None


@pytest.mark.skipif(not TENSORFLOW_AVAILABLE, reason="TensorFlow 未安裝")
class TestMultipleFeatureSetsWorkflow:
    """測試多特徵集工作流程"""

    def test_all_feature_sets_workflow(self, e2e_test_data):
        """測試所有特徵集的完整流程"""
        from src.models.lstm_baseline import build_lstm_model
        from src.models.model_builder import train_model

        csv_file = e2e_test_data["csv_file"]
        temp_dir = e2e_test_data["temp_dir"]
        df_raw = load_csv_data(csv_file)

        for feature_set_id in ["Set A", "Set B", "Set C"]:
            # 特徵工程
            df_features, target = engineer_features(df_raw, feature_set_id=feature_set_id)

            # 資料分割
            result = split_time_series(
                df_features, target, time_steps=60, val_ratio=0.15, test_ratio=0.15
            )

            X_train = result["X_train"]
            y_train = result["y_train"]
            X_val = result["X_val"]
            y_val = result["y_val"]

            # 建立與訓練模型
            model = build_lstm_model(
                time_steps=X_train.shape[1], n_features=X_train.shape[2]
            )

            trained_model, history = train_model(
                model,
                X_train,
                y_train,
                X_val,
                y_val,
                epochs=2,
                batch_size=16,
                output_dir=temp_dir,
                model_name=f"model_{feature_set_id}",
                verbose=0,
            )

            assert trained_model is not None
            assert len(history["loss"]) > 0


@pytest.mark.skipif(not TENSORFLOW_AVAILABLE, reason="TensorFlow 未安裝")
class TestWorkflowErrorHandling:
    """測試工作流程錯誤處理"""

    def test_invalid_date_handling(self, e2e_test_data):
        """測試無效日期的錯誤處理"""
        from src.models.lstm_baseline import build_lstm_model
        from src.prediction.predictor import predict_for_date

        csv_file = e2e_test_data["csv_file"]
        temp_dir = e2e_test_data["temp_dir"]

        # 建立並儲存模型
        model = build_lstm_model(time_steps=60, n_features=12)
        model_file = Path(temp_dir) / "test_model.h5"
        model.save(str(model_file))

        # 載入資料
        df_raw = load_csv_data(csv_file)

        from src.prediction.predictor import load_trained_model

        loaded_model = load_trained_model(str(model_file))

        # 測試無效日期
        with pytest.raises(ValueError):
            predict_for_date(
                model=loaded_model,
                df_raw=df_raw,
                input_date="2099-12-31",  # 未來日期
                feature_set_id="Set A",
                time_steps=60,
            )

    def test_insufficient_data_handling(self, e2e_test_data):
        """測試資料不足的錯誤處理"""
        from src.models.lstm_baseline import build_lstm_model
        from src.prediction.predictor import predict_for_date, load_trained_model

        csv_file = e2e_test_data["csv_file"]
        temp_dir = e2e_test_data["temp_dir"]

        # 建立模型
        model = build_lstm_model(time_steps=60, n_features=12)
        model_file = Path(temp_dir) / "test_model.h5"
        model.save(str(model_file))

        df_raw = load_csv_data(csv_file)
        loaded_model = load_trained_model(str(model_file))

        # 測試第一天（資料不足 60 天）
        with pytest.raises(ValueError, match="資料不足"):
            predict_for_date(
                model=loaded_model,
                df_raw=df_raw,
                input_date="2024-01-01",
                feature_set_id="Set A",
                time_steps=60,
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
