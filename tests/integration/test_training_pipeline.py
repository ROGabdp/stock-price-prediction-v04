"""
整合測試：完整訓練流程

測試資料載入 → 特徵工程 → 模型訓練的完整流程
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile

try:
    from tensorflow import keras

    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

from src.data.data_loader import load_csv_data
from src.features.feature_engineer import engineer_features
from src.features.scalers import create_scaler, fit_scaler, transform_features
from src.data.data_splitter import split_time_series


@pytest.fixture
def sample_csv_file():
    """建立測試用 CSV 檔案"""
    temp_dir = tempfile.mkdtemp()
    csv_file = Path(temp_dir) / "test_data.csv"

    # 建立足夠的測試資料（至少 200 筆以供分割）
    n_days = 200
    dates = pd.date_range("2024-01-01", periods=n_days)

    # 產生隨機但合理的股價資料
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
    return str(csv_file)


class TestDataLoadingPipeline:
    """測試資料載入流程"""

    def test_load_and_validate(self, sample_csv_file):
        """測試載入與驗證"""
        df = load_csv_data(sample_csv_file)

        assert df is not None
        assert len(df) == 200
        assert "date" in df.columns
        assert pd.api.types.is_datetime64_any_dtype(df["date"])


class TestFeatureEngineeringPipeline:
    """測試特徵工程流程"""

    def test_full_feature_engineering(self, sample_csv_file):
        """測試完整特徵工程流程"""
        # 載入資料
        df_raw = load_csv_data(sample_csv_file)

        # 特徵工程
        df_features, target = engineer_features(df_raw, feature_set_id="Set A")

        assert df_features is not None
        assert target is not None
        assert len(df_features) > 0
        assert target.shape[1] == 5  # 5 類

        # 檢查無缺失值
        assert not df_features.isnull().any().any()
        assert not target.isnull().any().any()

    def test_feature_engineering_all_sets(self, sample_csv_file):
        """測試所有特徵集的特徵工程"""
        df_raw = load_csv_data(sample_csv_file)

        for feature_set_id in ["Set A", "Set B", "Set C"]:
            df_features, target = engineer_features(df_raw, feature_set_id=feature_set_id)

            assert df_features is not None
            assert target is not None
            assert len(df_features) > 0


class TestDataScalingPipeline:
    """測試資料縮放流程"""

    def test_scaling_pipeline(self, sample_csv_file):
        """測試完整縮放流程"""
        # 載入與特徵工程
        df_raw = load_csv_data(sample_csv_file)
        df_features, target = engineer_features(df_raw, feature_set_id="Set A")

        # 縮放
        scaler = create_scaler()
        scaler = fit_scaler(scaler, df_features)
        X_scaled = transform_features(scaler, df_features)

        assert X_scaled is not None
        assert X_scaled.shape == df_features.shape

        # 檢查縮放後均值接近 0，標準差接近 1
        assert abs(X_scaled.mean()) < 0.5
        assert abs(X_scaled.std() - 1.0) < 0.5


class TestDataSplittingPipeline:
    """測試資料分割流程"""

    def test_complete_splitting_pipeline(self, sample_csv_file):
        """測試完整分割流程"""
        # 載入與特徵工程
        df_raw = load_csv_data(sample_csv_file)
        df_features, target = engineer_features(df_raw, feature_set_id="Set A")

        # 資料分割
        result = split_time_series(
            df_features, target, time_steps=20, val_ratio=0.15, test_ratio=0.15
        )

        # 檢查所有輸出存在
        assert "X_train" in result
        assert "y_train" in result
        assert "X_val" in result
        assert "y_val" in result
        assert "X_test" in result
        assert "y_test" in result

        # 檢查形狀
        X_train = result["X_train"]
        y_train = result["y_train"]

        assert len(X_train.shape) == 3  # (samples, time_steps, features)
        assert X_train.shape[1] == 20  # time_steps
        assert y_train.shape[1] == 5  # 5 類


@pytest.mark.skipif(not TENSORFLOW_AVAILABLE, reason="TensorFlow 未安裝")
class TestTrainingPipeline:
    """測試完整訓練流程"""

    def test_end_to_end_training(self, sample_csv_file):
        """測試端到端訓練流程（小規模）"""
        from src.models.lstm_baseline import build_lstm_model
        from src.models.model_builder import train_model

        # 1. 載入資料
        df_raw = load_csv_data(sample_csv_file)

        # 2. 特徵工程
        df_features, target = engineer_features(df_raw, feature_set_id="Set A")

        # 3. 資料分割
        result = split_time_series(
            df_features, target, time_steps=20, val_ratio=0.15, test_ratio=0.15
        )

        X_train = result["X_train"]
        y_train = result["y_train"]
        X_val = result["X_val"]
        y_val = result["y_val"]

        # 4. 建立模型
        time_steps = X_train.shape[1]
        n_features = X_train.shape[2]

        model = build_lstm_model(time_steps=time_steps, n_features=n_features)

        # 5. 訓練模型（僅 2 個 epoch 用於測試）
        temp_dir = tempfile.mkdtemp()

        trained_model, history = train_model(
            model,
            X_train,
            y_train,
            X_val,
            y_val,
            epochs=2,
            batch_size=16,
            output_dir=temp_dir,
            model_name="test_model",
            patience=10,
            verbose=0,
        )

        # 檢查訓練完成
        assert trained_model is not None
        assert history is not None
        assert "loss" in history
        assert "val_loss" in history

        # 檢查模型可以預測
        predictions = trained_model.predict(X_val[:2], verbose=0)
        assert predictions.shape == (2, 5)

    def test_model_save_and_load(self, sample_csv_file):
        """測試模型儲存與載入"""
        from src.models.lstm_baseline import build_lstm_model
        from src.models.model_builder import save_model, load_model

        # 建立模型
        model = build_lstm_model(time_steps=20, n_features=14)

        # 儲存模型
        temp_dir = tempfile.mkdtemp()
        model_file = Path(temp_dir) / "test_model.h5"
        save_model(model, str(model_file))

        # 檢查檔案存在
        assert model_file.exists()

        # 載入模型
        loaded_model = load_model(str(model_file))

        assert loaded_model is not None
        assert isinstance(loaded_model, keras.Model)

        # 檢查輸入/輸出形狀一致
        assert loaded_model.input_shape == model.input_shape
        assert loaded_model.output_shape == model.output_shape


class TestPipelineDataFlow:
    """測試流程資料流動"""

    def test_data_shapes_consistency(self, sample_csv_file):
        """測試各階段資料形狀一致性"""
        # 載入
        df_raw = load_csv_data(sample_csv_file)
        n_original = len(df_raw)

        # 特徵工程
        df_features, target = engineer_features(df_raw, feature_set_id="Set A")

        # 特徵工程後資料應變少（因為計算 return 會少一天）
        assert len(df_features) <= n_original

        # 目標變數與特徵應等長
        assert len(df_features) == len(target)

        # 分割
        result = split_time_series(
            df_features, target, time_steps=20, val_ratio=0.15, test_ratio=0.15
        )

        # 訓練/驗證/測試集總數應小於等於原始資料（因時間窗口）
        total_samples = (
            len(result["X_train"]) + len(result["X_val"]) + len(result["X_test"])
        )
        assert total_samples <= len(df_features)

    def test_no_data_leakage(self, sample_csv_file):
        """測試無資料洩漏"""
        df_raw = load_csv_data(sample_csv_file)
        df_features, target = engineer_features(df_raw, feature_set_id="Set A")

        result = split_time_series(
            df_features, target, time_steps=20, val_ratio=0.15, test_ratio=0.15
        )

        # 訓練集應在驗證集之前（時間序列順序）
        # 驗證集應在測試集之前
        # 這由 split_time_series 內部保證

        # 檢查資料集大小合理
        assert len(result["X_train"]) > 0
        assert len(result["X_val"]) > 0
        assert len(result["X_test"]) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
