"""
整合測試：完整預測流程

測試模型載入 → 預測 → 結果輸出的完整流程
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
from src.prediction.predictor import (
    load_trained_model,
    predict_for_date,
    format_prediction_result,
)


@pytest.fixture
def sample_data_and_model():
    """建立測試用資料與模型"""
    if not TENSORFLOW_AVAILABLE:
        pytest.skip("TensorFlow 未安裝")

    from src.models.lstm_baseline import build_lstm_model

    # 建立測試資料
    temp_dir = tempfile.mkdtemp()
    csv_file = Path(temp_dir) / "test_data.csv"

    n_days = 200
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

    # 建立並儲存測試模型
    model = build_lstm_model(time_steps=60, n_features=12)  # Set A 約 12 個特徵
    model_file = Path(temp_dir) / "test_model.h5"
    model.save(str(model_file))

    return {
        "csv_file": str(csv_file),
        "model_file": str(model_file),
        "temp_dir": temp_dir,
    }


@pytest.mark.skipif(not TENSORFLOW_AVAILABLE, reason="TensorFlow 未安裝")
class TestPredictionPipeline:
    """測試完整預測流程"""

    def test_load_model_and_data(self, sample_data_and_model):
        """測試載入模型與資料"""
        model_file = sample_data_and_model["model_file"]
        csv_file = sample_data_and_model["csv_file"]

        # 載入模型
        model = load_trained_model(model_file)
        assert model is not None

        # 載入資料
        df = load_csv_data(csv_file)
        assert df is not None
        assert len(df) == 200

    def test_predict_for_valid_date(self, sample_data_and_model):
        """測試有效日期的預測"""
        model_file = sample_data_and_model["model_file"]
        csv_file = sample_data_and_model["csv_file"]

        # 載入
        model = load_trained_model(model_file)
        df_raw = load_csv_data(csv_file)

        # 選擇一個有效的日期（確保有足夠的歷史資料）
        input_date = "2024-03-01"

        # 執行預測
        result = predict_for_date(
            model=model,
            df_raw=df_raw,
            input_date=input_date,
            feature_set_id="Set A",
            time_steps=60,
        )

        # 檢查預測結果
        assert result is not None
        assert "input_date" in result
        assert "prediction_date" in result
        assert "predicted_class" in result
        assert "confidence" in result
        assert "probability_vector" in result

    def test_prediction_result_structure(self, sample_data_and_model):
        """測試預測結果結構"""
        model_file = sample_data_and_model["model_file"]
        csv_file = sample_data_and_model["csv_file"]

        model = load_trained_model(model_file)
        df_raw = load_csv_data(csv_file)

        result = predict_for_date(
            model=model,
            df_raw=df_raw,
            input_date="2024-03-01",
            feature_set_id="Set A",
            time_steps=60,
        )

        # 檢查必要欄位
        required_keys = [
            "input_date",
            "prediction_date",
            "probability_vector",
            "predicted_class",
            "confidence",
            "predicted_price_range",
            "actual_close_price",
            "timestamp",
        ]

        for key in required_keys:
            assert key in result, f"缺少欄位: {key}"

    def test_prediction_probability_valid(self, sample_data_and_model):
        """測試預測機率有效性"""
        model_file = sample_data_and_model["model_file"]
        csv_file = sample_data_and_model["csv_file"]

        model = load_trained_model(model_file)
        df_raw = load_csv_data(csv_file)

        result = predict_for_date(
            model=model,
            df_raw=df_raw,
            input_date="2024-03-01",
            feature_set_id="Set A",
            time_steps=60,
        )

        prob_vector = result["probability_vector"]

        # 檢查機率向量
        assert len(prob_vector) == 5
        assert all(0 <= p <= 1 for p in prob_vector)
        assert abs(sum(prob_vector) - 1.0) < 0.01  # 總和應為 1

    def test_prediction_class_valid(self, sample_data_and_model):
        """測試預測類別有效性"""
        model_file = sample_data_and_model["model_file"]
        csv_file = sample_data_and_model["csv_file"]

        model = load_trained_model(model_file)
        df_raw = load_csv_data(csv_file)

        result = predict_for_date(
            model=model,
            df_raw=df_raw,
            input_date="2024-03-01",
            feature_set_id="Set A",
            time_steps=60,
        )

        predicted_class = result["predicted_class"]

        # 預測類別應在 0-4 範圍內
        assert 0 <= predicted_class <= 4

        # 預測類別應是機率最大值的索引
        prob_vector = result["probability_vector"]
        assert predicted_class == prob_vector.index(max(prob_vector))

    def test_prediction_confidence(self, sample_data_and_model):
        """測試預測信心度"""
        model_file = sample_data_and_model["model_file"]
        csv_file = sample_data_and_model["csv_file"]

        model = load_trained_model(model_file)
        df_raw = load_csv_data(csv_file)

        result = predict_for_date(
            model=model,
            df_raw=df_raw,
            input_date="2024-03-01",
            feature_set_id="Set A",
            time_steps=60,
        )

        confidence = result["confidence"]

        # 信心度應在 0-1 範圍
        assert 0 <= confidence <= 1

        # 信心度應等於最大機率
        prob_vector = result["probability_vector"]
        assert abs(confidence - max(prob_vector)) < 1e-6


@pytest.mark.skipif(not TENSORFLOW_AVAILABLE, reason="TensorFlow 未安裝")
class TestPredictionEdgeCases:
    """測試預測邊界案例"""

    def test_predict_with_insufficient_data(self, sample_data_and_model):
        """測試資料不足時的預測"""
        model_file = sample_data_and_model["model_file"]
        csv_file = sample_data_and_model["csv_file"]

        model = load_trained_model(model_file)
        df_raw = load_csv_data(csv_file)

        # 嘗試預測第一天（資料不足）
        with pytest.raises(ValueError, match="資料不足"):
            predict_for_date(
                model=model,
                df_raw=df_raw,
                input_date="2024-01-01",
                feature_set_id="Set A",
                time_steps=60,
            )

    def test_predict_with_invalid_date(self, sample_data_and_model):
        """測試無效日期的預測"""
        model_file = sample_data_and_model["model_file"]
        csv_file = sample_data_and_model["csv_file"]

        model = load_trained_model(model_file)
        df_raw = load_csv_data(csv_file)

        # 不存在的日期
        with pytest.raises(ValueError, match="不存在於歷史資料"):
            predict_for_date(
                model=model,
                df_raw=df_raw,
                input_date="2025-12-31",
                feature_set_id="Set A",
                time_steps=60,
            )


@pytest.mark.skipif(not TENSORFLOW_AVAILABLE, reason="TensorFlow 未安裝")
class TestResultFormatting:
    """測試結果格式化"""

    def test_format_prediction_output(self, sample_data_and_model):
        """測試預測結果格式化"""
        model_file = sample_data_and_model["model_file"]
        csv_file = sample_data_and_model["csv_file"]

        model = load_trained_model(model_file)
        df_raw = load_csv_data(csv_file)

        result = predict_for_date(
            model=model,
            df_raw=df_raw,
            input_date="2024-03-01",
            feature_set_id="Set A",
            time_steps=60,
        )

        # 格式化結果
        formatted = format_prediction_result(result)

        assert formatted is not None
        assert isinstance(formatted, str)
        assert len(formatted) > 0

        # 檢查包含關鍵資訊
        assert result["input_date"] in formatted
        assert str(result["predicted_class"]) in formatted

    def test_formatted_output_readability(self, sample_data_and_model):
        """測試格式化輸出可讀性"""
        model_file = sample_data_and_model["model_file"]
        csv_file = sample_data_and_model["csv_file"]

        model = load_trained_model(model_file)
        df_raw = load_csv_data(csv_file)

        result = predict_for_date(
            model=model,
            df_raw=df_raw,
            input_date="2024-03-01",
            feature_set_id="Set A",
            time_steps=60,
        )

        formatted = format_prediction_result(result)

        # 檢查包含中文標籤
        assert any(
            label in formatted
            for label in ["極度下跌", "溫和下跌", "區間震盪", "溫和上漲", "極度上漲"]
        )

        # 檢查包含日期
        assert "2024" in formatted

        # 檢查包含機率分佈
        assert "機率分佈" in formatted or "機率" in formatted


@pytest.mark.skipif(not TENSORFLOW_AVAILABLE, reason="TensorFlow 未安裝")
class TestPredictionPerformance:
    """測試預測效能"""

    def test_prediction_response_time(self, sample_data_and_model):
        """測試預測響應時間"""
        import time

        model_file = sample_data_and_model["model_file"]
        csv_file = sample_data_and_model["csv_file"]

        model = load_trained_model(model_file)
        df_raw = load_csv_data(csv_file)

        start_time = time.time()

        result = predict_for_date(
            model=model,
            df_raw=df_raw,
            input_date="2024-03-01",
            feature_set_id="Set A",
            time_steps=60,
        )

        elapsed_time = time.time() - start_time

        # 預測應在 10 秒內完成
        assert elapsed_time < 10.0

        # 通常應該更快（< 5 秒）
        assert elapsed_time < 5.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
