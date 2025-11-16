"""
單元測試：預測模組

測試預測邏輯、結果格式化
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

from src.prediction.predictor import (
    format_prediction_result,
    CLASS_MAPPING,
)


class TestClassMapping:
    """測試類別映射定義"""

    def test_class_mapping_exists(self):
        """測試類別映射存在"""
        assert CLASS_MAPPING is not None
        assert len(CLASS_MAPPING) == 5

    def test_class_mapping_keys(self):
        """測試類別映射包含所有類別"""
        for i in range(5):
            assert i in CLASS_MAPPING

    def test_class_mapping_structure(self):
        """測試類別映射結構正確"""
        for class_id, class_info in CLASS_MAPPING.items():
            assert "label" in class_info
            assert "range" in class_info
            assert "min_change" in class_info or class_info["min_change"] is None
            assert "max_change" in class_info or class_info["max_change"] is None

    def test_class_0_extreme_down(self):
        """測試類別 0（極度下跌）定義"""
        class_info = CLASS_MAPPING[0]
        assert "極度下跌" in class_info["label"]
        assert class_info["max_change"] == -0.05

    def test_class_4_extreme_up(self):
        """測試類別 4（極度上漲）定義"""
        class_info = CLASS_MAPPING[4]
        assert "極度上漲" in class_info["label"]
        assert class_info["min_change"] == 0.05


class TestFormatPredictionResult:
    """測試預測結果格式化"""

    def setup_method(self):
        """建立測試用預測結果"""
        self.sample_result = {
            "input_date": "2024-01-15",
            "input_close_price": 15800.0,
            "prediction_date": "2024-02-14",
            "probability_vector": [0.023, 0.087, 0.124, 0.685, 0.081],
            "predicted_class": 3,
            "predicted_class_label": "溫和上漲",
            "predicted_class_range": "+2.5% < R ≤ +5.0%",
            "confidence": 0.685,
            "predicted_price_range": {"min": 16195.0, "max": 16590.0},
            "actual_close_price": 16120.0,
            "timestamp": "2025-11-15 14:35:21",
        }

    def test_format_returns_string(self):
        """測試格式化回傳字串"""
        formatted = format_prediction_result(self.sample_result)
        assert isinstance(formatted, str)
        assert len(formatted) > 0

    def test_format_contains_dates(self):
        """測試格式化結果包含日期"""
        formatted = format_prediction_result(self.sample_result)
        assert "2024-01-15" in formatted
        assert "2024-02-14" in formatted

    def test_format_contains_prediction_class(self):
        """測試格式化結果包含預測類別"""
        formatted = format_prediction_result(self.sample_result)
        assert "溫和上漲" in formatted
        assert "3" in formatted

    def test_format_contains_confidence(self):
        """測試格式化結果包含信心度"""
        formatted = format_prediction_result(self.sample_result)
        # 應該顯示百分比形式
        assert "68" in formatted or "69" in formatted  # 68.5% 可能顯示為 68% 或 69%

    def test_format_contains_price_range(self):
        """測試格式化結果包含價格區間"""
        formatted = format_prediction_result(self.sample_result)
        assert "16195" in formatted or "16,195" in formatted
        assert "16590" in formatted or "16,590" in formatted

    def test_format_with_actual_price(self):
        """測試格式化結果包含實際價格比較"""
        formatted = format_prediction_result(self.sample_result)
        assert "16120" in formatted or "16,120" in formatted
        # 應該顯示是否在預測區間內
        assert "✅" in formatted or "✓" in formatted or "落在" in formatted

    def test_format_without_actual_price(self):
        """測試無實際價格時的格式化"""
        result = self.sample_result.copy()
        result["actual_close_price"] = None

        formatted = format_prediction_result(result)
        assert "資料尚未公布" in formatted or "未公布" in formatted or "無" in formatted

    def test_format_probability_distribution(self):
        """測試格式化結果包含機率分佈"""
        formatted = format_prediction_result(self.sample_result)

        # 應包含所有類別的標籤
        assert "極度下跌" in formatted
        assert "溫和下跌" in formatted
        assert "區間震盪" in formatted
        assert "溫和上漲" in formatted
        assert "極度上漲" in formatted


class TestPredictionValidation:
    """測試預測結果驗證"""

    def test_probability_sum_to_one(self):
        """測試機率總和為 1"""
        prob_vector = np.array([0.1, 0.2, 0.3, 0.25, 0.15])

        # 機率總和應接近 1
        assert abs(prob_vector.sum() - 1.0) < 1e-6

    def test_probability_all_positive(self):
        """測試機率均為正值"""
        prob_vector = np.array([0.1, 0.2, 0.3, 0.25, 0.15])

        # 所有機率應 >= 0
        assert (prob_vector >= 0).all()

        # 所有機率應 <= 1
        assert (prob_vector <= 1).all()

    def test_predicted_class_is_argmax(self):
        """測試預測類別是機率最大值的索引"""
        prob_vector = np.array([0.023, 0.087, 0.124, 0.685, 0.081])

        predicted_class = np.argmax(prob_vector)
        assert predicted_class == 3

    def test_confidence_is_max_probability(self):
        """測試信心度是最大機率"""
        prob_vector = np.array([0.023, 0.087, 0.124, 0.685, 0.081])

        confidence = np.max(prob_vector)
        assert abs(confidence - 0.685) < 1e-6


class TestPriceRangeCalculation:
    """測試價格區間計算"""

    def test_class_0_price_range(self):
        """測試類別 0（極度下跌）的價格區間"""
        input_price = 10000.0
        class_info = CLASS_MAPPING[0]

        # R < -5% → 最高價 = 10000 * (1 - 0.05) = 9500
        max_price = input_price * (1 + class_info["max_change"])
        assert abs(max_price - 9500.0) < 1e-6

    def test_class_2_price_range(self):
        """測試類別 2（區間震盪）的價格區間"""
        input_price = 10000.0
        class_info = CLASS_MAPPING[2]

        # -2.5% ≤ R ≤ +2.5%
        min_price = input_price * (1 + class_info["min_change"])
        max_price = input_price * (1 + class_info["max_change"])

        assert abs(min_price - 9750.0) < 1e-6
        assert abs(max_price - 10250.0) < 1e-6

    def test_class_4_price_range(self):
        """測試類別 4（極度上漲）的價格區間"""
        input_price = 10000.0
        class_info = CLASS_MAPPING[4]

        # R > +5% → 最低價 = 10000 * (1 + 0.05) = 10500
        min_price = input_price * (1 + class_info["min_change"])
        assert abs(min_price - 10500.0) < 1e-6


@pytest.mark.skipif(not TENSORFLOW_AVAILABLE, reason="TensorFlow 未安裝")
class TestLoadTrainedModel:
    """測試模型載入功能"""

    def setup_method(self):
        """建立臨時模型檔案"""
        from src.models.lstm_baseline import build_lstm_model

        self.temp_dir = tempfile.mkdtemp()
        self.model_file = Path(self.temp_dir) / "test_model.h5"

        # 建立並儲存測試模型
        model = build_lstm_model(time_steps=60, n_features=14)
        model.save(str(self.model_file))

    def test_load_model_success(self):
        """測試成功載入模型"""
        from src.prediction.predictor import load_trained_model

        model = load_trained_model(str(self.model_file))

        assert model is not None
        assert isinstance(model, keras.Model)

    def test_load_nonexistent_model(self):
        """測試載入不存在的模型"""
        from src.prediction.predictor import load_trained_model

        with pytest.raises(FileNotFoundError):
            load_trained_model("nonexistent_model.h5")


class TestDateValidation:
    """測試日期驗證"""

    def test_valid_date_format(self):
        """測試有效的日期格式"""
        valid_dates = ["2024-01-15", "2024/1/15", "2024-12-31"]

        for date_str in valid_dates:
            try:
                dt = pd.to_datetime(date_str)
                assert dt is not None
            except Exception:
                pytest.fail(f"日期格式應該有效: {date_str}")

    def test_invalid_date_format(self):
        """測試無效的日期格式"""
        invalid_dates = ["2024-13-01", "2024-02-30", "invalid"]

        for date_str in invalid_dates:
            with pytest.raises((ValueError, pd.errors.ParserError)):
                pd.to_datetime(date_str, errors="raise")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
