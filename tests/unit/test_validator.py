"""
驗證模組單元測試

測試歷史資料驗證邏輯的正確性
"""

import pytest
import pandas as pd
import numpy as np
from src.visualization.validator import (
    calculate_actual_class_5,
    map_5class_to_3class,
    get_actual_data,
)


class TestCalculateActualClass5:
    """測試 calculate_actual_class_5 函式"""

    def test_extreme_downtrend(self):
        """測試極度下跌 (< -5.0%)"""
        assert calculate_actual_class_5(-0.06) == 0  # -6.0%
        assert calculate_actual_class_5(-0.10) == 0  # -10.0%
        assert calculate_actual_class_5(-0.051) == 0  # -5.1%

    def test_mild_downtrend(self):
        """測試溫和下跌 (-5.0% ≤ R < -2.5%)"""
        assert calculate_actual_class_5(-0.05) == 1  # -5.0% (邊界)
        assert calculate_actual_class_5(-0.0375) == 1  # -3.75%
        assert calculate_actual_class_5(-0.03) == 1  # -3.0%

    def test_neutral(self):
        """測試區間震盪 (-2.5% ≤ R ≤ +2.5%)"""
        assert calculate_actual_class_5(-0.025) == 2  # -2.5% (下邊界)
        assert calculate_actual_class_5(0.0) == 2  # 0%
        assert calculate_actual_class_5(0.01) == 2  # +1.0%
        assert calculate_actual_class_5(0.025) == 2  # +2.5% (上邊界)

    def test_mild_uptrend(self):
        """測試溫和上漲 (+2.5% < R ≤ +5.0%)"""
        assert calculate_actual_class_5(0.03) == 3  # +3.0%
        assert calculate_actual_class_5(0.0375) == 3  # +3.75%
        assert calculate_actual_class_5(0.05) == 3  # +5.0% (邊界)

    def test_extreme_uptrend(self):
        """測試極度上漲 (> +5.0%)"""
        assert calculate_actual_class_5(0.051) == 4  # +5.1%
        assert calculate_actual_class_5(0.10) == 4  # +10.0%
        assert calculate_actual_class_5(0.15) == 4  # +15.0%

    def test_boundary_values(self):
        """測試邊界值處理"""
        # -5.0% 邊界
        assert calculate_actual_class_5(-0.05) == 1  # 屬於溫和下跌
        assert calculate_actual_class_5(-0.050001) == 0  # 屬於極度下跌

        # -2.5% 邊界
        assert calculate_actual_class_5(-0.025) == 2  # 屬於震盪
        assert calculate_actual_class_5(-0.0250001) != 2  # 不屬於震盪

        # +2.5% 邊界
        assert calculate_actual_class_5(0.025) == 2  # 屬於震盪
        assert calculate_actual_class_5(0.0250001) != 2  # 不屬於震盪

        # +5.0% 邊界
        assert calculate_actual_class_5(0.05) == 3  # 屬於溫和上漲
        assert calculate_actual_class_5(0.050001) == 4  # 屬於極度上漲

    def test_small_changes(self):
        """測試微小漲跌幅"""
        assert calculate_actual_class_5(0.001) == 2  # +0.1%
        assert calculate_actual_class_5(-0.001) == 2  # -0.1%
        assert calculate_actual_class_5(0.0001) == 2  # +0.01%

    def test_large_changes(self):
        """測試極端漲跌幅"""
        assert calculate_actual_class_5(-0.20) == 0  # -20%
        assert calculate_actual_class_5(0.30) == 4  # +30%


class TestMap5ClassTo3Class:
    """測試 map_5class_to_3class 函式"""

    def test_extreme_downtrend_to_bearish(self):
        """測試 0 (極度下跌) → 看跌"""
        assert map_5class_to_3class(0) == "看跌"

    def test_mild_downtrend_to_bearish(self):
        """測試 1 (溫和下跌) → 看跌"""
        assert map_5class_to_3class(1) == "看跌"

    def test_neutral_to_neutral(self):
        """測試 2 (震盪) → 震盪"""
        assert map_5class_to_3class(2) == "震盪"

    def test_mild_uptrend_to_bullish(self):
        """測試 3 (溫和上漲) → 看漲"""
        assert map_5class_to_3class(3) == "看漲"

    def test_extreme_uptrend_to_bullish(self):
        """測試 4 (極度上漲) → 看漲"""
        assert map_5class_to_3class(4) == "看漲"

    def test_invalid_class_negative(self):
        """測試錯誤處理: 負數類別"""
        with pytest.raises(ValueError, match="必須在 0-4 範圍內"):
            map_5class_to_3class(-1)

    def test_invalid_class_too_large(self):
        """測試錯誤處理: 超出範圍"""
        with pytest.raises(ValueError, match="必須在 0-4 範圍內"):
            map_5class_to_3class(5)


class TestGetActualData:
    """測試 get_actual_data 函式"""

    @pytest.fixture
    def sample_dataframe(self):
        """建立測試用 DataFrame"""
        data = {
            "date": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-15",
                    "2024-02-01",
                    "2024-02-15",
                    "2024-03-01",
                ]
            ),
            "close": [10000.0, 10050.0, 10200.0, 10500.0, 10800.0, 11000.0],
        }
        return pd.DataFrame(data)

    def test_actual_data_exists_uptrend(self, sample_dataframe):
        """測試實際資料存在且為上漲"""
        result = get_actual_data(
            sample_dataframe,
            input_date="2024-01-15",
            prediction_date="2024-02-15",
            predicted_class_5=3,  # 預測溫和上漲
            predicted_class_3="看漲",
        )

        assert result is not None
        assert result["actual_close"] == 10800.0
        # 漲跌幅: (10800 - 10200) / 10200 = 0.0588 = +5.88%
        assert result["actual_change_pct"] == pytest.approx(0.0588, abs=0.0001)
        assert result["actual_class_5"] == 4  # 極度上漲 (> +5%)
        assert result["actual_class_3"] == "看漲"
        assert result["is_correct_5class"] == False  # 預測3但實際4
        assert result["is_correct_3class"] == True  # 都是看漲

    def test_actual_data_exists_downtrend(self, sample_dataframe):
        """測試實際資料存在且為下跌"""
        result = get_actual_data(
            sample_dataframe,
            input_date="2024-02-15",
            prediction_date="2024-01-15",  # 反向查詢
            predicted_class_5=0,  # 預測極度下跌
            predicted_class_3="看跌",
        )

        assert result is not None
        assert result["actual_close"] == 10200.0
        # 漲跌幅: (10200 - 10800) / 10800 = -0.0556 = -5.56%
        assert result["actual_change_pct"] == pytest.approx(-0.0556, abs=0.0001)
        assert result["actual_class_5"] == 0  # 極度下跌 (< -5%)
        assert result["actual_class_3"] == "看跌"
        assert result["is_correct_5class"] == True
        assert result["is_correct_3class"] == True

    def test_actual_data_not_exists(self, sample_dataframe):
        """測試實際資料不存在"""
        result = get_actual_data(
            sample_dataframe,
            input_date="2024-01-15",
            prediction_date="2024-12-31",  # 未來日期
            predicted_class_5=3,
            predicted_class_3="看漲",
        )

        assert result is None

    def test_input_date_not_exists(self, sample_dataframe):
        """測試輸入日期不存在"""
        result = get_actual_data(
            sample_dataframe,
            input_date="2023-01-01",  # 不存在
            prediction_date="2024-02-15",
            predicted_class_5=3,
            predicted_class_3="看漲",
        )

        assert result is None

    def test_correct_5class_prediction(self, sample_dataframe):
        """測試5分類預測正確"""
        # 構造數據: 2024-01-15 (10200) → 2024-02-01 (10500)
        # 漲跌幅: +2.94% (屬於溫和上漲 class 3)
        result = get_actual_data(
            sample_dataframe,
            input_date="2024-01-15",
            prediction_date="2024-02-01",
            predicted_class_5=3,  # 預測溫和上漲
            predicted_class_3="看漲",
        )

        assert result is not None
        assert result["actual_change_pct"] == pytest.approx(0.0294, abs=0.0001)
        assert result["actual_class_5"] == 3  # 溫和上漲
        assert result["is_correct_5class"] == True

    def test_incorrect_5class_but_correct_3class(self, sample_dataframe):
        """測試5分類預測錯誤但3分類正確"""
        # 2024-01-15 (10200) → 2024-03-01 (11000)
        # 漲跌幅: +7.84% (屬於極度上漲 class 4)
        result = get_actual_data(
            sample_dataframe,
            input_date="2024-01-15",
            prediction_date="2024-03-01",
            predicted_class_5=3,  # 預測溫和上漲 (錯誤)
            predicted_class_3="看漲",  # 預測看漲 (正確)
        )

        assert result is not None
        assert result["actual_class_5"] == 4  # 實際極度上漲
        assert result["is_correct_5class"] == False
        assert result["is_correct_3class"] == True  # 都是看漲

    def test_dataframe_with_string_dates(self):
        """測試 DataFrame 日期為字串型別"""
        df = pd.DataFrame(
            {"date": ["2024-01-01", "2024-02-01"], "close": [10000.0, 10500.0]}
        )

        result = get_actual_data(
            df,
            input_date="2024-01-01",
            prediction_date="2024-02-01",
            predicted_class_5=3,
            predicted_class_3="看漲",
        )

        assert result is not None
        assert result["actual_close"] == 10500.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
