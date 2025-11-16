"""
聚合模組單元測試

測試3分類聚合邏輯的正確性
"""

import pytest
import numpy as np
from src.visualization.aggregator import (
    aggregate_to_3_categories,
    get_predicted_class_3,
)


class TestAggregateToThe3Categories:
    """測試 aggregate_to_3_categories 函式"""

    def test_normal_aggregation(self):
        """測試正常聚合情況"""
        prob_5 = np.array([0.05, 0.15, 0.25, 0.35, 0.20])
        result = aggregate_to_3_categories(prob_5)

        assert "看跌" in result
        assert "震盪" in result
        assert "看漲" in result

        # 驗證聚合邏輯
        assert result["看跌"] == pytest.approx(0.20, abs=1e-10)  # 0.05 + 0.15
        assert result["震盪"] == pytest.approx(0.25, abs=1e-10)  # 0.25
        assert result["看漲"] == pytest.approx(0.55, abs=1e-10)  # 0.35 + 0.20

        # 驗證總和為1.0
        total = result["看跌"] + result["震盪"] + result["看漲"]
        assert abs(total - 1.0) < 1e-10

    def test_extreme_downtrend_100_percent(self):
        """測試邊界情況: 極度下跌100%"""
        prob_5 = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
        result = aggregate_to_3_categories(prob_5)

        assert result["看跌"] == pytest.approx(1.0, abs=1e-10)
        assert result["震盪"] == pytest.approx(0.0, abs=1e-10)
        assert result["看漲"] == pytest.approx(0.0, abs=1e-10)

    def test_extreme_uptrend_100_percent(self):
        """測試邊界情況: 極度上漲100%"""
        prob_5 = np.array([0.0, 0.0, 0.0, 0.0, 1.0])
        result = aggregate_to_3_categories(prob_5)

        assert result["看跌"] == pytest.approx(0.0, abs=1e-10)
        assert result["震盪"] == pytest.approx(0.0, abs=1e-10)
        assert result["看漲"] == pytest.approx(1.0, abs=1e-10)

    def test_neutral_100_percent(self):
        """測試邊界情況: 震盪100%"""
        prob_5 = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
        result = aggregate_to_3_categories(prob_5)

        assert result["看跌"] == pytest.approx(0.0, abs=1e-10)
        assert result["震盪"] == pytest.approx(1.0, abs=1e-10)
        assert result["看漲"] == pytest.approx(0.0, abs=1e-10)

    def test_uniform_distribution(self):
        """測試邊界情況: 均勻分佈"""
        prob_5 = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        result = aggregate_to_3_categories(prob_5)

        assert result["看跌"] == pytest.approx(0.4, abs=1e-10)
        assert result["震盪"] == pytest.approx(0.2, abs=1e-10)
        assert result["看漲"] == pytest.approx(0.4, abs=1e-10)

    def test_invalid_type_list(self):
        """測試錯誤處理: 輸入為list而非numpy array"""
        with pytest.raises(TypeError, match="必須是 numpy.ndarray 型別"):
            aggregate_to_3_categories([0.2, 0.2, 0.2, 0.2, 0.2])

    def test_invalid_shape(self):
        """測試錯誤處理: 形狀錯誤"""
        with pytest.raises(ValueError, match="形狀必須為 \\(5,\\)"):
            aggregate_to_3_categories(np.array([0.5, 0.5]))

    def test_invalid_sum_not_one(self):
        """測試錯誤處理: 機率總和不為1.0"""
        prob_5 = np.array([0.1, 0.1, 0.1, 0.1, 0.1])  # 總和=0.5
        with pytest.raises(AssertionError, match="5分類機率總和必須為1.0"):
            aggregate_to_3_categories(prob_5)

    def test_sum_precision(self):
        """測試機率總和精確度"""
        # 測試浮點數精度問題
        prob_5 = np.array([0.1, 0.2, 0.3, 0.25, 0.15])
        result = aggregate_to_3_categories(prob_5)

        # 驗證總和誤差在允許範圍內
        total = result["看跌"] + result["震盪"] + result["看漲"]
        assert abs(total - 1.0) < 1e-10


class TestGetPredictedClass3:
    """測試 get_predicted_class_3 函式"""

    def test_bearish_highest(self):
        """測試看跌機率最高"""
        prob_3 = {"看跌": 0.60, "震盪": 0.25, "看漲": 0.15}
        result = get_predicted_class_3(prob_3)
        assert result == "看跌"

    def test_neutral_highest(self):
        """測試震盪機率最高"""
        prob_3 = {"看跌": 0.20, "震盪": 0.55, "看漲": 0.25}
        result = get_predicted_class_3(prob_3)
        assert result == "震盪"

    def test_bullish_highest(self):
        """測試看漲機率最高"""
        prob_3 = {"看跌": 0.20, "震盪": 0.25, "看漲": 0.55}
        result = get_predicted_class_3(prob_3)
        assert result == "看漲"

    def test_tie_bearish_neutral(self):
        """測試相同機率: 看跌=震盪 (優先級: 看跌 > 震盪)"""
        prob_3 = {"看跌": 0.40, "震盪": 0.40, "看漲": 0.20}
        result = get_predicted_class_3(prob_3)
        assert result == "看跌"  # 看跌優先

    def test_tie_neutral_bullish(self):
        """測試相同機率: 震盪=看漲 (優先級: 震盪 > 看漲)"""
        prob_3 = {"看跌": 0.20, "震盪": 0.40, "看漲": 0.40}
        result = get_predicted_class_3(prob_3)
        assert result == "震盪"  # 震盪優先

    def test_tie_all_equal(self):
        """測試相同機率: 三者均等 (優先級: 看跌最高)"""
        prob_3 = {"看跌": 0.333, "震盪": 0.333, "看漲": 0.334}
        result = get_predicted_class_3(prob_3)
        # 看跌和震盪都是0.333,但看漲稍高
        assert result == "看漲"

    def test_uniform_distribution(self):
        """測試均勻分佈 (優先級: 看跌)"""
        prob_3 = {"看跌": 1 / 3, "震盪": 1 / 3, "看漲": 1 / 3}
        result = get_predicted_class_3(prob_3)
        assert result == "看跌"  # 看跌優先

    def test_missing_key(self):
        """測試錯誤處理: 缺少必要鍵"""
        prob_3 = {"看跌": 0.50, "震盪": 0.50}  # 缺少"看漲"
        with pytest.raises(ValueError, match="必須包含"):
            get_predicted_class_3(prob_3)

    def test_all_zero(self):
        """測試錯誤處理: 所有機率為0"""
        prob_3 = {"看跌": 0.0, "震盪": 0.0, "看漲": 0.0}
        with pytest.raises(ValueError, match="所有3分類機率不可同時為0"):
            get_predicted_class_3(prob_3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
