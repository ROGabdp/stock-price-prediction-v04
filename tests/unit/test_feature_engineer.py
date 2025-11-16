"""
單元測試：特徵工程模組

測試特徵工程邏輯、Set A/B/C 定義正確性、目標變數計算
"""

import pytest
import pandas as pd
import numpy as np

from src.features.feature_engineer import (
    engineer_features,
    calculate_target_variable,
    calculate_price_returns,
)
from src.features.feature_sets import (
    get_feature_set,
    get_feature_names,
    list_all_feature_sets,
)


class TestFeatureSets:
    """測試特徵集定義"""

    def test_all_feature_sets_exist(self):
        """測試所有特徵集存在"""
        feature_sets = list_all_feature_sets()
        assert "Set A" in feature_sets
        assert "Set B" in feature_sets
        assert "Set C" in feature_sets

    def test_set_a_definition(self):
        """測試 Set A（動能型）定義"""
        config = get_feature_set("Set A")

        assert config["name"] == "動能型"
        assert "DIF12-26" in config["features"]
        assert "MACD9" in config["features"]
        assert "K(9,3)" not in config["features"]  # Set A 不包含 KD

    def test_set_b_definition(self):
        """測試 Set B（震盪型）定義"""
        config = get_feature_set("Set B")

        assert config["name"] == "震盪型"
        assert "K(9,3)" in config["features"]
        assert "D(9,3)" in config["features"]
        assert "DIF12-26" not in config["features"]  # Set B 不包含 MACD

    def test_set_c_definition(self):
        """測試 Set C（全特徵集）定義"""
        config = get_feature_set("Set C")

        assert config["name"] == "全特徵集"
        # Set C 應包含所有特徵
        assert "DIF12-26" in config["features"]
        assert "MACD9" in config["features"]
        assert "OSC" in config["features"]
        assert "K(9,3)" in config["features"]
        assert "D(9,3)" in config["features"]

    def test_feature_count(self):
        """測試各特徵集的特徵數量"""
        set_a_features = get_feature_names("Set A")
        set_b_features = get_feature_names("Set B")
        set_c_features = get_feature_names("Set C")

        # Set A 和 Set B 應有相似數量的特徵（約 12-15 個）
        assert 10 <= len(set_a_features) <= 20
        assert 10 <= len(set_b_features) <= 20

        # Set C 應該是最多的（約 16-22 個）
        assert len(set_c_features) >= len(set_a_features)
        assert len(set_c_features) >= len(set_b_features)


class TestCalculatePriceReturns:
    """測試價格變化率計算"""

    def test_price_returns_calculation(self):
        """測試價格變化率計算正確性"""
        df = pd.DataFrame(
            {
                "open": [100.0, 102.0, 104.0, 103.0],
                "high": [105.0, 107.0, 109.0, 108.0],
                "low": [98.0, 100.0, 102.0, 101.0],
                "close": [102.0, 104.0, 106.0, 105.0],
            }
        )

        df_returns = calculate_price_returns(df)

        # 檢查欄位存在
        assert "open_return" in df_returns.columns
        assert "high_return" in df_returns.columns
        assert "low_return" in df_returns.columns
        assert "close_return" in df_returns.columns

        # 第一行應該是 NaN（沒有前一天資料）
        assert pd.isna(df_returns["close_return"].iloc[0])

        # 檢查計算正確性
        # close_return[1] = (104 - 102) / 102 ≈ 0.0196
        expected_return = (104.0 - 102.0) / 102.0
        assert abs(df_returns["close_return"].iloc[1] - expected_return) < 1e-6

    def test_returns_range(self):
        """測試價格變化率範圍合理"""
        df = pd.DataFrame(
            {
                "open": [100.0, 102.0, 104.0],
                "high": [105.0, 107.0, 109.0],
                "low": [98.0, 100.0, 102.0],
                "close": [102.0, 104.0, 106.0],
            }
        )

        df_returns = calculate_price_returns(df)

        # 正常股價變化應在合理範圍內（例如 -20% 到 +20%）
        for col in ["open_return", "high_return", "low_return", "close_return"]:
            valid_returns = df_returns[col].dropna()
            assert (valid_returns >= -0.2).all()
            assert (valid_returns <= 0.2).all()


class TestCalculateTargetVariable:
    """測試目標變數計算"""

    def test_target_variable_shape(self):
        """測試目標變數形狀"""
        df = pd.DataFrame({"close": [100.0, 102.0, 104.0, 106.0, 105.0]})

        target = calculate_target_variable(df, future_days=20)

        # 前 20 天應該沒有目標（因為沒有未來資料）
        # 所以有效目標應少於原始資料
        assert len(target) <= len(df)

        # 目標應該是 5 類 One-Hot 編碼
        assert target.shape[1] == 5

    def test_target_classes(self):
        """測試目標類別正確性"""
        # 建立測試資料，確保每種漲跌幅都有
        prices = [
            100.0,
            105.0,  # +5% (極度上漲)
            103.0,  # -1.9% (區間震盪)
            97.0,  # -5.8% (極度下跌)
            100.0,  # +3.1% (溫和上漲)
            96.0,  # -4% (溫和下跌)
        ]

        df = pd.DataFrame({"close": prices})
        target = calculate_target_variable(df, future_days=1)

        # One-Hot 編碼每行應該只有一個 1
        row_sums = target.sum(axis=1)
        assert (row_sums == 1).all()

    def test_target_variable_one_hot(self):
        """測試 One-Hot 編碼正確性"""
        df = pd.DataFrame({"close": [100.0, 103.0, 106.0, 104.0]})

        target = calculate_target_variable(df, future_days=1)

        # 每一行應該恰好有一個 1，其餘為 0
        for i in range(len(target)):
            row = target.iloc[i]
            assert row.sum() == 1  # 恰好一個 1
            assert ((row == 0) | (row == 1)).all()  # 只有 0 或 1


class TestEngineerFeatures:
    """測試特徵工程主邏輯"""

    def setup_method(self):
        """建立測試用資料"""
        self.df_raw = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=100),
                "open": np.random.uniform(15000, 16000, 100),
                "high": np.random.uniform(15500, 16500, 100),
                "low": np.random.uniform(14500, 15500, 100),
                "close": np.random.uniform(15000, 16000, 100),
                "volume": np.random.uniform(1000, 2000, 100),
                "SMA5": np.random.uniform(15000, 16000, 100),
                "SMA10": np.random.uniform(15000, 16000, 100),
                "SMA20": np.random.uniform(15000, 16000, 100),
                "SMA60": np.random.uniform(15000, 16000, 100),
                "SMA120": np.random.uniform(15000, 16000, 100),
                "SMA240": np.random.uniform(15000, 16000, 100),
                "MA5": np.random.uniform(15000, 16000, 100),
                "MA10": np.random.uniform(15000, 16000, 100),
                "DIF12-26": np.random.uniform(-100, 100, 100),
                "MACD9": np.random.uniform(-100, 100, 100),
                "OSC": np.random.uniform(-50, 50, 100),
                "K(9,3)": np.random.uniform(0, 1, 100),
                "D(9,3)": np.random.uniform(0, 1, 100),
                "net buy sell": np.random.uniform(-1e6, 1e6, 100),
                "cumulative net buy sell": np.random.uniform(-1e7, 1e7, 100),
                "buy": np.random.uniform(0, 5e6, 100),
                "sell": np.random.uniform(0, 5e6, 100),
            }
        )

    def test_engineer_features_set_a(self):
        """測試 Set A 特徵工程"""
        df_features, target = engineer_features(self.df_raw, feature_set_id="Set A")

        # 檢查輸出形狀
        assert df_features is not None
        assert target is not None

        # Set A 應包含特定特徵
        assert "open_return" in df_features.columns
        assert "DIF12-26" in df_features.columns
        assert "MACD9" in df_features.columns

        # Set A 不應包含 KD
        assert "K(9,3)" not in df_features.columns
        assert "D(9,3)" not in df_features.columns

    def test_engineer_features_set_b(self):
        """測試 Set B 特徵工程"""
        df_features, target = engineer_features(self.df_raw, feature_set_id="Set B")

        # Set B 應包含 KD
        assert "K(9,3)" in df_features.columns
        assert "D(9,3)" in df_features.columns

        # Set B 不應包含 MACD
        assert "DIF12-26" not in df_features.columns
        assert "MACD9" not in df_features.columns

    def test_engineer_features_set_c(self):
        """測試 Set C 特徵工程"""
        df_features, target = engineer_features(self.df_raw, feature_set_id="Set C")

        # Set C 應包含所有特徵
        assert "DIF12-26" in df_features.columns
        assert "K(9,3)" in df_features.columns
        assert "OSC" in df_features.columns

    def test_no_missing_values_in_features(self):
        """測試特徵工程後無缺失值"""
        df_features, target = engineer_features(self.df_raw, feature_set_id="Set C")

        # 特徵矩陣不應有 NaN
        assert not df_features.isnull().any().any()

        # 目標變數不應有 NaN
        assert not target.isnull().any().any()

    def test_feature_scaling_ready(self):
        """測試特徵準備好進行縮放"""
        df_features, target = engineer_features(self.df_raw, feature_set_id="Set A")

        # 所有欄位應該是數值型
        assert df_features.select_dtypes(include=[np.number]).shape[1] == df_features.shape[1]

        # 不應有無限值
        assert not np.isinf(df_features.values).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
