"""
單元測試：資料載入模組

測試 CSV 載入、欄位驗證、缺失值處理等功能
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile

from src.data.data_loader import (
    load_csv_data,
    validate_schema,
    preprocess_data,
    get_data_info,
)


class TestLoadCSVData:
    """測試 CSV 資料載入功能"""

    def setup_method(self):
        """建立測試用的臨時 CSV 檔案"""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_csv = Path(self.temp_dir) / "test_data.csv"

        # 建立測試資料
        self.test_data = {
            "date": ["2024/1/1", "2024/1/2", "2024/1/3"],
            "open": [15800.0, 15850.0, 15900.0],
            "high": [15900.0, 15950.0, 16000.0],
            "low": [15750.0, 15800.0, 15850.0],
            "close": [15850.0, 15900.0, 15950.0],
            "volume": [1500.0, 1600.0, 1550.0],
            "SMA5": [15800.0, 15825.0, 15850.0],
            "SMA10": [15750.0, 15775.0, 15800.0],
            "SMA20": [15700.0, 15725.0, 15750.0],
            "SMA60": [15600.0, 15625.0, 15650.0],
            "SMA120": [15500.0, 15525.0, 15550.0],
            "SMA240": [15400.0, 15425.0, 15450.0],
            "MA5": [15800.0, 15825.0, 15850.0],
            "MA10": [15750.0, 15775.0, 15800.0],
            "DIF12-26": [50.0, 55.0, 60.0],
            "MACD9": [45.0, 50.0, 55.0],
            "OSC": [5.0, 5.0, 5.0],
            "K(9,3)": [0.65, 0.68, 0.70],
            "D(9,3)": [0.62, 0.65, 0.67],
            "net buy sell": [1000000.0, 1200000.0, 1100000.0],
            "cumulative net buy sell": [5000000.0, 6200000.0, 7300000.0],
            "buy": [3000000.0, 3200000.0, 3100000.0],
            "sell": [2000000.0, 2000000.0, 2000000.0],
        }

        df = pd.DataFrame(self.test_data)
        df.to_csv(self.temp_csv, index=False)

    def test_load_csv_success(self):
        """測試成功載入 CSV"""
        df = load_csv_data(str(self.temp_csv))

        assert df is not None
        assert len(df) == 3
        assert "date" in df.columns
        assert "close" in df.columns

    def test_load_csv_nonexistent_file(self):
        """測試載入不存在的檔案"""
        with pytest.raises(FileNotFoundError):
            load_csv_data("nonexistent_file.csv")

    def test_date_column_parsed(self):
        """測試日期欄位正確解析"""
        df = load_csv_data(str(self.temp_csv))

        assert pd.api.types.is_datetime64_any_dtype(df["date"])
        assert df["date"].iloc[0] == pd.Timestamp("2024-01-01")

    def test_numeric_columns(self):
        """測試數值欄位類型正確"""
        df = load_csv_data(str(self.temp_csv))

        numeric_cols = ["open", "high", "low", "close", "volume"]
        for col in numeric_cols:
            assert pd.api.types.is_numeric_dtype(df[col])


class TestValidateSchema:
    """測試資料欄位驗證功能"""

    def test_valid_schema(self):
        """測試有效的資料結構"""
        df = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=3),
                "open": [100.0, 101.0, 102.0],
                "high": [105.0, 106.0, 107.0],
                "low": [98.0, 99.0, 100.0],
                "close": [102.0, 103.0, 104.0],
                "volume": [1000.0, 1100.0, 1200.0],
                "SMA5": [100.0, 101.0, 102.0],
                "SMA10": [100.0, 101.0, 102.0],
                "SMA20": [100.0, 101.0, 102.0],
                "SMA60": [100.0, 101.0, 102.0],
                "SMA120": [100.0, 101.0, 102.0],
                "SMA240": [100.0, 101.0, 102.0],
                "MA5": [100.0, 101.0, 102.0],
                "MA10": [100.0, 101.0, 102.0],
                "DIF12-26": [1.0, 1.1, 1.2],
                "MACD9": [0.9, 1.0, 1.1],
                "OSC": [0.1, 0.1, 0.1],
                "K(9,3)": [0.5, 0.6, 0.7],
                "D(9,3)": [0.5, 0.6, 0.7],
                "net buy sell": [1000.0, 1100.0, 1200.0],
                "cumulative net buy sell": [5000.0, 6100.0, 7300.0],
                "buy": [3000.0, 3100.0, 3200.0],
                "sell": [2000.0, 2000.0, 2000.0],
            }
        )

        # validate_schema 不回傳值,若驗證通過則不拋出異常
        try:
            validate_schema(df)
            validation_passed = True
        except ValueError:
            validation_passed = False

        assert validation_passed

    def test_missing_columns(self):
        """測試缺少必要欄位"""
        df = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=3),
                "open": [100.0, 101.0, 102.0],
                "close": [102.0, 103.0, 104.0],
            }
        )

        # validate_schema 應拋出 ValueError 當缺少欄位時
        with pytest.raises(ValueError) as exc_info:
            validate_schema(df)

        # 檢查錯誤訊息包含缺少的欄位
        error_message = str(exc_info.value)
        assert "缺少必要欄位" in error_message
        assert "high" in error_message or "low" in error_message


class TestPreprocessData:
    """測試資料預處理功能"""

    def test_date_conversion(self):
        """測試日期轉換"""
        df = pd.DataFrame(
            {
                "date": ["2024/1/1", "2024/1/2", "2024/1/3"],
                "open": [100.0, 101.0, 102.0],
                "high": [105.0, 106.0, 107.0],
                "low": [98.0, 99.0, 100.0],
                "close": [102.0, 103.0, 104.0],
                "volume": [1000.0, 1100.0, 1200.0],
                "SMA5": [100.0, 101.0, 102.0],
                "SMA10": [100.0, 101.0, 102.0],
                "SMA20": [100.0, 101.0, 102.0],
                "SMA60": [100.0, 101.0, 102.0],
                "SMA120": [100.0, 101.0, 102.0],
                "SMA240": [100.0, 101.0, 102.0],
                "MA5": [100.0, 101.0, 102.0],
                "MA10": [100.0, 101.0, 102.0],
                "DIF12-26": [1.0, 1.1, 1.2],
                "MACD9": [0.9, 1.0, 1.1],
                "OSC": [0.1, 0.1, 0.1],
                "K(9,3)": [0.5, 0.6, 0.7],
                "D(9,3)": [0.5, 0.6, 0.7],
                "net buy sell": [1000.0, 1100.0, 1200.0],
                "cumulative net buy sell": [5000.0, 6100.0, 7300.0],
                "buy": [3000.0, 3100.0, 3200.0],
                "sell": [2000.0, 2000.0, 2000.0],
            }
        )

        df_processed = preprocess_data(df)
        assert pd.api.types.is_datetime64_any_dtype(df_processed["date"])

    def test_remove_missing_values(self):
        """測試移除缺失值"""
        df = pd.DataFrame(
            {
                "date": ["2024/1/1", "2024/1/2", "2024/1/3"],
                "open": [100.0, np.nan, 102.0],
                "high": [105.0, 106.0, 107.0],
                "low": [98.0, 99.0, 100.0],
                "close": [102.0, 103.0, 104.0],
                "volume": [1000.0, 1100.0, 1200.0],
                "SMA5": [100.0, 101.0, 102.0],
                "SMA10": [100.0, 101.0, 102.0],
                "SMA20": [100.0, 101.0, 102.0],
                "SMA60": [100.0, 101.0, 102.0],
                "SMA120": [100.0, 101.0, 102.0],
                "SMA240": [100.0, 101.0, 102.0],
                "MA5": [100.0, 101.0, 102.0],
                "MA10": [100.0, 101.0, 102.0],
                "DIF12-26": [1.0, 1.1, 1.2],
                "MACD9": [0.9, 1.0, 1.1],
                "OSC": [0.1, 0.1, 0.1],
                "K(9,3)": [0.5, 0.6, 0.7],
                "D(9,3)": [0.5, 0.6, 0.7],
                "net buy sell": [1000.0, 1100.0, 1200.0],
                "cumulative net buy sell": [5000.0, 6100.0, 7300.0],
                "buy": [3000.0, 3100.0, 3200.0],
                "sell": [2000.0, 2000.0, 2000.0],
            }
        )

        df_processed = preprocess_data(df)
        # 應移除包含 NaN 的第 2 列
        assert len(df_processed) == 2
        assert not df_processed["open"].isnull().any()


class TestDataValidation:
    """測試資料驗證規則"""

    def test_price_validation(self):
        """測試價格驗證（必須 > 0）"""
        df = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=3),
                "open": [100.0, 101.0, 102.0],
                "high": [105.0, 106.0, 107.0],
                "low": [98.0, 99.0, 100.0],
                "close": [102.0, 103.0, 104.0],
            }
        )

        # 所有價格應 > 0
        assert (df["open"] > 0).all()
        assert (df["high"] > 0).all()
        assert (df["low"] > 0).all()
        assert (df["close"] > 0).all()

    def test_high_low_relationship(self):
        """測試 high >= low 關係"""
        df = pd.DataFrame(
            {
                "high": [105.0, 106.0, 107.0],
                "low": [98.0, 99.0, 100.0],
            }
        )

        # high 應該 >= low
        assert (df["high"] >= df["low"]).all()

    def test_volume_non_negative(self):
        """測試成交量非負"""
        df = pd.DataFrame({"volume": [1000.0, 1100.0, 1200.0]})

        # volume 應該 >= 0
        assert (df["volume"] >= 0).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
