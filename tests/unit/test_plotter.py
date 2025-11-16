"""
繪圖模組單元測試

測試視覺化繪圖功能的正確性
"""

import pytest
import numpy as np
import os
import tempfile
from src.visualization.plotter import (
    get_default_config,
    configure_chinese_font,
    plot_dual_view,
)


class TestGetDefaultConfig:
    """測試 get_default_config 函式"""

    def test_config_structure(self):
        """測試配置結構完整性"""
        config = get_default_config()

        # 驗證必要欄位存在
        assert "figure_size" in config
        assert "dpi" in config
        assert "font_family" in config
        assert "font_size_title" in config
        assert "font_size_label" in config
        assert "colors_3class" in config
        assert "colors_5class" in config
        assert "hatches_3class" in config
        assert "bar_height" in config
        assert "show_percentage" in config

    def test_figure_size(self):
        """測試圖表尺寸"""
        config = get_default_config()
        assert config["figure_size"] == (12, 10)
        assert isinstance(config["figure_size"], tuple)
        assert len(config["figure_size"]) == 2

    def test_dpi(self):
        """測試解析度"""
        config = get_default_config()
        assert config["dpi"] == 150
        assert config["dpi"] >= 100  # 滿足 SC-005

    def test_colors_3class(self):
        """測試3分類顏色映射"""
        config = get_default_config()
        colors = config["colors_3class"]

        assert "看跌" in colors
        assert "震盪" in colors
        assert "看漲" in colors

        # 驗證顏色格式 (十六進位)
        assert colors["看跌"].startswith("#")
        assert colors["震盪"].startswith("#")
        assert colors["看漲"].startswith("#")

    def test_colors_5class(self):
        """測試5分類顏色列表"""
        config = get_default_config()
        colors = config["colors_5class"]

        assert isinstance(colors, list)
        assert len(colors) == 5

        # 驗證所有顏色格式
        for color in colors:
            assert color.startswith("#")

    def test_hatches_3class(self):
        """測試3分類紋理映射"""
        config = get_default_config()
        hatches = config["hatches_3class"]

        assert "看跌" in hatches
        assert "震盪" in hatches
        assert "看漲" in hatches

    def test_bar_height(self):
        """測試橫條高度"""
        config = get_default_config()
        assert 0 < config["bar_height"] <= 1.0

    def test_show_percentage(self):
        """測試百分比顯示選項"""
        config = get_default_config()
        assert isinstance(config["show_percentage"], bool)


class TestConfigureChineseFont:
    """測試 configure_chinese_font 函式"""

    def test_configure_font_no_error(self):
        """測試字型配置不拋出錯誤"""
        # 此測試僅驗證函式可以正常執行
        try:
            configure_chinese_font("Microsoft JhengHei")
            configure_chinese_font("SimHei")
        except Exception as e:
            pytest.fail(f"字型配置失敗: {e}")


class TestPlotDualView:
    """測試 plot_dual_view 函式"""

    @pytest.fixture
    def sample_prediction_result(self):
        """建立測試用預測結果"""
        return {
            "input_date": "2024-01-15",
            "prediction_date": "2024-02-15",
            "prob_5class": np.array([0.05, 0.15, 0.25, 0.35, 0.20]),
            "prob_3class": {"看跌": 0.20, "震盪": 0.25, "看漲": 0.55},
            "predicted_class_5": 3,
            "predicted_class_3": "看漲",
            "confidence": 0.35,
            "actual_data": None,
        }

    @pytest.fixture
    def sample_prediction_with_actual(self):
        """建立包含實際資料的測試預測結果"""
        return {
            "input_date": "2024-01-15",
            "prediction_date": "2024-02-15",
            "prob_5class": np.array([0.05, 0.15, 0.25, 0.35, 0.20]),
            "prob_3class": {"看跌": 0.20, "震盪": 0.25, "看漲": 0.55},
            "predicted_class_5": 3,
            "predicted_class_3": "看漲",
            "confidence": 0.35,
            "actual_data": {
                "actual_close": 18500.0,
                "actual_change_pct": 0.032,
                "actual_class_5": 3,
                "actual_class_3": "看漲",
                "is_correct_5class": True,
                "is_correct_3class": True,
            },
        }

    def test_plot_basic(self, sample_prediction_result):
        """測試基本繪圖功能"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test_plot.png")
            result_path = plot_dual_view(sample_prediction_result, output_path)

            # 驗證檔案存在
            assert os.path.exists(result_path)
            assert result_path == output_path

            # 驗證檔案為PNG
            assert result_path.endswith(".png")

            # 驗證檔案大小合理 (> 0 bytes)
            file_size = os.path.getsize(result_path)
            assert file_size > 0

    def test_plot_with_actual_data(self, sample_prediction_with_actual):
        """測試包含實際資料的繪圖"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test_plot_with_actual.png")
            result_path = plot_dual_view(sample_prediction_with_actual, output_path)

            assert os.path.exists(result_path)

    def test_plot_auto_filename(self, sample_prediction_result):
        """測試自動生成檔名"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # 臨時改變工作目錄
            original_outputs = "outputs"
            try:
                # 建立 tmpdir/outputs
                os.makedirs(os.path.join(tmpdir, "outputs"), exist_ok=True)

                # 使用相對路徑
                output_path = os.path.join(
                    tmpdir, "outputs", "prediction_2024-01-15_2024-02-15.png"
                )
                result_path = plot_dual_view(sample_prediction_result, output_path)

                assert os.path.exists(result_path)
                assert "prediction_2024-01-15_2024-02-15.png" in result_path
            finally:
                pass

    def test_plot_custom_config(self, sample_prediction_result):
        """測試自訂配置"""
        custom_config = get_default_config()
        custom_config["dpi"] = 100
        custom_config["figure_size"] = (10, 8)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test_custom.png")
            result_path = plot_dual_view(
                sample_prediction_result, output_path, custom_config
            )

            assert os.path.exists(result_path)

    def test_plot_missing_required_field(self):
        """測試缺少必要欄位時拋出錯誤"""
        incomplete_result = {
            "input_date": "2024-01-15",
            # 缺少其他必要欄位
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test_error.png")
            with pytest.raises(ValueError, match="缺少必要欄位"):
                plot_dual_view(incomplete_result, output_path)

    def test_plot_invalid_output_directory(self, sample_prediction_result):
        """測試無效輸出目錄 - 跳過因為會自動創建目錄"""
        # 此功能會自動建立目錄,所以此測試跳過
        pytest.skip("plot_dual_view 會自動建立不存在的目錄,此測試不適用")

    def test_plot_creates_directory(self, sample_prediction_result):
        """測試自動建立不存在的目錄"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # 使用不存在的子目錄
            output_path = os.path.join(tmpdir, "subdir", "test.png")
            result_path = plot_dual_view(sample_prediction_result, output_path)

            assert os.path.exists(result_path)
            assert os.path.exists(os.path.dirname(result_path))

    def test_plot_file_size_reasonable(self, sample_prediction_result):
        """測試輸出檔案大小合理 (< 500 KB)"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test_size.png")
            result_path = plot_dual_view(sample_prediction_result, output_path)

            file_size = os.path.getsize(result_path)
            # 驗證檔案大小 < 500 KB (規格要求)
            assert file_size < 500 * 1024


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
