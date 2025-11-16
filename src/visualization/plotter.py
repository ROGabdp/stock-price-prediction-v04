"""
視覺化繪圖模組

此模組提供matplotlib橫條圖繪製功能,生成3分類和5分類的雙視圖視覺化
"""

from typing import Dict, Any, Optional, List
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import logging

# 從預測模組導入分類映射常數
from src.prediction.predictor import CLASS_MAPPING


def get_default_config() -> Dict[str, Any]:
    """
    取得預設視覺化配置

    Returns:
        Dict: VisualizationConfigDict (data-model.md定義)
    """
    return {
        "figure_size": (12, 10),
        "dpi": 150,
        "font_family": "Microsoft JhengHei",
        "font_size_title": 14,
        "font_size_label": 11,
        "colors_3class": {
            "看跌": "#DC143C",  # 深紅色 (Crimson)
            "震盪": "#808080",  # 灰色
            "看漲": "#228B22",  # 深綠色 (Forest Green)
        },
        "colors_5class": [
            "#B22222",  # 極度下跌: 暗紅色 (Fire Brick)
            "#FF6347",  # 溫和下跌: 番茄紅 (Tomato)
            "#A9A9A9",  # 區間震盪: 深灰色 (Dark Gray)
            "#32CD32",  # 溫和上漲: 淺綠色 (Lime Green)
            "#006400",  # 極度上漲: 深綠色 (Dark Green)
        ],
        "hatches_3class": {
            "看跌": "///",  # 斜線紋理
            "震盪": "",  # 無紋理
            "看漲": "\\\\\\",  # 反斜線紋理
        },
        "bar_height": 0.6,
        "show_percentage": True,
    }


def configure_chinese_font(font_family: str = "Microsoft JhengHei") -> None:
    """
    配置matplotlib中文字型

    Args:
        font_family: 字型家族名稱
    """
    try:
        # 設定中文字型
        plt.rcParams["font.sans-serif"] = [
            font_family,
            "SimHei",
            "Arial Unicode MS",
            "DejaVu Sans",
        ]
        plt.rcParams["axes.unicode_minus"] = False  # 解決負號顯示問題
        logging.debug(f"matplotlib 中文字型設定成功: {font_family}")
    except Exception as e:
        logging.warning(f"設定中文字型失敗,將使用預設字型: {e}")


def plot_3class_bar(
    ax: matplotlib.axes.Axes,
    prob_3class: Dict[str, float],
    predicted_class_3: str,
    config: Dict[str, Any],
) -> None:
    """
    繪製3分類橫條圖 (輔助函式)

    Args:
        ax: matplotlib axes 物件
        prob_3class: 3分類機率字典
        predicted_class_3: 預測類別
        config: 視覺化配置
    """
    categories = ["看跌", "震盪", "看漲"]
    probabilities = [prob_3class[cat] for cat in categories]
    y_positions = range(len(categories))

    # 繪製橫條圖 (所有橫條從 x=0 開始,左側對齊)
    bars = ax.barh(
        y_positions,
        probabilities,
        height=config["bar_height"],
        left=0,  # 關鍵: 確保所有橫條從0開始
        color=[config["colors_3class"][cat] for cat in categories],
        edgecolor="black",
        linewidth=1.5,
    )

    # 加入紋理 (色盲輔助)
    for bar, cat in zip(bars, categories):
        bar.set_hatch(config["hatches_3class"][cat])

    # 設定軸標籤
    ax.set_yticks(y_positions)
    ax.set_yticklabels(categories, fontsize=config["font_size_label"])
    ax.set_xlabel("機率", fontsize=config["font_size_label"])
    ax.set_xlim(0, 1.0)
    ax.set_title("3分類聚合預測", fontsize=config["font_size_title"], fontweight="bold")

    # 標註百分比於橫條右側
    if config["show_percentage"]:
        for i, prob in enumerate(probabilities):
            ax.text(
                prob + 0.02,  # 橫條右側
                i,
                f"{prob*100:.1f}%",
                va="center",
                fontsize=config["font_size_label"],
            )

    # 標記預測類別 (箭頭)
    for i, cat in enumerate(categories):
        if cat == predicted_class_3:
            ax.text(
                -0.05,  # 橫條左側
                i,
                "←",
                va="center",
                ha="right",
                fontsize=14,
                fontweight="bold",
                color="red",
            )

    # 加入網格線
    ax.grid(axis="x", alpha=0.3, linestyle="--")


def plot_5class_bar(
    ax: matplotlib.axes.Axes,
    prob_5class: np.ndarray,
    predicted_class_5: int,
    config: Dict[str, Any],
) -> None:
    """
    繪製5分類橫條圖 (輔助函式)

    Args:
        ax: matplotlib axes 物件
        prob_5class: 5分類機率向量 (numpy array)
        predicted_class_5: 預測類別索引 (0-4)
        config: 視覺化配置
    """
    # 取得類別標籤
    categories = [CLASS_MAPPING[i]["label"] for i in range(5)]
    probabilities = prob_5class.tolist()
    y_positions = range(len(categories))

    # 繪製橫條圖 (所有橫條從 x=0 開始,左側對齊)
    bars = ax.barh(
        y_positions,
        probabilities,
        height=config["bar_height"],
        left=0,  # 關鍵: 確保所有橫條從0開始
        color=config["colors_5class"],
        edgecolor="black",
        linewidth=1.5,
    )

    # 設定軸標籤
    ax.set_yticks(y_positions)
    ax.set_yticklabels(categories, fontsize=config["font_size_label"])
    ax.set_xlabel("機率", fontsize=config["font_size_label"])
    ax.set_xlim(0, 1.0)
    ax.set_title("5分類詳細預測", fontsize=config["font_size_title"], fontweight="bold")

    # 標註百分比於橫條右側
    if config["show_percentage"]:
        for i, prob in enumerate(probabilities):
            ax.text(
                prob + 0.02,  # 橫條右側
                i,
                f"{prob*100:.1f}%",
                va="center",
                fontsize=config["font_size_label"],
            )

    # 標記預測類別 (箭頭)
    ax.text(
        -0.05,  # 橫條左側
        predicted_class_5,
        "←",
        va="center",
        ha="right",
        fontsize=14,
        fontweight="bold",
        color="red",
    )

    # 加入網格線
    ax.grid(axis="x", alpha=0.3, linestyle="--")


def plot_dual_view(
    prediction_result: Dict[str, Any],
    output_path: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
) -> str:
    """
    繪製雙視圖 (3分類 + 5分類) 橫條圖並儲存為PNG檔案

    Args:
        prediction_result: PredictionResultDict (data-model.md定義)
        output_path: 輸出檔案路徑,若為 None 則自動生成
        config: VisualizationConfigDict,若為 None 使用預設配置

    Returns:
        str: 實際儲存的檔案路徑

    Raises:
        ValueError: 若 prediction_result 格式不正確
        IOError: 若檔案寫入失敗

    Example:
        >>> result = {...}  # PredictionResultDict
        >>> path = plot_dual_view(result, "output/prediction.png")
        >>> print(f"圖表已儲存至: {path}")
    """
    # 使用預設配置 (若未提供)
    if config is None:
        config = get_default_config()

    # 驗證必要欄位
    required_fields = [
        "input_date",
        "prediction_date",
        "prob_5class",
        "prob_3class",
        "predicted_class_5",
        "predicted_class_3",
        "confidence",
    ]
    for field in required_fields:
        if field not in prediction_result:
            raise ValueError(f"prediction_result 缺少必要欄位: {field}")

    # 配置中文字型
    configure_chinese_font(config["font_family"])

    # 生成輸出路徑 (若未提供)
    if output_path is None:
        input_date = prediction_result["input_date"]
        prediction_date = prediction_result["prediction_date"]
        filename = f"prediction_{input_date}_{prediction_date}.png"
        output_path = os.path.join("outputs", filename)

    # 確保輸出目錄存在
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir, exist_ok=True)
            logging.debug(f"建立輸出目錄: {output_dir}")
        except Exception as e:
            raise IOError(f"無法建立輸出目錄 {output_dir}: {e}")

    # 建立2x1子圖 (上方3分類,下方5分類)
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=config["figure_size"], dpi=config["dpi"]
    )

    # 繪製3分類子圖 (上方)
    plot_3class_bar(
        ax1,
        prediction_result["prob_3class"],
        prediction_result["predicted_class_3"],
        config,
    )

    # 繪製5分類子圖 (下方)
    plot_5class_bar(
        ax2,
        prediction_result["prob_5class"],
        prediction_result["predicted_class_5"],
        config,
    )

    # 若實際資料存在,加入驗證摘要
    actual_data = prediction_result.get("actual_data")
    if actual_data is not None:
        # 組裝驗證摘要文字
        actual_close = actual_data["actual_close"]
        actual_change_pct = actual_data["actual_change_pct"]
        is_correct_5 = actual_data["is_correct_5class"]
        is_correct_3 = actual_data["is_correct_3class"]

        # 格式化收盤價
        actual_close_str = f"{actual_close:,.0f}"
        change_str = f"{actual_change_pct:+.2%}"

        # 正確性標記
        result_5_str = "正確 ✓" if is_correct_5 else "錯誤 ✗"
        result_3_str = "正確 ✓" if is_correct_3 else "錯誤 ✗"

        # 顏色
        color_5 = "green" if is_correct_5 else "red"
        color_3 = "green" if is_correct_3 else "red"

        # 在標題上方加入驗證摘要 (使用 fig.text)
        summary_text = (
            f"【實際結果】\n"
            f"實際收盤價: {actual_close_str} ({change_str})\n"
            f"5分類預測: {result_5_str}  |  3分類預測: {result_3_str}"
        )

        fig.text(
            0.5,
            0.97,
            summary_text,
            ha="center",
            va="top",
            fontsize=config["font_size_label"],
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    # 調整子圖間距
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # 為驗證摘要預留空間

    # 儲存圖片
    try:
        plt.savefig(output_path, dpi=config["dpi"], bbox_inches="tight")
        logging.info(f"視覺化圖表已儲存至: {output_path}")
    except Exception as e:
        plt.close(fig)  # 確保釋放記憶體
        raise IOError(f"儲存圖片失敗: {e}")
    finally:
        # 關閉 figure 釋放記憶體
        plt.close(fig)

    return output_path
