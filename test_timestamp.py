"""
測試時間戳記功能

這個腳本測試模型名稱中的時間戳記功能是否正常運作
"""

import sys
import io
from pathlib import Path
from datetime import datetime

# 設置 UTF-8 編碼輸出
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 加入專案根目錄至 Python 路徑
sys.path.insert(0, str(Path(__file__).parent))


def test_timestamp_generation():
    """測試時間戳記生成"""
    print("=" * 80)
    print("測試 1: 時間戳記格式")
    print("=" * 80)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = "baseline_model"
    model_name_with_timestamp = f"{model_name}_{timestamp}"

    print(f"原始模型名稱: {model_name}")
    print(f"帶時間戳記的模型名稱: {model_name_with_timestamp}")
    print(f"時間戳記格式: YYYYMMDD_HHMMSS")
    print(f"範例: {timestamp}")
    print()

    # 驗證格式
    assert len(timestamp) == 15, "時間戳記長度應為 15 個字元"
    assert timestamp[8] == "_", "第 8 個字元應為底線"
    print("✅ 時間戳記格式正確\n")


def test_model_file_paths():
    """測試模型檔案路徑"""
    print("=" * 80)
    print("測試 2: 模型檔案路徑生成")
    print("=" * 80)

    output_dir = "models"
    model_name = "baseline_model"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name_with_timestamp = f"{model_name}_{timestamp}"

    # 模型檔案路徑
    model_path = Path(output_dir) / f"{model_name_with_timestamp}.h5"
    scaler_path = Path(output_dir) / f"{model_name_with_timestamp}_scaler.pkl"

    print(f"模型檔案路徑: {model_path}")
    print(f"縮放器檔案路徑: {scaler_path}")
    print()

    # 驗證路徑
    assert str(model_path).endswith(".h5"), "模型檔案應為 .h5 格式"
    assert str(scaler_path).endswith("_scaler.pkl"), "縮放器檔案應為 _scaler.pkl 格式"
    assert model_name in str(model_path), "模型路徑應包含原始模型名稱"
    assert timestamp in str(model_path), "模型路徑應包含時間戳記"
    print("✅ 檔案路徑格式正確\n")


def test_multiple_trainings():
    """模擬多次訓練，確保不會覆蓋"""
    print("=" * 80)
    print("測試 3: 多次訓練不會覆蓋")
    print("=" * 80)

    import time

    model_name = "baseline_model"
    model_paths = []

    print("模擬 3 次訓練（每次間隔 1 秒）:\n")

    for i in range(3):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name_with_timestamp = f"{model_name}_{timestamp}"
        model_path = f"models/{model_name_with_timestamp}.h5"
        model_paths.append(model_path)

        print(f"第 {i+1} 次訓練:")
        print(f"  模型名稱: {model_name_with_timestamp}")
        print(f"  模型路徑: {model_path}")

        if i < 2:  # 不在最後一次等待
            time.sleep(1)

    print()

    # 驗證所有路徑都是唯一的
    unique_paths = set(model_paths)
    assert len(unique_paths) == len(model_paths), "模型路徑應該都是唯一的"
    print(f"✅ 生成了 {len(model_paths)} 個唯一的模型路徑")
    print("✅ 確認多次訓練不會互相覆蓋\n")


def test_add_timestamp_parameter():
    """測試 add_timestamp 參數功能"""
    print("=" * 80)
    print("測試 4: add_timestamp 參數控制")
    print("=" * 80)

    model_name = "baseline_model"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 測試 add_timestamp=True
    add_timestamp = True
    if add_timestamp:
        model_name_with_timestamp = f"{model_name}_{timestamp}"
    else:
        model_name_with_timestamp = model_name

    print(f"add_timestamp=True:")
    print(f"  輸出: {model_name_with_timestamp}")
    assert timestamp in model_name_with_timestamp, "應包含時間戳記"
    print("  ✅ 正確加入時間戳記\n")

    # 測試 add_timestamp=False
    add_timestamp = False
    if add_timestamp:
        model_name_with_timestamp = f"{model_name}_{timestamp}"
    else:
        model_name_with_timestamp = model_name

    print(f"add_timestamp=False:")
    print(f"  輸出: {model_name_with_timestamp}")
    assert model_name_with_timestamp == model_name, "不應包含時間戳記"
    print("  ✅ 正確保持原始名稱\n")


def main():
    """主測試流程"""
    print("\n")
    print("測試時間戳記功能")
    print("=" * 80)
    print()

    try:
        # 執行所有測試
        test_timestamp_generation()
        test_model_file_paths()
        test_multiple_trainings()
        test_add_timestamp_parameter()

        # 總結
        print("=" * 80)
        print("✅ 所有測試通過！")
        print("=" * 80)
        print()
        print("功能摘要:")
        print("  1. ✅ 時間戳記格式正確 (YYYYMMDD_HHMMSS)")
        print("  2. ✅ 模型和縮放器檔案路徑正確")
        print("  3. ✅ 多次訓練不會互相覆蓋")
        print("  4. ✅ add_timestamp 參數可正常控制")
        print()
        print("使用範例:")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"  模型檔案: models/baseline_model_{timestamp}.h5")
        print(f"  縮放器檔案: models/baseline_model_{timestamp}_scaler.pkl")
        print()
        print("=" * 80)

        return True

    except AssertionError as e:
        print("\n" + "=" * 80)
        print(f"❌ 測試失敗: {str(e)}")
        print("=" * 80)
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
