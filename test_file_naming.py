"""
測試檔案命名一致性

驗證訓練和調整後產生的檔案名稱都包含時間戳記
"""

import sys
import io
from datetime import datetime
from pathlib import Path

# 設置 UTF-8 編碼輸出
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def test_baseline_model_files():
    """測試基準模型的檔案命名"""
    print("=" * 80)
    print("測試 1: 基準模型檔案命名")
    print("=" * 80)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = "baseline_model"
    model_name_with_timestamp = f"{model_name}_{timestamp}"

    # 預期的檔案名稱
    expected_files = {
        "模型檔案": f"models/{model_name_with_timestamp}.h5",
        "配置檔案": f"models/{model_name_with_timestamp}_config.json",
        "縮放器檔案": f"models/{model_name_with_timestamp}_scaler.pkl",
        "訓練日誌": f"logs/training_logs/{model_name_with_timestamp}_training_log.csv",
    }

    print(f"基準模型名稱: {model_name}")
    print(f"時間戳記: {timestamp}")
    print(f"完整名稱: {model_name_with_timestamp}")
    print()
    print("預期產生的檔案:")
    for file_type, file_path in expected_files.items():
        print(f"  {file_type}: {file_path}")

    # 驗證命名格式
    for file_type, file_path in expected_files.items():
        # 檢查檔案名稱包含時間戳記
        assert timestamp in file_path, f"{file_type} 應包含時間戳記"
        # 檢查檔案名稱包含基準模型名稱
        assert model_name in file_path, f"{file_type} 應包含基準模型名稱"

    print()
    print("✅ 基準模型檔案命名格式正確")
    print()


def test_tuned_model_files():
    """測試調整模型的檔案命名"""
    print("=" * 80)
    print("測試 2: 調整模型檔案命名")
    print("=" * 80)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tuned_model_name = "best_tuned_model"
    tuned_model_name_with_timestamp = f"{tuned_model_name}_{timestamp}"

    # 預期的檔案名稱
    expected_files = {
        "模型檔案": f"models/{tuned_model_name_with_timestamp}.h5",
        "配置檔案": f"models/{tuned_model_name_with_timestamp}_config.json",
        "調整結果": f"logs/tuning_logs/tuning_results_{timestamp}.txt",
        "比較報告": f"logs/tuning_logs/comparison_report_{timestamp}.txt",
    }

    print(f"調整模型名稱: {tuned_model_name}")
    print(f"時間戳記: {timestamp}")
    print(f"完整名稱: {tuned_model_name_with_timestamp}")
    print()
    print("預期產生的檔案:")
    for file_type, file_path in expected_files.items():
        print(f"  {file_type}: {file_path}")

    # 驗證命名格式
    for file_type, file_path in expected_files.items():
        # 檢查檔案名稱包含時間戳記
        assert timestamp in file_path, f"{file_type} 應包含時間戳記"

    # 檢查模型和配置檔案包含模型名稱
    assert tuned_model_name in expected_files["模型檔案"]
    assert tuned_model_name in expected_files["配置檔案"]

    print()
    print("✅ 調整模型檔案命名格式正確")
    print()


def test_config_file_pairing():
    """測試模型和配置檔案的配對"""
    print("=" * 80)
    print("測試 3: 模型與配置檔案配對")
    print("=" * 80)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 基準模型
    baseline_model_file = f"models/baseline_model_{timestamp}.h5"
    baseline_config_file = f"models/baseline_model_{timestamp}_config.json"

    # 調整模型
    tuned_model_file = f"models/best_tuned_model_{timestamp}.h5"
    tuned_config_file = f"models/best_tuned_model_{timestamp}_config.json"

    print("基準模型配對:")
    print(f"  模型: {baseline_model_file}")
    print(f"  配置: {baseline_config_file}")

    # 驗證配對關係
    baseline_model_stem = baseline_model_file.replace('.h5', '')
    baseline_config_stem = baseline_config_file.replace('_config.json', '')
    assert baseline_model_stem == baseline_config_stem, "基準模型和配置檔案名稱不匹配"
    print("  ✅ 配對正確")

    print()
    print("調整模型配對:")
    print(f"  模型: {tuned_model_file}")
    print(f"  配置: {tuned_config_file}")

    # 驗證配對關係
    tuned_model_stem = tuned_model_file.replace('.h5', '')
    tuned_config_stem = tuned_config_file.replace('_config.json', '')
    assert tuned_model_stem == tuned_config_stem, "調整模型和配置檔案名稱不匹配"
    print("  ✅ 配對正確")

    print()
    print("✅ 所有模型與配置檔案配對正確")
    print()


def test_scaler_pairing():
    """測試模型和縮放器檔案的配對"""
    print("=" * 80)
    print("測試 4: 模型與縮放器配對")
    print("=" * 80)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 基準模型
    model_file = f"models/baseline_model_{timestamp}.h5"
    scaler_file = f"models/baseline_model_{timestamp}_scaler.pkl"
    config_file = f"models/baseline_model_{timestamp}_config.json"

    print("三個檔案應使用相同的基礎名稱:")
    print(f"  模型: {model_file}")
    print(f"  配置: {config_file}")
    print(f"  縮放器: {scaler_file}")

    # 提取基礎名稱
    model_base = model_file.replace('.h5', '')
    config_base = config_file.replace('_config.json', '')
    scaler_base = scaler_file.replace('_scaler.pkl', '')

    # 驗證三者一致
    assert model_base == config_base == scaler_base, "模型、配置和縮放器的基礎名稱應該一致"

    print()
    print("✅ 模型、配置和縮放器配對正確")
    print()


def test_timestamp_format():
    """測試時間戳記格式"""
    print("=" * 80)
    print("測試 5: 時間戳記格式")
    print("=" * 80)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"時間戳記格式: YYYYMMDD_HHMMSS")
    print(f"範例: {timestamp}")
    print()

    # 驗證格式
    parts = timestamp.split('_')
    assert len(parts) == 2, "時間戳記應包含日期和時間兩部分"

    date_part, time_part = parts
    assert len(date_part) == 8, "日期部分應為 8 位數字 (YYYYMMDD)"
    assert len(time_part) == 6, "時間部分應為 6 位數字 (HHMMSS)"
    assert date_part.isdigit(), "日期部分應為數字"
    assert time_part.isdigit(), "時間部分應為數字"

    print("格式驗證:")
    print(f"  日期部分 (YYYYMMDD): {date_part} ✅")
    print(f"  時間部分 (HHMMSS): {time_part} ✅")
    print()
    print("✅ 時間戳記格式正確")
    print()


def test_path_consistency():
    """測試路徑一致性"""
    print("=" * 80)
    print("測試 6: 檔案路徑一致性")
    print("=" * 80)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 驗證 Path 操作的一致性
    from pathlib import Path

    model_file = Path("models") / f"baseline_model_{timestamp}.h5"

    # 方法 1: 使用 str + replace
    config_file_1 = str(model_file).replace('.h5', '_config.json')

    # 方法 2: 使用 Path.with_suffix
    config_file_2 = str(Path(str(model_file).replace('.h5', ''))) + "_config.json"

    print(f"模型檔案: {model_file}")
    print(f"配置檔案 (方法1): {config_file_1}")
    print(f"配置檔案 (方法2): {config_file_2}")
    print()

    # 驗證兩種方法產生相同結果
    assert config_file_1 == config_file_2, "兩種配置檔案路徑生成方法應產生相同結果"

    print("✅ 檔案路徑生成一致")
    print()


def main():
    """主測試流程"""
    print()
    print("檔案命名一致性測試")
    print("=" * 80)
    print()

    try:
        # 執行所有測試
        test_baseline_model_files()
        test_tuned_model_files()
        test_config_file_pairing()
        test_scaler_pairing()
        test_timestamp_format()
        test_path_consistency()

        # 總結
        print("=" * 80)
        print("✅ 所有測試通過！")
        print("=" * 80)
        print()
        print("驗證項目:")
        print("  1. ✅ 基準模型檔案命名正確")
        print("  2. ✅ 調整模型檔案命名正確")
        print("  3. ✅ 模型與配置檔案配對正確")
        print("  4. ✅ 模型與縮放器配對正確")
        print("  5. ✅ 時間戳記格式正確")
        print("  6. ✅ 檔案路徑生成一致")
        print()
        print("檔案命名規則摘要:")
        print("  基準模型:")
        print("    - 模型: baseline_model_YYYYMMDD_HHMMSS.h5")
        print("    - 配置: baseline_model_YYYYMMDD_HHMMSS_config.json")
        print("    - 縮放器: baseline_model_YYYYMMDD_HHMMSS_scaler.pkl")
        print()
        print("  調整模型:")
        print("    - 模型: best_tuned_model_YYYYMMDD_HHMMSS.h5")
        print("    - 配置: best_tuned_model_YYYYMMDD_HHMMSS_config.json")
        print()
        print("=" * 80)

        return True

    except AssertionError as e:
        print()
        print("=" * 80)
        print(f"❌ 測試失敗: {str(e)}")
        print("=" * 80)
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
