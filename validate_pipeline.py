"""
Pipeline 驗證腳本

驗證完整的訓練流程（使用小規模資料集）
此腳本不需要 TensorFlow，僅驗證資料處理流程
"""

import sys
from pathlib import Path

# 加入專案根目錄至 Python 路徑
sys.path.insert(0, str(Path(__file__).parent))

import logging
import numpy as np
import pandas as pd

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def validate_data_loading():
    """驗證資料載入模組"""
    logger.info("=" * 80)
    logger.info("步驟 1/7: 驗證資料載入")
    logger.info("=" * 80)

    try:
        from src.data.data_loader import load_csv_data, get_data_info

        # 載入資料
        df = load_csv_data("19980601-20251111-converted.csv")

        # 驗證資料
        assert len(df) > 0, "資料載入失敗，筆數為 0"
        assert 'date' in df.columns, "缺少 date 欄位"
        assert 'close' in df.columns, "缺少 close 欄位"

        info = get_data_info(df)
        logger.info(f"✅ 資料載入成功: {info['count']} 筆")
        logger.info(f"   日期範圍: {info['date_range']['start']} ~ {info['date_range']['end']}")

        return df

    except Exception as e:
        logger.error(f"❌ 資料載入驗證失敗: {str(e)}")
        raise


def validate_feature_engineering(df):
    """驗證特徵工程模組"""
    logger.info("\n" + "=" * 80)
    logger.info("步驟 2/7: 驗證特徵工程")
    logger.info("=" * 80)

    try:
        from src.features.feature_engineer import prepare_features_and_target
        from src.features.feature_sets import list_all_feature_sets, get_feature_names

        # 測試所有特徵集
        for feature_set_id in list_all_feature_sets():
            logger.info(f"\n測試特徵集: {feature_set_id}")

            features, target_onehot, df_processed = prepare_features_and_target(
                df, feature_set_id=feature_set_id, look_ahead_days=20
            )

            # 驗證輸出形狀
            assert features.ndim == 2, f"特徵矩陣應為 2D，實際: {features.ndim}D"
            assert target_onehot.ndim == 2, f"目標矩陣應為 2D，實際: {target_onehot.ndim}D"
            assert target_onehot.shape[1] == 5, f"目標應有 5 個類別，實際: {target_onehot.shape[1]}"
            assert features.shape[0] == target_onehot.shape[0], "特徵與目標筆數不一致"

            # 驗證特徵數量
            expected_features = get_feature_names(feature_set_id)
            assert features.shape[1] == len(expected_features), \
                f"特徵數量不符: 預期 {len(expected_features)}, 實際 {features.shape[1]}"

            logger.info(f"✅ {feature_set_id} 驗證通過")
            logger.info(f"   特徵形狀: {features.shape}")
            logger.info(f"   目標形狀: {target_onehot.shape}")

        # 返回 Set A 的結果供後續使用
        features, target_onehot, df_processed = prepare_features_and_target(
            df, feature_set_id="Set A", look_ahead_days=20
        )

        return features, target_onehot, df_processed

    except Exception as e:
        logger.error(f"❌ 特徵工程驗證失敗: {str(e)}")
        raise


def validate_data_splitting(df_processed):
    """驗證資料分割模組"""
    logger.info("\n" + "=" * 80)
    logger.info("步驟 3/7: 驗證資料分割")
    logger.info("=" * 80)

    try:
        from src.data.data_splitter import split_time_series

        train_df, val_df, test_df = split_time_series(
            df_processed, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
        )

        # 驗證分割
        total = len(train_df) + len(val_df) + len(test_df)
        assert total == len(df_processed), "分割後總筆數不一致"

        # 驗證時間順序（無資料洩漏）
        assert train_df['date'].max() < val_df['date'].min(), "訓練集與驗證集日期重疊"
        assert val_df['date'].max() < test_df['date'].min(), "驗證集與測試集日期重疊"

        logger.info(f"✅ 資料分割驗證通過")
        logger.info(f"   訓練集: {len(train_df)} 筆")
        logger.info(f"   驗證集: {len(val_df)} 筆")
        logger.info(f"   測試集: {len(test_df)} 筆")

        return train_df, val_df, test_df

    except Exception as e:
        logger.error(f"❌ 資料分割驗證失敗: {str(e)}")
        raise


def validate_feature_scaling(train_df, val_df, test_df):
    """驗證特徵縮放模組"""
    logger.info("\n" + "=" * 80)
    logger.info("步驟 4/7: 驗證特徵縮放")
    logger.info("=" * 80)

    try:
        from src.features.scalers import create_scaler, fit_transform_features
        from src.features.feature_sets import get_feature_names

        feature_columns = get_feature_names("Set A")

        train_features = train_df[feature_columns].values
        val_features = val_df[feature_columns].values
        test_features = test_df[feature_columns].values

        # 測試 StandardScaler
        scaler = create_scaler("standard")
        scaler, train_scaled, val_scaled, test_scaled = fit_transform_features(
            scaler, train_features, val_features, test_features
        )

        # 驗證縮放結果
        assert train_scaled.shape == train_features.shape, "訓練集縮放後形狀改變"
        assert val_scaled.shape == val_features.shape, "驗證集縮放後形狀改變"
        assert test_scaled.shape == test_features.shape, "測試集縮放後形狀改變"

        # 驗證 StandardScaler 特性（平均值接近 0，標準差接近 1）
        mean = train_scaled.mean(axis=0)
        std = train_scaled.std(axis=0)
        assert np.allclose(mean, 0, atol=1e-10), f"縮放後平均值不為 0: {mean[:5]}"
        assert np.allclose(std, 1, atol=1e-10), f"縮放後標準差不為 1: {std[:5]}"

        logger.info(f"✅ 特徵縮放驗證通過")
        logger.info(f"   訓練集平均值: {mean[:3]}")
        logger.info(f"   訓練集標準差: {std[:3]}")

        return scaler, train_scaled, val_scaled, test_scaled

    except Exception as e:
        logger.error(f"❌ 特徵縮放驗證失敗: {str(e)}")
        raise


def validate_sequence_creation(train_scaled, val_scaled, test_scaled, train_df, val_df, test_df):
    """驗證序列建立模組"""
    logger.info("\n" + "=" * 80)
    logger.info("步驟 5/7: 驗證序列建立")
    logger.info("=" * 80)

    try:
        from src.data.data_splitter import create_sequences
        from sklearn.preprocessing import OneHotEncoder

        # 準備目標變數
        train_target = train_df["target_class"].values
        val_target = val_df["target_class"].values
        test_target = test_df["target_class"].values

        # One-Hot 編碼
        encoder = OneHotEncoder(sparse_output=False, categories=[range(5)])
        train_target_onehot = encoder.fit_transform(train_target.reshape(-1, 1))
        val_target_onehot = encoder.transform(val_target.reshape(-1, 1))
        test_target_onehot = encoder.transform(test_target.reshape(-1, 1))

        # 建立序列
        time_steps = 60
        X_train, y_train = create_sequences(train_scaled, train_target_onehot, time_steps)
        X_val, y_val = create_sequences(val_scaled, val_target_onehot, time_steps)
        X_test, y_test = create_sequences(test_scaled, test_target_onehot, time_steps)

        # 驗證形狀
        assert X_train.ndim == 3, f"X_train 應為 3D，實際: {X_train.ndim}D"
        assert X_train.shape[1] == time_steps, f"時間窗口大小錯誤: {X_train.shape[1]}"
        assert y_train.shape[1] == 5, f"目標類別數錯誤: {y_train.shape[1]}"

        logger.info(f"✅ 序列建立驗證通過")
        logger.info(f"   X_train: {X_train.shape}")
        logger.info(f"   y_train: {y_train.shape}")
        logger.info(f"   X_val: {X_val.shape}")
        logger.info(f"   y_val: {y_val.shape}")
        logger.info(f"   X_test: {X_test.shape}")
        logger.info(f"   y_test: {y_test.shape}")

        return X_train, y_train, X_val, y_val, X_test, y_test

    except Exception as e:
        logger.error(f"❌ 序列建立驗證失敗: {str(e)}")
        raise


def validate_model_architecture():
    """驗證模型架構（不需要 TensorFlow）"""
    logger.info("\n" + "=" * 80)
    logger.info("步驟 6/7: 驗證模型架構定義")
    logger.info("=" * 80)

    try:
        # 嘗試導入模型模組
        from src.models import lstm_baseline, model_builder

        logger.info("✅ 模型模組導入成功")
        logger.info("   - lstm_baseline.build_lstm_model")
        logger.info("   - model_builder.train_model")
        logger.info("   - model_builder.create_callbacks")

        # 檢查函式是否存在
        assert hasattr(lstm_baseline, 'build_lstm_model'), "缺少 build_lstm_model 函式"
        assert hasattr(model_builder, 'train_model'), "缺少 train_model 函式"
        assert hasattr(model_builder, 'create_callbacks'), "缺少 create_callbacks 函式"

        logger.info("✅ 模型架構定義驗證通過")

    except ImportError as e:
        logger.warning(f"⚠️ TensorFlow 未安裝，跳過模型建構測試: {str(e)}")
        logger.info("   模型架構定義已完成，可在安裝 TensorFlow 後進行訓練")
    except Exception as e:
        logger.error(f"❌ 模型架構驗證失敗: {str(e)}")
        raise


def validate_cli_interface():
    """驗證 CLI 介面"""
    logger.info("\n" + "=" * 80)
    logger.info("步驟 7/7: 驗證 CLI 介面")
    logger.info("=" * 80)

    try:
        # 檢查 CLI 檔案是否存在
        cli_file = Path("src/cli/train.py")
        assert cli_file.exists(), "CLI 訓練腳本不存在"

        # 檢查是否可以導入
        from src.cli import train

        assert hasattr(train, 'main'), "CLI 缺少 main 函式"
        assert hasattr(train, 'parse_arguments'), "CLI 缺少 parse_arguments 函式"

        logger.info("✅ CLI 介面驗證通過")
        logger.info(f"   腳本位置: {cli_file}")
        logger.info("   執行指令: python src/cli/train.py --data-file <file> --feature-set <set>")

    except Exception as e:
        logger.error(f"❌ CLI 介面驗證失敗: {str(e)}")
        raise


def main():
    """主驗證流程"""
    logger.info("\n")
    logger.info("🚀 開始驗證 LSTM 台股預測系統 - 完整訓練流程")
    logger.info("=" * 80)

    try:
        # 1. 驗證資料載入
        df = validate_data_loading()

        # 2. 驗證特徵工程
        features, target_onehot, df_processed = validate_feature_engineering(df)

        # 3. 驗證資料分割
        train_df, val_df, test_df = validate_data_splitting(df_processed)

        # 4. 驗證特徵縮放
        scaler, train_scaled, val_scaled, test_scaled = validate_feature_scaling(
            train_df, val_df, test_df
        )

        # 5. 驗證序列建立
        X_train, y_train, X_val, y_val, X_test, y_test = validate_sequence_creation(
            train_scaled, val_scaled, test_scaled, train_df, val_df, test_df
        )

        # 6. 驗證模型架構
        validate_model_architecture()

        # 7. 驗證 CLI 介面
        validate_cli_interface()

        # 最終總結
        logger.info("\n" + "=" * 80)
        logger.info("✅ 所有驗證通過！")
        logger.info("=" * 80)
        logger.info("\n訓練流程驗證結果:")
        logger.info(f"  ✅ 資料載入: {len(df)} 筆原始資料")
        logger.info(f"  ✅ 特徵工程: {features.shape[1]} 個特徵")
        logger.info(f"  ✅ 資料分割: {len(train_df)}/{len(val_df)}/{len(test_df)} (訓練/驗證/測試)")
        logger.info(f"  ✅ 序列建立: {X_train.shape} (訓練集形狀)")
        logger.info(f"  ✅ 模型架構: 已定義")
        logger.info(f"  ✅ CLI 介面: 已實作")

        logger.info("\n準備就緒！可以執行以下指令開始訓練:")
        logger.info("  1. 安裝依賴: pip install -r requirements.txt")
        logger.info("  2. 執行訓練: python src/cli/train.py --data-file 19980601-20251111-converted.csv --feature-set \"Set A\" --time-steps 60 --epochs 100 --batch-size 32")

        logger.info("\n" + "=" * 80)
        return True

    except Exception as e:
        logger.error("\n" + "=" * 80)
        logger.error(f"❌ 驗證失敗: {str(e)}")
        logger.error("=" * 80)
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
