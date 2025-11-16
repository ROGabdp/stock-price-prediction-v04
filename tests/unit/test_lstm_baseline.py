"""
單元測試：LSTM 基準模型

測試模型建構、輸入/輸出形狀正確性
"""

import pytest
import numpy as np

try:
    import tensorflow as tf
    from tensorflow import keras
    from src.models.lstm_baseline import build_lstm_model
    from src.models.model_builder import build_dynamic_lstm_model

    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False


@pytest.mark.skipif(not TENSORFLOW_AVAILABLE, reason="TensorFlow 未安裝")
class TestLSTMBaseline:
    """測試 LSTM 基準模型"""

    def test_build_lstm_model(self):
        """測試建立基準 LSTM 模型"""
        time_steps = 60
        n_features = 14
        n_classes = 5

        model = build_lstm_model(
            time_steps=time_steps, n_features=n_features, n_classes=n_classes
        )

        assert model is not None
        assert isinstance(model, keras.Model)

    def test_model_input_shape(self):
        """測試模型輸入形狀"""
        time_steps = 60
        n_features = 14

        model = build_lstm_model(time_steps=time_steps, n_features=n_features)

        # 輸入形狀應為 (None, time_steps, n_features)
        expected_input_shape = (None, time_steps, n_features)
        assert model.input_shape == expected_input_shape

    def test_model_output_shape(self):
        """測試模型輸出形狀"""
        time_steps = 60
        n_features = 14
        n_classes = 5

        model = build_lstm_model(
            time_steps=time_steps, n_features=n_features, n_classes=n_classes
        )

        # 輸出形狀應為 (None, n_classes)
        expected_output_shape = (None, n_classes)
        assert model.output_shape == expected_output_shape

    def test_model_compiled(self):
        """測試模型已編譯"""
        model = build_lstm_model(time_steps=60, n_features=14)

        # 檢查模型已編譯（有 optimizer）
        assert model.optimizer is not None

    def test_model_prediction(self):
        """測試模型可以進行預測"""
        time_steps = 60
        n_features = 14
        batch_size = 2

        model = build_lstm_model(time_steps=time_steps, n_features=n_features)

        # 建立測試輸入
        X_test = np.random.randn(batch_size, time_steps, n_features).astype(np.float32)

        # 執行預測
        predictions = model.predict(X_test, verbose=0)

        # 檢查預測結果形狀
        assert predictions.shape == (batch_size, 5)

        # 檢查機率總和為 1（Softmax 輸出）
        prob_sums = predictions.sum(axis=1)
        np.testing.assert_almost_equal(prob_sums, np.ones(batch_size), decimal=5)

    def test_model_trainable_parameters(self):
        """測試模型有可訓練參數"""
        model = build_lstm_model(time_steps=60, n_features=14)

        # 檢查模型有參數
        trainable_params = model.count_params()
        assert trainable_params > 0


@pytest.mark.skipif(not TENSORFLOW_AVAILABLE, reason="TensorFlow 未安裝")
class TestDynamicLSTM:
    """測試動態 LSTM 模型"""

    def test_build_dynamic_model_2_layers(self):
        """測試建立 2 層 LSTM 模型"""
        model = build_dynamic_lstm_model(
            time_steps=60, n_features=14, num_layers=2, dropout_rate=0.2
        )

        assert model is not None
        assert isinstance(model, keras.Model)

    def test_build_dynamic_model_3_layers(self):
        """測試建立 3 層 LSTM 模型"""
        model = build_dynamic_lstm_model(
            time_steps=60, n_features=14, num_layers=3, dropout_rate=0.2
        )

        assert model is not None

    def test_build_dynamic_model_4_layers(self):
        """測試建立 4 層 LSTM 模型"""
        model = build_dynamic_lstm_model(
            time_steps=60, n_features=14, num_layers=4, dropout_rate=0.2
        )

        assert model is not None

    def test_custom_units_per_layer(self):
        """測試自訂每層單元數"""
        units_per_layer = [128, 96, 64]

        model = build_dynamic_lstm_model(
            time_steps=60,
            n_features=14,
            num_layers=3,
            units_per_layer=units_per_layer,
        )

        assert model is not None

    def test_invalid_units_length(self):
        """測試無效的單元數清單長度"""
        with pytest.raises(ValueError):
            build_dynamic_lstm_model(
                time_steps=60,
                n_features=14,
                num_layers=3,
                units_per_layer=[128, 64],  # 長度不匹配
            )

    def test_dropout_rate_validation(self):
        """測試 Dropout 比率驗證"""
        # 有效的 dropout
        model = build_dynamic_lstm_model(
            time_steps=60, n_features=14, num_layers=2, dropout_rate=0.3
        )
        assert model is not None

        # 無效的 dropout（超出範圍）
        with pytest.raises(ValueError):
            build_dynamic_lstm_model(
                time_steps=60, n_features=14, num_layers=2, dropout_rate=0.6
            )

    def test_different_learning_rates(self):
        """測試不同學習率"""
        for lr in [0.001, 0.0005, 0.0001]:
            model = build_dynamic_lstm_model(
                time_steps=60, n_features=14, num_layers=2, learning_rate=lr
            )

            assert model is not None
            # 檢查 optimizer 學習率
            assert abs(float(model.optimizer.learning_rate) - lr) < 1e-6

    def test_dynamic_model_prediction(self):
        """測試動態模型預測"""
        time_steps = 60
        n_features = 14
        batch_size = 2

        model = build_dynamic_lstm_model(
            time_steps=time_steps,
            n_features=n_features,
            num_layers=3,
            units_per_layer=[128, 64, 32],
        )

        # 建立測試輸入
        X_test = np.random.randn(batch_size, time_steps, n_features).astype(np.float32)

        # 執行預測
        predictions = model.predict(X_test, verbose=0)

        # 檢查輸出
        assert predictions.shape == (batch_size, 5)

        # 檢查機率分佈
        assert (predictions >= 0).all()
        assert (predictions <= 1).all()
        np.testing.assert_almost_equal(
            predictions.sum(axis=1), np.ones(batch_size), decimal=5
        )


@pytest.mark.skipif(not TENSORFLOW_AVAILABLE, reason="TensorFlow 未安裝")
class TestModelArchitecture:
    """測試模型架構細節"""

    def test_lstm_layers_exist(self):
        """測試 LSTM 層存在"""
        model = build_lstm_model(time_steps=60, n_features=14)

        # 檢查模型包含 LSTM 層
        lstm_layers = [
            layer for layer in model.layers if isinstance(layer, keras.layers.LSTM)
        ]
        assert len(lstm_layers) > 0

    def test_dropout_layers_exist(self):
        """測試 Dropout 層存在"""
        model = build_lstm_model(time_steps=60, n_features=14)

        # 檢查模型包含 Dropout 層
        dropout_layers = [
            layer for layer in model.layers if isinstance(layer, keras.layers.Dropout)
        ]
        assert len(dropout_layers) > 0

    def test_output_layer_activation(self):
        """測試輸出層使用 Softmax"""
        model = build_lstm_model(time_steps=60, n_features=14)

        # 最後一層應該是 Dense + Softmax
        last_layer = model.layers[-1]
        assert isinstance(last_layer, keras.layers.Dense)
        assert last_layer.activation == keras.activations.softmax

    def test_loss_function(self):
        """測試損失函數"""
        model = build_lstm_model(time_steps=60, n_features=14)

        # 應使用 categorical_crossentropy
        assert model.loss == "categorical_crossentropy"

    def test_optimizer(self):
        """測試優化器"""
        model = build_lstm_model(time_steps=60, n_features=14)

        # 應使用 Adam optimizer
        assert isinstance(model.optimizer, keras.optimizers.Adam)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
