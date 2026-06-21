import importlib.util
import unittest

import numpy as np

from config import DeepModelConfig


@unittest.skipIf(importlib.util.find_spec("torch") is None, "torch is not installed")
class DeepModelTests(unittest.TestCase):
    def _make_data(self, n=12, n_time=10, n_channels=3, n_targets=8):
        rng = np.random.default_rng(42)
        x = rng.normal(size=(n, n_time, n_channels)).astype(np.float32)
        y = rng.normal(size=(n, n_targets)).astype(np.float32)
        return x, y

    def test_cnn1d_basic_shapes(self):
        from deep_models import CNN1DRegressor

        x, y = self._make_data()
        model = CNN1DRegressor(DeepModelConfig(epochs=1, batch_size=4, hidden_size=16, device="cpu"))
        model.fit(x, y)
        pred = model.predict(x[:2])

        self.assertEqual(pred.mu.shape, (2, y.shape[1]))
        self.assertEqual(pred.sigma.shape, (2, y.shape[1]))
        self.assertTrue(np.all(pred.sigma > 0))

    def test_fit_with_pca_reduces_internal_targets(self):
        from deep_models import CNN1DRegressor

        x, y = self._make_data(n=16, n_targets=10)
        n_comp = 4
        model = CNN1DRegressor(
            DeepModelConfig(epochs=1, batch_size=4, hidden_size=16, device="cpu", n_components=n_comp)
        )
        model.fit(x, y)

        # PCA fitted; internal targets dimension is n_comp
        self.assertIsNotNone(model.pca_)
        self.assertEqual(model.n_targets_, n_comp)
        # Output is still inverse-transformed back to full target space
        pred = model.predict(x[:3])
        self.assertEqual(pred.mu.shape, (3, y.shape[1]))
        self.assertEqual(pred.sigma.shape, (3, y.shape[1]))
        self.assertTrue(np.all(pred.sigma > 0))

    def test_fit_with_val_calibrates_sigma(self):
        from deep_models import GRURegressor

        x, y = self._make_data(n=16)
        x_tr, x_va = x[:12], x[12:]
        y_tr, y_va = y[:12], y[12:]

        model_no_cal = GRURegressor(
            DeepModelConfig(epochs=2, batch_size=4, hidden_size=16, device="cpu")
        )
        model_no_cal.fit(x_tr, y_tr)

        model_cal = GRURegressor(
            DeepModelConfig(epochs=2, batch_size=4, hidden_size=16, device="cpu")
        )
        model_cal.fit(x_tr, y_tr, x_val=x_va, y_val=y_va)

        # With calibration the calibrator scale should differ from default 1.0
        self.assertNotEqual(model_cal.sigma_calibrator.scale_, 1.0)

    def test_save_load_weights_with_pca_reproduces_predictions(self):
        import tempfile
        from pathlib import Path

        from deep_models import CNN1DRegressor

        x, y = self._make_data(n=16, n_time=10, n_channels=3, n_targets=10)
        cfg = DeepModelConfig(epochs=2, batch_size=4, hidden_size=16, device="cpu", n_components=4)
        model = CNN1DRegressor(cfg)
        model.fit(x, y)
        ref = model._predict_uncalibrated(x[:3])  # uncalibrated -> independent of saved calibrator

        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "w.pt"
            model.save_weights(p)
            # Reload WITHOUT fit: pass y_train so the (deterministic) target PCA is refit.
            loaded = CNN1DRegressor(cfg).load_weights(p, n_time=10, n_channels=3, y_train=y)

        self.assertIsNotNone(loaded.pca_)
        self.assertEqual(loaded.n_targets_, 4)
        out = loaded._predict_uncalibrated(x[:3])
        self.assertEqual(out.mu.shape, (3, y.shape[1]))         # back in full 10-dim target space
        np.testing.assert_allclose(out.mu, ref.mu, rtol=1e-5, atol=1e-6)

    def test_load_weights_requires_y_train_when_pca_used(self):
        import tempfile
        from pathlib import Path

        from deep_models import CNN1DRegressor

        x, y = self._make_data(n=16, n_targets=10)
        cfg = DeepModelConfig(epochs=1, batch_size=4, hidden_size=16, device="cpu", n_components=4)
        model = CNN1DRegressor(cfg)
        model.fit(x, y)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "w.pt"
            model.save_weights(p)
            # Must NOT silently return PCA-space predictions: raise when y_train is missing.
            with self.assertRaises(ValueError):
                CNN1DRegressor(cfg).load_weights(p, n_time=10, n_channels=3, n_targets=4)

    def test_all_architectures_smoke(self):
        from deep_models import (
            AutoencoderMLPRegressor,
            CNN1DRegressor,
            GRURegressor,
            LSTMRegressor,
            TCNRegressor,
            TransformerSequenceRegressor,
        )

        x, y = self._make_data(n=10, n_time=12, n_channels=3, n_targets=6)
        for cls in [CNN1DRegressor, TCNRegressor, LSTMRegressor, GRURegressor,
                    TransformerSequenceRegressor, AutoencoderMLPRegressor]:
            with self.subTest(cls=cls.__name__):
                model = cls(DeepModelConfig(epochs=1, batch_size=4, hidden_size=16, device="cpu"))
                model.fit(x, y)
                pred = model.predict(x[:2])
                self.assertEqual(pred.mu.shape, (2, y.shape[1]))
                self.assertTrue(np.all(pred.sigma > 0))


if __name__ == "__main__":
    unittest.main()
