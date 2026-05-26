import importlib.util
import unittest

import numpy as np

from ariel_ml.config import DeepModelConfig


@unittest.skipIf(importlib.util.find_spec("torch") is None, "torch is not installed")
class DeepModelTests(unittest.TestCase):
    def test_cnn1d_sequence_regressor_smoke(self):
        from ariel_ml.deep_models import CNN1DRegressor

        rng = np.random.default_rng(12)
        x = rng.normal(size=(8, 12, 3)).astype(np.float32)
        y = rng.normal(size=(8, 5)).astype(np.float32)
        model = CNN1DRegressor(
            DeepModelConfig(epochs=1, batch_size=4, hidden_size=16, device="cpu")
        )

        model.fit(x, y)
        prediction = model.predict(x[:2])

        self.assertEqual(prediction.mu.shape, (2, 5))
        self.assertEqual(prediction.sigma.shape, (2, 5))
        self.assertTrue(np.all(prediction.sigma > 0))


if __name__ == "__main__":
    unittest.main()
