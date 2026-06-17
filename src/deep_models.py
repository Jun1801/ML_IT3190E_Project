from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA

from config import DeepModelConfig
from metrics import SigmaCalibrator
from models import ModelPrediction


class TorchSequenceRegressor:
    """Base class for deep sequence models on [samples, time, channels] tensors.

    Mirrors the tabular TargetPCARegressor interface:
    - Optional target PCA (``DeepModelConfig.n_components``) reduces output dimension.
    - Gaussian NLL training on PCA components (or raw targets if no PCA).
    - Post-hoc ``SigmaCalibrator`` on held-out validation data (pass ``x_val``/``y_val``
      to ``fit()``), matching what tabular models do.
    """

    def __init__(self, architecture: str, config: DeepModelConfig | None = None) -> None:
        self.architecture = architecture
        self.config = config or DeepModelConfig()
        try:
            import torch
            from torch import nn
        except ImportError as exc:
            raise ImportError("Install torch to use deep learning baselines.") from exc
        self.torch = torch
        self.nn = nn
        self.device = self._select_device()
        self.model = None
        self.n_targets_: int | None = None
        self.pca_: PCA | None = None
        self.sigma_calibrator = SigmaCalibrator(sigma_floor=self.config.sigma_floor)
        torch.manual_seed(self.config.random_state)

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        x_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> "TorchSequenceRegressor":
        x_arr = np.asarray(x, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.float32)
        if x_arr.ndim != 3:
            raise ValueError("Deep sequence models expect X shape [samples, time, channels].")

        # Optional target PCA — reduces output dimension; mirrors tabular TargetPCARegressor.
        if self.config.n_components is not None:
            n_comp = min(self.config.n_components, y_arr.shape[0], y_arr.shape[1])
            self.pca_ = PCA(n_components=n_comp, random_state=self.config.random_state)
            y_train = self.pca_.fit_transform(y_arr).astype(np.float32)
        else:
            y_train = y_arr

        self.n_targets_ = y_train.shape[1]
        self.model = self._build_model(x_arr.shape[1], x_arr.shape[2], self.n_targets_).to(self.device)

        dataset = self.torch.utils.data.TensorDataset(
            self.torch.as_tensor(x_arr),
            self.torch.as_tensor(y_train),
        )
        loader = self.torch.utils.data.DataLoader(
            dataset, batch_size=self.config.batch_size, shuffle=True
        )
        optimizer = self.torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        self.model.train()
        for _ in range(self.config.epochs):
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                optimizer.zero_grad()
                mu, log_sigma = self.model(batch_x)
                sigma = self.torch.nn.functional.softplus(log_sigma) + self.config.sigma_floor
                loss = (0.5 * ((batch_y - mu) / sigma) ** 2 + self.torch.log(sigma)).mean()
                loss.backward()
                optimizer.step()

        # Post-hoc sigma calibration in full target space — mirrors tabular SigmaCalibrator.
        if x_val is not None and y_val is not None:
            uncal = self._predict_uncalibrated(x_val)
            self.sigma_calibrator.fit(
                np.asarray(y_val, dtype=float), uncal.mu, uncal.sigma
            )

        return self

    def _predict_uncalibrated(self, x: np.ndarray) -> ModelPrediction:
        """Predict in full target space with uncalibrated sigma."""
        if self.model is None:
            raise RuntimeError("Model must be fitted before prediction.")
        x_arr = np.asarray(x, dtype=np.float32)
        self.model.eval()
        with self.torch.inference_mode():
            tensor = self.torch.as_tensor(x_arr).to(self.device)
            mu_t, log_sigma_t = self.model(tensor)
            sigma_t = self.torch.nn.functional.softplus(log_sigma_t) + self.config.sigma_floor
        mu_np = mu_t.cpu().numpy().astype(float)
        sigma_np = sigma_t.cpu().numpy().astype(float)

        if self.pca_ is not None:
            mu_full = self.pca_.inverse_transform(mu_np)
            # Propagate uncertainty through PCA inverse: var_j = sum_k (sigma_k * components_kj)^2
            sigma_full = np.sqrt(np.maximum(sigma_np ** 2 @ self.pca_.components_ ** 2, 0.0))
            return ModelPrediction(mu=mu_full, sigma=sigma_full)
        return ModelPrediction(mu=mu_np, sigma=sigma_np)

    def predict(self, x: np.ndarray) -> ModelPrediction:
        pred = self._predict_uncalibrated(x)
        return ModelPrediction(
            mu=pred.mu,
            sigma=self.sigma_calibrator.transform(pred.sigma),
        )

    def save_weights(self, path) -> None:
        if self.model is None:
            raise RuntimeError("Model must be fitted before saving weights.")
        self.torch.save(self.model.state_dict(), path)

    def load_weights(self, path, n_time: int, n_channels: int, n_targets: int) -> "TorchSequenceRegressor":
        self.n_targets_ = n_targets
        self.model = self._build_model(n_time, n_channels, n_targets).to(self.device)
        self.model.load_state_dict(self.torch.load(path, map_location=self.device))
        self.model.eval()
        return self

    def _select_device(self):
        if self.config.device != "auto":
            return self.torch.device(self.config.device)
        if self.torch.cuda.is_available():
            return self.torch.device("cuda")
        if hasattr(self.torch.backends, "mps") and self.torch.backends.mps.is_available():
            return self.torch.device("mps")
        return self.torch.device("cpu")

    def _build_model(self, n_time: int, n_channels: int, n_targets: int):
        nn, config = self.nn, self.config
        if self.architecture == "cnn1d":
            return _build_cnn1d(nn, n_channels, n_targets, config)
        if self.architecture == "tcn":
            return _build_tcn(nn, n_channels, n_targets, config)
        if self.architecture == "lstm":
            return _build_rnn(nn, n_channels, n_targets, config, cell="lstm")
        if self.architecture == "gru":
            return _build_rnn(nn, n_channels, n_targets, config, cell="gru")
        if self.architecture == "transformer":
            return _build_transformer(nn, n_channels, n_targets, config)
        if self.architecture == "autoencoder_mlp":
            return _build_autoencoder_mlp(nn, n_time, n_channels, n_targets, config)
        raise ValueError(f"Unknown deep architecture: {self.architecture}")


# ---------------------------------------------------------------------------
# Public subclass aliases — preserve the same external names
# ---------------------------------------------------------------------------

class CNN1DRegressor(TorchSequenceRegressor):
    def __init__(self, config: DeepModelConfig | None = None) -> None:
        super().__init__("cnn1d", config)


class TCNRegressor(TorchSequenceRegressor):
    def __init__(self, config: DeepModelConfig | None = None) -> None:
        super().__init__("tcn", config)


class LSTMRegressor(TorchSequenceRegressor):
    def __init__(self, config: DeepModelConfig | None = None) -> None:
        super().__init__("lstm", config)


class GRURegressor(TorchSequenceRegressor):
    def __init__(self, config: DeepModelConfig | None = None) -> None:
        super().__init__("gru", config)


class TransformerSequenceRegressor(TorchSequenceRegressor):
    def __init__(self, config: DeepModelConfig | None = None) -> None:
        super().__init__("transformer", config)


class AutoencoderMLPRegressor(TorchSequenceRegressor):
    def __init__(self, config: DeepModelConfig | None = None) -> None:
        super().__init__("autoencoder_mlp", config)


# ---------------------------------------------------------------------------
# nn.Module factories — each returns a proper torch.nn.Module instance.
# Classes are defined inline so `nn` (torch.nn) is available as a closure
# without requiring a top-level torch import.
# ---------------------------------------------------------------------------

def _build_cnn1d(nn, n_channels: int, n_targets: int, config: DeepModelConfig):
    hidden = config.hidden_size

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(n_channels, hidden // 2, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.Conv1d(hidden // 2, hidden, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Dropout(config.dropout),
                nn.Linear(hidden, n_targets * 2),
            )

        def forward(self, x):
            return self.net(x.transpose(1, 2)).chunk(2, dim=-1)

    return _Model()


def _build_tcn(nn, n_channels: int, n_targets: int, config: DeepModelConfig):
    hidden = config.hidden_size

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(n_channels, hidden // 2, kernel_size=3, padding=1, dilation=1),
                nn.ReLU(),
                nn.Conv1d(hidden // 2, hidden, kernel_size=3, padding=2, dilation=2),
                nn.ReLU(),
                nn.Conv1d(hidden, hidden, kernel_size=3, padding=4, dilation=4),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Dropout(config.dropout),
                nn.Linear(hidden, n_targets * 2),
            )

        def forward(self, x):
            return self.net(x.transpose(1, 2)).chunk(2, dim=-1)

    return _Model()


def _build_rnn(nn, n_channels: int, n_targets: int, config: DeepModelConfig, *, cell: str):
    hidden = config.hidden_size
    rnn_cls = nn.LSTM if cell == "lstm" else nn.GRU

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.rnn = rnn_cls(n_channels, hidden, batch_first=True)
            self.dropout = nn.Dropout(config.dropout)
            self.head = nn.Linear(hidden, n_targets * 2)

        def forward(self, x):
            out, _ = self.rnn(x)
            return self.head(self.dropout(out[:, -1, :])).chunk(2, dim=-1)

    return _Model()


def _build_transformer(nn, n_channels: int, n_targets: int, config: DeepModelConfig):
    hidden = config.hidden_size

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_proj = nn.Linear(n_channels, hidden)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden,
                nhead=4,
                dim_feedforward=hidden * 2,
                dropout=config.dropout,
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
            self.head = nn.Linear(hidden, n_targets * 2)

        def forward(self, x):
            return self.head(self.encoder(self.input_proj(x)).mean(dim=1)).chunk(2, dim=-1)

    return _Model()


def _build_autoencoder_mlp(nn, n_time: int, n_channels: int, n_targets: int, config: DeepModelConfig):
    hidden = config.hidden_size
    flat = n_time * n_channels

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Flatten(),
                nn.Linear(flat, hidden * 2),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(hidden * 2, hidden),
                nn.ReLU(),
                nn.Linear(hidden, n_targets * 2),
            )

        def forward(self, x):
            return self.net(x).chunk(2, dim=-1)

    return _Model()
