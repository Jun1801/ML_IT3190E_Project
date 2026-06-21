from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA

from config import DeepModelConfig
from metrics import SigmaCalibrator
from models import ModelPrediction


class TorchSequenceRegressor:
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

        x_val_t = y_val_t = None
        use_early_stop = x_val is not None and y_val is not None and self.config.patience > 0
        if use_early_stop:
            y_val_arr = np.asarray(y_val, dtype=np.float32)
            y_val_pca = (
                self.pca_.transform(y_val_arr).astype(np.float32)
                if self.pca_ is not None
                else y_val_arr
            )
            x_val_t = self.torch.as_tensor(np.asarray(x_val, dtype=np.float32)).to(self.device)
            y_val_t = self.torch.as_tensor(y_val_pca).to(self.device)

        best_val_loss = float("inf")
        patience_counter = 0
        best_state: dict | None = None

        for _ in range(self.config.epochs):
            self.model.train()
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                optimizer.zero_grad()
                mu, log_sigma = self.model(batch_x)
                sigma = self.torch.nn.functional.softplus(log_sigma) + self.config.sigma_floor
                loss = (0.5 * ((batch_y - mu) / sigma) ** 2 + self.torch.log(sigma)).mean()
                loss.backward()
                optimizer.step()

            if use_early_stop:
                self.model.eval()
                with self.torch.inference_mode():
                    mu_v, log_sigma_v = self.model(x_val_t)
                    sigma_v = self.torch.nn.functional.softplus(log_sigma_v) + self.config.sigma_floor
                    val_loss = (
                        0.5 * ((y_val_t - mu_v) / sigma_v) ** 2 + self.torch.log(sigma_v)
                    ).mean().item()
                if val_loss < best_val_loss - 1e-6:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                else:
                    patience_counter += 1
                    if patience_counter >= self.config.patience:
                        break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        if x_val is not None and y_val is not None:
            uncal = self._predict_uncalibrated(x_val)
            self.sigma_calibrator.fit(
                np.asarray(y_val, dtype=float), uncal.mu, uncal.sigma
            )

        return self

    def _predict_uncalibrated(self, x: np.ndarray) -> ModelPrediction:
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

    def load_weights(
        self,
        path,
        n_time: int,
        n_channels: int,
        n_targets: int | None = None,
        *,
        y_train: np.ndarray | None = None,
    ) -> "TorchSequenceRegressor":
        if y_train is not None and self.config.n_components is not None:
            y_arr = np.asarray(y_train, dtype=np.float32)
            n_comp = min(self.config.n_components, y_arr.shape[0], y_arr.shape[1])
            self.pca_ = PCA(n_components=n_comp, random_state=self.config.random_state).fit(y_arr)
            n_targets = n_comp
        if n_targets is None:
            raise ValueError("load_weights requires n_targets, or y_train when n_components is set.")
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
            return _build_autoencoder_mlp(nn, n_channels, n_targets, config)
        raise ValueError(f"Unknown deep architecture: {self.architecture}")


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


def _build_cnn1d(nn, n_channels: int, n_targets: int, config: DeepModelConfig):
    hidden = config.hidden_size

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(n_channels, hidden // 2, kernel_size=7, padding=3),
                nn.BatchNorm1d(hidden // 2),
                nn.ReLU(),
                nn.Conv1d(hidden // 2, hidden, kernel_size=5, padding=2),
                nn.BatchNorm1d(hidden),
                nn.ReLU(),
                nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
                nn.BatchNorm1d(hidden),
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
                nn.BatchNorm1d(hidden // 2),
                nn.ReLU(),
                nn.Conv1d(hidden // 2, hidden, kernel_size=3, padding=2, dilation=2),
                nn.BatchNorm1d(hidden),
                nn.ReLU(),
                nn.Conv1d(hidden, hidden, kernel_size=3, padding=4, dilation=4),
                nn.BatchNorm1d(hidden),
                nn.ReLU(),
                nn.Conv1d(hidden, hidden, kernel_size=3, padding=8, dilation=8),
                nn.BatchNorm1d(hidden),
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
            self.rnn = rnn_cls(
                n_channels, hidden, num_layers=2, batch_first=True, dropout=config.dropout
            )
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
                dim_feedforward=hidden * 4,
                dropout=config.dropout,
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)
            self.head = nn.Linear(hidden, n_targets * 2)

        def forward(self, x):
            return self.head(self.encoder(self.input_proj(x)).mean(dim=1)).chunk(2, dim=-1)

    return _Model()


def _build_autoencoder_mlp(nn, n_channels: int, n_targets: int, config: DeepModelConfig):
    hidden = config.hidden_size
    latent_time = 16

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv1d(n_channels, hidden, kernel_size=5, padding=2),
                nn.BatchNorm1d(hidden),
                nn.ReLU(),
                nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
                nn.BatchNorm1d(hidden),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(latent_time),
                nn.Flatten(),
            )
            self.head = nn.Sequential(
                nn.Dropout(config.dropout),
                nn.Linear(hidden * latent_time, hidden),
                nn.ReLU(),
                nn.Linear(hidden, n_targets * 2),
            )

        def forward(self, x):
            return self.head(self.encoder(x.transpose(1, 2))).chunk(2, dim=-1)

    return _Model()
