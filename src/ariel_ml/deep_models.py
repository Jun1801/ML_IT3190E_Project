from __future__ import annotations

import numpy as np

from ariel_ml.config import DeepModelConfig
from ariel_ml.models import ModelPrediction


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
        torch.manual_seed(self.config.random_state)

    def fit(self, x: np.ndarray, y: np.ndarray) -> "TorchSequenceRegressor":
        x_arr = np.asarray(x, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.float32)
        if x_arr.ndim != 3:
            raise ValueError("Deep sequence models expect X shape [samples, time, channels].")
        self.n_targets_ = y_arr.shape[1]
        self.model = self._build_model(x_arr.shape[1], x_arr.shape[2], y_arr.shape[1]).to(self.device)

        dataset = self.torch.utils.data.TensorDataset(
            self.torch.as_tensor(x_arr),
            self.torch.as_tensor(y_arr),
        )
        loader = self.torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
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
                loss = 0.5 * (((batch_y - mu) / sigma) ** 2 + 2.0 * self.torch.log(sigma))
                loss = loss.mean()
                loss.backward()
                optimizer.step()
        return self

    def predict(self, x: np.ndarray) -> ModelPrediction:
        if self.model is None:
            raise RuntimeError("Model must be fitted before prediction.")
        x_arr = np.asarray(x, dtype=np.float32)
        self.model.eval()
        with self.torch.inference_mode():
            tensor = self.torch.as_tensor(x_arr).to(self.device)
            mu, log_sigma = self.model(tensor)
            sigma = self.torch.nn.functional.softplus(log_sigma) + self.config.sigma_floor
        return ModelPrediction(mu=mu.cpu().numpy(), sigma=sigma.cpu().numpy())

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
        if self.architecture == "cnn1d":
            return _CNN1DRegressor(self.nn, n_channels, n_targets, self.config)
        if self.architecture == "tcn":
            return _TCNRegressor(self.nn, n_channels, n_targets, self.config)
        if self.architecture == "lstm":
            return _RNNRegressor(self.nn, n_channels, n_targets, self.config, cell="lstm")
        if self.architecture == "gru":
            return _RNNRegressor(self.nn, n_channels, n_targets, self.config, cell="gru")
        if self.architecture == "transformer":
            return _TransformerRegressor(self.nn, n_channels, n_targets, self.config)
        if self.architecture == "autoencoder_mlp":
            return _AutoencoderMLPRegressor(self.nn, n_time, n_channels, n_targets, self.config)
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


class _HeadMixin:
    def _head(self, nn, hidden_size: int, n_targets: int):
        return nn.Linear(hidden_size, n_targets * 2)

    def _split(self, output):
        return output.chunk(2, dim=-1)


class _CNN1DRegressor(_HeadMixin):
    def __init__(self, nn, n_channels: int, n_targets: int, config: DeepModelConfig) -> None:
        super().__init__()
        hidden = config.hidden_size
        self.net = nn.Sequential(
            nn.Conv1d(n_channels, hidden // 2, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(hidden // 2, hidden, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(config.dropout),
            self._head(nn, hidden, n_targets),
        )

    def __call__(self, x):
        return self.forward(x)

    def to(self, device):
        self.net = self.net.to(device)
        return self

    def parameters(self):
        return self.net.parameters()

    def train(self):
        self.net.train()

    def eval(self):
        self.net.eval()

    def forward(self, x):
        return self._split(self.net(x.transpose(1, 2)))


class _TCNRegressor(_CNN1DRegressor):
    def __init__(self, nn, n_channels: int, n_targets: int, config: DeepModelConfig) -> None:
        _HeadMixin.__init__(self)
        hidden = config.hidden_size
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
            self._head(nn, hidden, n_targets),
        )


class _RNNRegressor(_HeadMixin):
    def __init__(self, nn, n_channels: int, n_targets: int, config: DeepModelConfig, *, cell: str) -> None:
        super().__init__()
        rnn_cls = nn.LSTM if cell == "lstm" else nn.GRU
        self.rnn = rnn_cls(n_channels, config.hidden_size, batch_first=True, dropout=0.0)
        self.dropout = nn.Dropout(config.dropout)
        self.head = self._head(nn, config.hidden_size, n_targets)

    def to(self, device):
        self.rnn = self.rnn.to(device)
        self.dropout = self.dropout.to(device)
        self.head = self.head.to(device)
        return self

    def parameters(self):
        return list(self.rnn.parameters()) + list(self.head.parameters())

    def train(self):
        self.rnn.train()
        self.head.train()

    def eval(self):
        self.rnn.eval()
        self.head.eval()

    def __call__(self, x):
        output, _ = self.rnn(x)
        return self._split(self.head(self.dropout(output[:, -1, :])))


class _TransformerRegressor(_HeadMixin):
    def __init__(self, nn, n_channels: int, n_targets: int, config: DeepModelConfig) -> None:
        super().__init__()
        hidden = config.hidden_size
        self.input = nn.Linear(n_channels, hidden)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=4,
            dim_feedforward=hidden * 2,
            dropout=config.dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.head = self._head(nn, hidden, n_targets)

    def to(self, device):
        self.input = self.input.to(device)
        self.encoder = self.encoder.to(device)
        self.head = self.head.to(device)
        return self

    def parameters(self):
        return list(self.input.parameters()) + list(self.encoder.parameters()) + list(self.head.parameters())

    def train(self):
        self.input.train()
        self.encoder.train()
        self.head.train()

    def eval(self):
        self.input.eval()
        self.encoder.eval()
        self.head.eval()

    def __call__(self, x):
        encoded = self.encoder(self.input(x))
        return self._split(self.head(encoded.mean(dim=1)))


class _AutoencoderMLPRegressor(_HeadMixin):
    def __init__(self, nn, n_time: int, n_channels: int, n_targets: int, config: DeepModelConfig) -> None:
        super().__init__()
        hidden = config.hidden_size
        flat = n_time * n_channels
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, hidden * 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            self._head(nn, hidden, n_targets),
        )

    def to(self, device):
        self.net = self.net.to(device)
        return self

    def parameters(self):
        return self.net.parameters()

    def train(self):
        self.net.train()

    def eval(self):
        self.net.eval()

    def __call__(self, x):
        return self._split(self.net(x))
