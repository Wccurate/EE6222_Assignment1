"""Autoencoder dimensionality reduction wrapper."""

from __future__ import annotations

import numpy as np

from . import DRMethod


class AutoEncoderDR(DRMethod):
    """Lightweight MLP autoencoder used as nonlinear DR."""

    def __init__(
        self,
        n_components: int,
        hidden_dim: int = 256,
        epochs: int = 10,
        batch_size: int = 256,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        device: str = "cpu",
        max_train_samples: int | None = None,
    ) -> None:
        self.n_components = n_components
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.device = device
        self.max_train_samples = max_train_samples

        self.model = None
        self._torch = None

    def _build_model(self, input_dim: int):
        import torch
        import torch.nn as nn

        class _AE(nn.Module):
            def __init__(self, in_dim: int, hidden: int, latent: int) -> None:
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(in_dim, hidden),
                    nn.ReLU(),
                    nn.Linear(hidden, latent),
                )
                self.decoder = nn.Sequential(
                    nn.Linear(latent, hidden),
                    nn.ReLU(),
                    nn.Linear(hidden, in_dim),
                    nn.Sigmoid(),
                )

            def forward(self, x):
                z = self.encoder(x)
                x_hat = self.decoder(z)
                return x_hat

        return _AE(input_dim, self.hidden_dim, self.n_components)

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "AutoEncoderDR":
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        self._torch = torch

        X_train = X.astype(np.float32)
        if self.max_train_samples is not None and self.max_train_samples < X_train.shape[0]:
            idx = np.random.default_rng(0).choice(X_train.shape[0], self.max_train_samples, replace=False)
            X_train = X_train[idx]

        self.model = self._build_model(X_train.shape[1]).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        criterion = torch.nn.MSELoss()

        dataset = TensorDataset(torch.from_numpy(X_train))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        for _ in range(self.epochs):
            for (xb,) in loader:
                xb = xb.to(self.device)
                optimizer.zero_grad()
                x_hat = self.model(xb)
                loss = criterion(x_hat, xb)
                loss.backward()
                optimizer.step()
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.model is None or self._torch is None:
            raise RuntimeError("AE model is not fitted")
        torch = self._torch
        self.model.eval()
        with torch.no_grad():
            x = torch.from_numpy(X.astype(np.float32)).to(self.device)
            z = self.model.encoder(x)
        return z.cpu().numpy()

    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        """Return reconstructed samples for visualization."""
        if self.model is None or self._torch is None:
            raise RuntimeError("AE model is not fitted")
        torch = self._torch
        self.model.eval()
        with torch.no_grad():
            x = torch.from_numpy(X.astype(np.float32)).to(self.device)
            x_hat = self.model(x)
        return x_hat.cpu().numpy()
