"""Variational autoencoder dimensionality reduction wrapper."""

from __future__ import annotations

import numpy as np

from . import DRMethod


class VAEDR(DRMethod):
    """Lightweight MLP VAE used as probabilistic latent DR."""

    def __init__(
        self,
        n_components: int,
        hidden_dim: int = 256,
        epochs: int = 10,
        batch_size: int = 256,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        beta: float = 1.0,
        device: str = "cpu",
        max_train_samples: int | None = None,
    ) -> None:
        self.n_components = n_components
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.beta = beta
        self.device = device
        self.max_train_samples = max_train_samples

        self.model = None
        self._torch = None

    def _build_model(self, input_dim: int):
        import torch
        import torch.nn as nn

        class _VAE(nn.Module):
            def __init__(self, in_dim: int, hidden: int, latent: int) -> None:
                super().__init__()
                self.fc1 = nn.Linear(in_dim, hidden)
                self.fc_mu = nn.Linear(hidden, latent)
                self.fc_logvar = nn.Linear(hidden, latent)
                self.fc2 = nn.Linear(latent, hidden)
                self.fc3 = nn.Linear(hidden, in_dim)

            def encode(self, x):
                h = torch.relu(self.fc1(x))
                return self.fc_mu(h), self.fc_logvar(h)

            def reparameterize(self, mu, logvar):
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std)
                return mu + eps * std

            def decode(self, z):
                h = torch.relu(self.fc2(z))
                return torch.sigmoid(self.fc3(h))

            def forward(self, x):
                mu, logvar = self.encode(x)
                z = self.reparameterize(mu, logvar)
                x_hat = self.decode(z)
                return x_hat, mu, logvar

        return _VAE(input_dim, self.hidden_dim, self.n_components)

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "VAEDR":
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        self._torch = torch

        X_train = X.astype(np.float32)
        if self.max_train_samples is not None and self.max_train_samples < X_train.shape[0]:
            idx = np.random.default_rng(0).choice(X_train.shape[0], self.max_train_samples, replace=False)
            X_train = X_train[idx]

        self.model = self._build_model(X_train.shape[1]).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        dataset = TensorDataset(torch.from_numpy(X_train))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        for _ in range(self.epochs):
            for (xb,) in loader:
                xb = xb.to(self.device)
                optimizer.zero_grad()
                x_hat, mu, logvar = self.model(xb)

                recon = torch.nn.functional.mse_loss(x_hat, xb, reduction="mean")
                kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                loss = recon + self.beta * kl

                loss.backward()
                optimizer.step()
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.model is None or self._torch is None:
            raise RuntimeError("VAE model is not fitted")
        torch = self._torch
        self.model.eval()
        with torch.no_grad():
            x = torch.from_numpy(X.astype(np.float32)).to(self.device)
            mu, _ = self.model.encode(x)
        return mu.cpu().numpy()

    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        """Return reconstructions from latent mean for visualization."""
        if self.model is None or self._torch is None:
            raise RuntimeError("VAE model is not fitted")
        torch = self._torch
        self.model.eval()
        with torch.no_grad():
            x = torch.from_numpy(X.astype(np.float32)).to(self.device)
            mu, _ = self.model.encode(x)
            x_hat = self.model.decode(mu)
        return x_hat.cpu().numpy()
