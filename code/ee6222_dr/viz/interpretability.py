"""Interpretability figure generation."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import NMF, PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from ee6222_dr.dr.ae import AutoEncoderDR


def _plot_image_grid(
    images: np.ndarray,
    image_shape: tuple[int, int],
    out_path: Path,
    title: str,
    n_cols: int = 4,
) -> None:
    n_images = images.shape[0]
    n_rows = int(np.ceil(n_images / n_cols))

    plt.figure(figsize=(2.2 * n_cols, 2.2 * n_rows))
    for i in range(n_images):
        ax = plt.subplot(n_rows, n_cols, i + 1)
        ax.imshow(images[i].reshape(image_shape), cmap="gray")
        ax.axis("off")
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def save_pca_eigenimages(
    X_train: np.ndarray,
    image_shape: tuple[int, int],
    out_path: Path,
    n_components: int = 16,
) -> None:
    """Save top PCA components as eigenimages."""
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_train)

    n = min(n_components, Xs.shape[1], Xs.shape[0] - 1)
    if n <= 0:
        return

    model = PCA(n_components=n)
    model.fit(Xs)
    comps = model.components_
    _plot_image_grid(comps, image_shape, out_path, title="PCA Eigenimages")


def save_nmf_components(
    X_train: np.ndarray,
    image_shape: tuple[int, int],
    out_path: Path,
    n_components: int = 16,
) -> None:
    """Save NMF components as parts-based basis images."""
    scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    Xm = scaler.fit_transform(X_train)

    n = min(n_components, Xm.shape[1], Xm.shape[0])
    if n <= 0:
        return

    model = NMF(n_components=n, init="nndsvd", max_iter=300, random_state=0)
    model.fit(Xm)
    comps = model.components_
    _plot_image_grid(comps, image_shape, out_path, title="NMF Components")


def save_ae_reconstruction(
    X_train: np.ndarray,
    image_shape: tuple[int, int],
    out_path: Path,
    device: str,
    ae_d: int = 16,
    epochs: int = 8,
) -> None:
    """Train a small AE and save input/reconstruction pairs."""
    scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    Xm = scaler.fit_transform(X_train)

    d = min(ae_d, Xm.shape[1])
    if d <= 0:
        return

    model = AutoEncoderDR(
        n_components=d,
        hidden_dim=256,
        epochs=epochs,
        batch_size=256,
        lr=1e-3,
        weight_decay=1e-5,
        device=device,
        max_train_samples=min(4000, Xm.shape[0]),
    )
    model.fit(Xm)

    n_vis = min(8, Xm.shape[0])
    samples = Xm[:n_vis]
    recon = model.reconstruct(samples)

    plt.figure(figsize=(2.2 * n_vis, 4.5))
    for i in range(n_vis):
        ax1 = plt.subplot(2, n_vis, i + 1)
        ax1.imshow(samples[i].reshape(image_shape), cmap="gray")
        ax1.axis("off")
        if i == 0:
            ax1.set_title("Input")

        ax2 = plt.subplot(2, n_vis, n_vis + i + 1)
        ax2.imshow(recon[i].reshape(image_shape), cmap="gray")
        ax2.axis("off")
        if i == 0:
            ax2.set_title(f"AE Recon (d={d})")

    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def generate_interpretability_figures(
    dataset_name: str,
    X_train: np.ndarray,
    image_shape: tuple[int, int],
    figures_dir: str | Path,
    device: str,
    ae_d: int = 16,
    ae_epochs: int = 8,
) -> list[Path]:
    """Generate PCA/NMF/AE interpretability figures for one dataset."""
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    out_paths = []

    pca_path = figures_dir / f"pca_eigenimages_{dataset_name}.png"
    save_pca_eigenimages(X_train=X_train, image_shape=image_shape, out_path=pca_path)
    out_paths.append(pca_path)

    nmf_path = figures_dir / f"nmf_components_{dataset_name}.png"
    save_nmf_components(X_train=X_train, image_shape=image_shape, out_path=nmf_path)
    out_paths.append(nmf_path)

    ae_path = figures_dir / f"ae_reconstruction_{dataset_name}_d{ae_d}.png"
    save_ae_reconstruction(
        X_train=X_train,
        image_shape=image_shape,
        out_path=ae_path,
        device=device,
        ae_d=ae_d,
        epochs=ae_epochs,
    )
    out_paths.append(ae_path)

    return out_paths
