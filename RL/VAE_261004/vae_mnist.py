"""Minimal, readable PyTorch VAE on MNIST.

Formula map
-----------
Encoder:
    q_phi(z|x) = N(mu_phi(x), diag(sigma_phi(x)^2))

Reparameterization:
    eps ~ N(0, I)
    z = mu + sigma * eps

Decoder:
    p_theta(x|z) = Bernoulli(sigmoid(logits_theta(z)))

Training objective:
    loss = reconstruction_nll + KL(q_phi(z|x) || p(z))
         = - ELBO

Prior used for generation:
    p(z) = N(0, I)
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image


class VAE(nn.Module):
    def __init__(self, latent_dim: int = 20) -> None:
        super().__init__()
        self.latent_dim = latent_dim

        # q_phi(z|x): x -> hidden -> (mu, logvar)
        self.enc_fc1 = nn.Linear(28 * 28, 400)
        self.enc_mu = nn.Linear(400, latent_dim)
        self.enc_logvar = nn.Linear(400, latent_dim)

        # p_theta(x|z): z -> hidden -> Bernoulli logits for 784 pixels
        self.dec_fc1 = nn.Linear(latent_dim, 400)
        self.dec_logits = nn.Linear(400, 28 * 28)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = F.relu(self.enc_fc1(x))
        mu = self.enc_mu(h)
        logvar = self.enc_logvar(h)
        return mu, logvar

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """z = mu + sigma * eps, eps ~ N(0, I).

        Randomness is isolated in eps, while z remains differentiable with
        respect to mu and logvar.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.dec_fc1(z))
        return self.dec_logits(h)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x_flat = x.view(x.size(0), -1)
        mu, logvar = self.encode(x_flat)
        z = self.reparameterize(mu, logvar)
        logits = self.decode(z)
        return logits, mu, logvar, z


def vae_loss(
    logits: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (negative ELBO, reconstruction NLL, prior KL).

    For binary MNIST pixels:
        -log p_theta(x|z) = BCE(x, decoder_logits)

    For q=N(mu, diag(sigma^2)) and p=N(0,I):
        KL(q||p) = -1/2 sum(1 + logvar - mu^2 - exp(logvar))
    """
    x_flat = x.view(x.size(0), -1)

    # Monte Carlo estimate of -E_q[log p_theta(x|z)] using one z sample.
    recon_nll = F.binary_cross_entropy_with_logits(
        logits,
        x_flat,
        reduction="sum",
    )

    # Closed-form KL(q_phi(z|x) || p(z)), p(z)=N(0,I).
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # Minimizing this is equivalent to maximizing the ELBO.
    loss = recon_nll + kl
    return loss, recon_nll, kl


def train_epoch(
    model: VAE,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> None:
    model.train()
    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0

    for x, _ in loader:
        x = x.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits, mu, logvar, _ = model(x)
        loss, recon_nll, kl = vae_loss(logits, x, mu, logvar)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_recon += recon_nll.item()
        total_kl += kl.item()

    n = len(loader.dataset)
    print(
        f"epoch={epoch:03d} "
        f"loss/sample={total_loss / n:.4f} "
        f"recon/sample={total_recon / n:.4f} "
        f"kl/sample={total_kl / n:.4f}"
    )


@torch.no_grad()
def evaluate(model: VAE, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0

    for x, _ in loader:
        x = x.to(device)
        logits, mu, logvar, _ = model(x)
        loss, _, _ = vae_loss(logits, x, mu, logvar)
        total_loss += loss.item()

    return total_loss / len(loader.dataset)


@torch.no_grad()
def sample_from_prior(
    model: VAE,
    device: torch.device,
    out_path: Path,
    n: int = 64,
) -> None:
    """Generation path: z ~ p(z)=N(0,I) -> decoder, no encoder needed."""
    model.eval()
    z = torch.randn(n, model.latent_dim, device=device)
    logits = model.decode(z)
    probs = torch.sigmoid(logits).view(n, 1, 28, 28)
    save_image(probs, out_path, nrow=8)


@torch.no_grad()
def save_reconstructions(
    model: VAE,
    loader: DataLoader,
    device: torch.device,
    out_path: Path,
    n: int = 8,
) -> None:
    model.eval()
    x, _ = next(iter(loader))
    x = x[:n].to(device)
    logits, _, _, _ = model(x)
    recon = torch.sigmoid(logits).view(n, 1, 28, 28)

    # First row originals, second row reconstructions.
    grid = torch.cat([x, recon], dim=0)
    save_image(grid, out_path, nrow=n)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a simple MNIST VAE")
    p.add_argument("--data-dir", type=Path, default=Path("./data"))
    p.add_argument("--out-dir", type=Path, default=Path("./vae_outputs"))
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--latent-dim", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-cuda", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    use_cuda = torch.cuda.is_available() and not args.no_cuda
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"device={device}")

    transform = transforms.ToTensor()
    train_set = datasets.MNIST(
        root=args.data_dir,
        train=True,
        download=True,
        transform=transform,
    )
    test_set = datasets.MNIST(
        root=args.data_dir,
        train=False,
        download=True,
        transform=transform,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=use_cuda,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=use_cuda,
    )

    model = VAE(latent_dim=args.latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        train_epoch(model, train_loader, optimizer, device, epoch)
        test_loss = evaluate(model, test_loader, device)
        print(f"           test negative-ELBO/sample={test_loss:.4f}")

    torch.save(model.state_dict(), args.out_dir / "vae_mnist.pt")
    sample_from_prior(model, device, args.out_dir / "samples.png")
    save_reconstructions(model, test_loader, device, args.out_dir / "reconstructions.png")

    print(f"saved outputs to: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
