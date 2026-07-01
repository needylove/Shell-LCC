"""Shell-LCC manifold model (Shell Local Coordinate Coding).

Self-contained, no internal dependencies. Used both to train the manifold
(two stages, see train_manifold_2stage.py) and to score generated latents as a
reward (ShellCoordinateManifold.distance_reward, see scripts/train_manifold.py).

Architecture
  LCC skeleton : basis (codebook) + global_scale/global_bias + predictor (amortized encoder)
  Shell head   : surface_head, predicts a per-dimension log-variance (shell thickness)
                 given the skeleton point.

forward(z, stage=...)
  stage='lcc'   -- Stage 1: train the LCC skeleton only.
                   loss = l1(reconstruction) + l2(local constraint)
                          + [lambda_usage * KL(usage || uniform)]
                          + [lambda_basis_to_patch * basis_to_patch]
                   The two regularizers are optional (set lambda_*=0 to disable).
                   surface_head is never touched.
  stage='joint' -- Stage 2 (call after freezing the LCC params): adds the shell loss
                   on top of the LCC loss, so only surface_head receives gradients.

distance_reward(z) -- Inference reward: mean normalized distance of latent patches to the
                      learned shell (smaller = closer to the data manifold). Kept
                      differentiable so the reward can back-propagate into the generator.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ShellCoordinateManifold(nn.Module):
    def __init__(self, embedding_dim=16, num_bases=4096, hidden_dim=256):
        super().__init__()
        self.num_bases = num_bases
        self.embedding_dim = embedding_dim

        # --- LCC skeleton ---
        self.basis = nn.Parameter(torch.randn(num_bases, embedding_dim))  # codebook
        nn.init.xavier_uniform_(self.basis)
        self.global_scale = nn.Parameter(torch.ones(1))                   # global scale
        self.global_bias = nn.Parameter(torch.zeros(1, embedding_dim))    # global shift
        self.predictor = nn.Sequential(                                   # latent point -> basis weights
            nn.Linear(embedding_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_bases),
        )
        # --- Shell head: predicts local log-variance from the skeleton point ---
        self.surface_head = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def cal_local_loss(self, recovered, latent, basis, w):
        """LCC loss: l1 (reconstruction MSE) + l2 (weighted local constraint).
        Uses the expansion ||z-c||^2 = ||z||^2 + ||c||^2 - 2<z,c> to avoid an (N,K,C) tensor."""
        l1 = F.mse_loss(recovered, latent)
        term_z = torch.sum(latent ** 2, dim=1, keepdim=True)    # (N,1)
        term_c = torch.sum(basis ** 2, dim=1).unsqueeze(0)      # (1,K)
        term_cross = torch.matmul(latent, basis.t())            # (N,K)
        dist_sq = torch.relu(term_z + term_c - 2 * term_cross)  # (N,K) = ||z-c||^2
        l2 = torch.mean(torch.sum(w * dist_sq, dim=1))          # mean_n sum_k w_nk ||z_n - c_k||^2
        return l1 + l2, {"l1": l1.item(), "l2": l2.item()}

    def basis_to_patch_loss(self, z, basis, topk=1):
        """Attract each basis toward its nearest data patch(es) so the codebook covers the data."""
        z2 = (z ** 2).sum(dim=1, keepdim=True)                  # (N,1)
        b2 = (basis ** 2).sum(dim=1).unsqueeze(0)              # (1,K)
        dist = z2 + b2 - 2 * z @ basis.t()                     # (N,K)
        _, didx = torch.topk(dist.t(), topk, dim=1, largest=False)  # nearest patches per basis
        z_anchor = z[didx].mean(dim=1)                         # (K,D)
        return ((basis - z_anchor) ** 2).sum(dim=-1).mean()

    def forward(self, z, stage='joint', lambda_usage=0.1, lambda_basis_to_patch=0.1):
        # z: (B, C, T, H, W) 3D VAE latent
        B, C, T, H, W = z.shape
        z_flat = z.permute(0, 2, 3, 4, 1).reshape(-1, C)                        # (N,C)
        w = F.softmax(self.predictor(z_flat), dim=1)                            # (N,K) coding weights
        z_skel = torch.matmul(w, self.basis) * self.global_scale + self.global_bias  # (N,C) skeleton
        loss, loss_dict = self.cal_local_loss(z_skel, z_flat, self.basis, w)

        # ---- Stage 1: LCC only ----
        if stage == 'lcc':
            if lambda_usage > 0:
                # Push the average codebook usage toward uniform (prevents collapse to a few bases).
                usage = torch.mean(w, dim=0) + 1e-8
                usage = usage / usage.sum()
                uniform = torch.full_like(usage, 1.0 / usage.numel())
                l_usage = torch.sum(usage * (torch.log(usage) - torch.log(uniform))) * lambda_usage
                loss = loss + l_usage
                loss_dict['kl'] = l_usage.item()
            if lambda_basis_to_patch > 0:
                l_b2p = self.basis_to_patch_loss(z_flat, self.basis)
                loss = loss + lambda_basis_to_patch * l_b2p
                loss_dict['l_basis_to_patch'] = l_b2p.item()
            z_hat = z_skel.view(B, T, H, W, C).permute(0, 4, 1, 2, 3)
            return z_hat, loss, loss_dict, w

        # ---- Stage 2: joint (call after freezing LCC) -> only surface_head updates ----
        log_var = self.surface_head(z_skel)
        var = torch.exp(log_var)
        delta = (z_flat - z_skel) / (var + 1e-6)
        dist = torch.norm(delta, p=2, dim=1, keepdim=True)
        loss_shell = torch.mean((dist - 1.0) ** 2)             # normalized residual should sit on the unit shell
        loss_reg = 0.01 * torch.mean(log_var ** 2)             # keep log_var from drifting
        loss = loss + loss_shell + loss_reg
        loss_dict['l_NLL'] = loss_shell.item()
        z_hat = z_skel.view(B, T, H, W, C).permute(0, 4, 1, 2, 3)
        return z_hat, loss, loss_dict, w

    def distance_reward(self, z):
        """Reward at inference: mean normalized distance of latent patches to the shell
        (smaller = closer to the data manifold). Not wrapped in no_grad -- the reward must
        back-propagate into the generator during reward training."""
        B, C, T, H, W = z.shape
        z_flat = z.permute(0, 2, 3, 4, 1).reshape(-1, C)
        w = F.softmax(self.predictor(z_flat), dim=1)
        z_skel = torch.matmul(w, self.basis) * self.global_scale + self.global_bias
        var = torch.exp(self.surface_head(z_skel))
        dist = torch.norm((z_flat - z_skel) / (var + 1e-6), p=2, dim=1, keepdim=True)
        return torch.mean(dist)


@torch.no_grad()
def init_basis_with_random_samples(model, dataloader, device):
    """Initialize the codebook (basis) with real data points -- more stable and faster than random init."""
    model.eval()
    target = model.basis.shape[0]
    pts = []
    for batch in dataloader:
        flat = batch.permute(0, 2, 3, 4, 1).contiguous().view(-1, model.embedding_dim)
        pts.append(flat)
        if sum(x.shape[0] for x in pts) >= target:
            break
    pts = torch.cat(pts, dim=0)
    sel = pts[torch.randperm(pts.size(0))[:target]]
    model.basis.data = sel.to(device).clone()
    print(f"[init] basis <- {target} real points | mean {model.basis.mean():.4f} max {model.basis.max():.4f}")
