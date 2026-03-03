#!/usr/bin/env python3
"""
Step 2: NCA-SGD Training Experiment on CIFAR-100 + ViT-Tiny (v2 - fixed)

Fixes from v1:
  1. Random-SGD dispatch bug (was never actually boosting)
  2. Coverage: ALL target layers updated per NCA step (not round-robin)
  3. Multiple subsets per layer for higher coverage
  4. Boost factor increased to compensate for sub-Hessian coverage
  5. Unified engine interface with .update() and .apply()

Three-agent comparison:
  1. SGD (baseline)
  2. NCA-SGD (null cone assisted)
  3. Random-SGD (random direction boosting - ablation control)

Usage (Colab):
    Config.epochs = 20
    main()
"""

import time
import json
from datetime import datetime
from pathlib import Path
import numpy as np


# ============================================================
# 1. Null Cone Core
# ============================================================

def make_symplectic_J(n):
    I_n = np.eye(n)
    return np.block([[np.zeros((n, n)), I_n], [-I_n, np.zeros((n, n))]])


def compute_sub_hessian(model, loss_fn, inputs, labels, param_indices,
                        layer_params, device):
    import torch
    sub_dim = len(param_indices)
    H = np.zeros((sub_dim, sub_dim), dtype=np.float64)
    model.zero_grad()
    outputs = model(inputs).logits
    loss = loss_fn(outputs, labels)
    grad = torch.autograd.grad(loss, layer_params, create_graph=True)[0]
    grad_flat = grad.reshape(-1)
    sub_grad = grad_flat[param_indices]
    for i in range(sub_dim):
        grad2 = torch.autograd.grad(sub_grad[i], layer_params, retain_graph=True)[0]
        H[i, :] = grad2.reshape(-1)[param_indices].detach().cpu().numpy().astype(np.float64)
    return H


def find_null_directions(H, sub_dim, lam=1e-6):
    """Returns list of null direction vectors (normalized), or empty list."""
    H_sym = 0.5 * (H + H.T)
    H_reg = H_sym + lam * np.eye(sub_dim)
    evals = np.linalg.eigvalsh(H_sym)
    if not (np.any(evals > 0) and np.any(evals < 0)):
        return []
    try:
        H_reg_inv = np.linalg.inv(H_reg)
    except np.linalg.LinAlgError:
        return []
    J = make_symplectic_J(sub_dim // 2)
    M = H_reg_inv @ J
    evals_M, evecs_M = np.linalg.eig(M)
    real_mask = np.abs(evals_M.imag) < 1e-8
    real_indices = np.where(real_mask)[0]
    if len(real_indices) == 0:
        return []
    H_reg_norm = np.linalg.norm(H_reg)
    null_vecs = []
    for idx in real_indices:
        e = evecs_M[:, idx].real
        lam_e = evals_M[idx].real
        if np.linalg.norm(M @ e - lam_e * e) / (np.linalg.norm(e) + 1e-30) > 1e-6:
            continue
        e_norm = e / (np.linalg.norm(e) + 1e-30)
        if abs(e_norm @ H_reg @ e_norm) / (H_reg_norm + 1e-30) < 1e-6:
            null_vecs.append(e_norm)
    return null_vecs


# ============================================================
# 2. Per-Layer Boost State
# ============================================================

class LayerBoostState:
    """Stores boost directions and indices for one layer."""
    __slots__ = ['param_indices_list', 'Q_list', 'num_params']

    def __init__(self, num_params):
        self.num_params = num_params
        self.param_indices_list = []  # list of index tensors
        self.Q_list = []              # list of Q matrices (one per subset)

    def clear(self):
        self.param_indices_list.clear()
        self.Q_list.clear()

    @property
    def coverage(self):
        """Fraction of parameters covered by at least one subset."""
        if self.num_params == 0:
            return 0.0
        covered = set()
        for idx in self.param_indices_list:
            covered.update(idx.cpu().numpy().tolist())
        return len(covered) / self.num_params


# ============================================================
# 3. Unified Boost Engines
# ============================================================

class NCAEngine:
    """
    All-layer NCA: computes null directions for ALL target layers per update.
    Multiple random subsets per layer for higher coverage.
    """
    def __init__(self, model, target_layers, sub_dim=50, n_subsets=3,
                 boost=5.0, lam=1e-6, device='cuda'):
        import torch
        self.model = model
        self.target_layers = target_layers
        self.sub_dim = sub_dim
        self.n_subsets = n_subsets
        self.boost = boost
        self.lam = lam
        self.device = device

        self.layer_states = {
            name: LayerBoostState(p.numel())
            for name, p in target_layers.items()
        }
        self.stats = {
            "total_updates": 0, "layers_with_null": 0,
            "total_null_dirs": 0, "residuals": [],
            "coverage": [],
        }

    def update(self, loss_fn, inputs, labels):
        """Compute null directions for all layers. Call every k steps."""
        import torch
        self.stats["total_updates"] += 1

        for name, param in self.target_layers.items():
            state = self.layer_states[name]
            state.clear()

            num_params = param.numel()
            sub_dim = min(self.sub_dim, num_params)
            if sub_dim % 2 != 0:
                sub_dim -= 1
            if sub_dim < 4:
                continue

            param.requires_grad_(True)
            layer_null_count = 0

            for s in range(self.n_subsets):
                rng = np.random.RandomState(
                    int(time.time() * 1000 + s * 7919) % (2**31))
                indices = torch.tensor(
                    sorted(rng.choice(num_params, size=sub_dim, replace=False)),
                    dtype=torch.long, device=self.device)

                try:
                    H = compute_sub_hessian(
                        self.model, loss_fn, inputs, labels,
                        indices, param, self.device)
                    null_vecs = find_null_directions(H, sub_dim, self.lam)
                except Exception:
                    null_vecs = []

                if null_vecs:
                    V = np.column_stack(null_vecs)
                    Q, _ = np.linalg.qr(V)
                    state.param_indices_list.append(indices)
                    state.Q_list.append(Q)
                    layer_null_count += len(null_vecs)

            param.requires_grad_(False)

            if layer_null_count > 0:
                self.stats["layers_with_null"] += 1
                self.stats["total_null_dirs"] += layer_null_count
            self.stats["coverage"].append(state.coverage)

    def apply(self):
        """Apply null-direction boosting to gradients of all layers."""
        import torch
        total_frac = 0.0
        n_applied = 0

        for name, param in self.target_layers.items():
            if param.grad is None:
                continue
            state = self.layer_states[name]
            if not state.Q_list:
                continue

            grad_flat = param.grad.reshape(-1)

            for indices, Q in zip(state.param_indices_list, state.Q_list):
                sub_grad = grad_flat[indices].detach().cpu().numpy().astype(np.float64)
                g_null = Q @ (Q.T @ sub_grad)
                g_rem = sub_grad - g_null
                frac = np.linalg.norm(g_null) / (np.linalg.norm(sub_grad) + 1e-30)
                total_frac += frac
                n_applied += 1
                boosted = self.boost * g_null + g_rem
                with torch.no_grad():
                    grad_flat[indices] = torch.tensor(
                        boosted, dtype=grad_flat.dtype, device=self.device)

        return total_frac / max(n_applied, 1)


class RandomEngine:
    """
    Random direction boosting. Same structure/coverage as NCA
    but uses random orthogonal directions. Ablation control.
    """
    def __init__(self, model, target_layers, sub_dim=50, n_subsets=3,
                 boost=5.0, device='cuda'):
        import torch
        self.model = model
        self.target_layers = target_layers
        self.sub_dim = sub_dim
        self.n_subsets = n_subsets
        self.boost = boost
        self.device = device
        self.layer_states = {
            name: LayerBoostState(p.numel())
            for name, p in target_layers.items()
        }
        self.stats = {"total_updates": 0}

    def update(self, loss_fn=None, inputs=None, labels=None):
        """Generate random directions for all layers."""
        import torch
        self.stats["total_updates"] += 1

        for name, param in self.target_layers.items():
            state = self.layer_states[name]
            state.clear()
            num_params = param.numel()
            sub_dim = min(self.sub_dim, num_params)
            if sub_dim % 2 != 0:
                sub_dim -= 1
            if sub_dim < 4:
                continue

            for s in range(self.n_subsets):
                rng = np.random.RandomState(
                    int(time.time() * 1000 + s * 7919) % (2**31))
                indices = torch.tensor(
                    sorted(rng.choice(num_params, size=sub_dim, replace=False)),
                    dtype=torch.long, device=self.device)
                # Random orthogonal directions (match typical null count ~8)
                n_dirs = min(8, sub_dim)
                A = rng.randn(sub_dim, n_dirs)
                Q, _ = np.linalg.qr(A)
                state.param_indices_list.append(indices)
                state.Q_list.append(Q)

    def apply(self):
        """Apply random-direction boosting to gradients."""
        import torch
        total_frac = 0.0
        n_applied = 0

        for name, param in self.target_layers.items():
            if param.grad is None:
                continue
            state = self.layer_states[name]
            if not state.Q_list:
                continue
            grad_flat = param.grad.reshape(-1)

            for indices, Q in zip(state.param_indices_list, state.Q_list):
                sub_grad = grad_flat[indices].detach().cpu().numpy().astype(np.float64)
                g_proj = Q @ (Q.T @ sub_grad)
                g_rem = sub_grad - g_proj
                frac = np.linalg.norm(g_proj) / (np.linalg.norm(sub_grad) + 1e-30)
                total_frac += frac
                n_applied += 1
                boosted = self.boost * g_proj + g_rem
                with torch.no_grad():
                    grad_flat[indices] = torch.tensor(
                        boosted, dtype=grad_flat.dtype, device=self.device)

        return total_frac / max(n_applied, 1)


# ============================================================
# 4. Training
# ============================================================

def get_vit_target_layers(model):
    """All attention + MLP layers from ViT-Tiny (12 layers)."""
    targets = {}
    for li in range(12):
        block = model.vit.encoder.layer[li]
        targets[f"attn_L{li}_q"] = block.attention.attention.query.weight
        targets[f"attn_L{li}_o"] = block.attention.output.dense.weight
        targets[f"mlp_L{li}_f1"] = block.intermediate.dense.weight
        targets[f"mlp_L{li}_f2"] = block.output.dense.weight
    return targets


def train_one_agent(agent_name, model, train_loader, val_loader, optimizer,
                    loss_fn, device, epochs, engine=None, update_every=50):
    import torch
    history = {"train_loss": [], "val_loss": [], "val_acc": [],
               "boost_fracs": [], "updates": 0}
    step = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        n_batches = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Periodic direction update
            if engine is not None and step > 0 and step % update_every == 0:
                model.eval()
                with torch.enable_grad():
                    engine.update(loss_fn, images, labels)
                model.train()
                history["updates"] += 1

            # Forward / backward
            optimizer.zero_grad()
            outputs = model(images).logits
            loss = loss_fn(outputs, labels)
            loss.backward()

            # Apply boost
            if engine is not None:
                frac = engine.apply()
                if frac > 0:
                    history["boost_fracs"].append(frac)

            optimizer.step()
            running_loss += loss.item()
            n_batches += 1
            step += 1

        avg_train = running_loss / n_batches
        history["train_loss"].append(avg_train)

        # Validation
        model.eval()
        vloss = correct = total = 0
        with torch.no_grad():
            for imgs, labs in val_loader:
                imgs, labs = imgs.to(device), labs.to(device)
                out = model(imgs).logits
                vloss += loss_fn(out, labs).item()
                correct += out.argmax(1).eq(labs).sum().item()
                total += labs.size(0)

        avg_val = vloss / len(val_loader)
        acc = 100.0 * correct / total
        history["val_loss"].append(avg_val)
        history["val_acc"].append(acc)

        extra = ""
        if engine is not None:
            extra = f" | updates={history['updates']}"
            if hasattr(engine, 'stats'):
                s = engine.stats
                if 'layers_with_null' in s:
                    extra += f" null_layers={s['layers_with_null']}"

        print(f"  [{agent_name}] Ep {epoch+1}/{epochs}: "
              f"train={avg_train:.4f} val={avg_val:.4f} acc={acc:.1f}%{extra}")

    return history


# ============================================================
# 5. Main
# ============================================================

def run_experiment(args):
    import torch
    import torch.nn as nn
    import torchvision
    import torchvision.transforms as transforms
    from transformers import ViTForImageClassification

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")
    print(f"Epochs: {args.epochs}, LR: {args.lr}")
    print(f"NCA: every {args.update_every} steps, {args.n_subsets} subsets/layer, "
          f"boost={args.boost}, sub_dim={args.sub_dim}")
    print(f"Seeds: {args.seeds}, Subset: {args.subset_size or 'full'}")
    print("=" * 70)

    # Disable SDPA fused kernels
    if hasattr(torch.backends, 'cuda'):
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)

    # Data
    print("\nLoading CIFAR-100...")
    mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
    tf_train = transforms.Compose([
        transforms.Resize(224), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), transforms.Normalize(mean, std)])
    tf_val = transforms.Compose([
        transforms.Resize(224), transforms.ToTensor(),
        transforms.Normalize(mean, std)])

    ds_train = torchvision.datasets.CIFAR100('./data', True, download=True, transform=tf_train)
    ds_val = torchvision.datasets.CIFAR100('./data', False, download=True, transform=tf_val)

    if args.subset_size > 0:
        ds_train = torch.utils.data.Subset(ds_train, list(range(min(args.subset_size, len(ds_train)))))
        ds_val = torch.utils.data.Subset(ds_val, list(range(min(args.subset_size // 5, len(ds_val)))))
        print(f"Subset: {len(ds_train)} train, {len(ds_val)} val")

    dl_train = torch.utils.data.DataLoader(ds_train, args.batch_size, shuffle=True,
                                            num_workers=2, pin_memory=True)
    dl_val = torch.utils.data.DataLoader(ds_val, args.batch_size, shuffle=False,
                                          num_workers=2, pin_memory=True)
    loss_fn = nn.CrossEntropyLoss()

    all_results = {}

    for seed in range(args.seeds):
        print(f"\n{'='*70}\nSeed {seed+1}/{args.seeds}\n{'='*70}")

        def make_model(s):
            torch.manual_seed(s)
            m = ViTForImageClassification.from_pretrained(
                'WinKawaks/vit-tiny-patch16-224',
                num_labels=100, ignore_mismatched_sizes=True,
                attn_implementation="eager")
            return m.to(device)

        def make_opt(m):
            return torch.optim.SGD(m.parameters(), lr=args.lr,
                                    momentum=0.9, weight_decay=1e-4)

        # --- SGD ---
        print("\n--- SGD ---")
        m1 = make_model(seed)
        h1 = train_one_agent("SGD", m1, dl_train, dl_val, make_opt(m1),
                              loss_fn, device, args.epochs)

        # --- NCA-SGD ---
        print("\n--- NCA-SGD ---")
        m2 = make_model(seed)
        tl2 = get_vit_target_layers(m2)
        nca = NCAEngine(m2, tl2, sub_dim=args.sub_dim,
                         n_subsets=args.n_subsets, boost=args.boost, device=device)
        h2 = train_one_agent("NCA-SGD", m2, dl_train, dl_val, make_opt(m2),
                              loss_fn, device, args.epochs,
                              engine=nca, update_every=args.update_every)
        h2["nca_stats"] = nca.stats

        # --- Random-SGD ---
        print("\n--- Random-SGD ---")
        m3 = make_model(seed)
        tl3 = get_vit_target_layers(m3)
        rnd = RandomEngine(m3, tl3, sub_dim=args.sub_dim,
                            n_subsets=args.n_subsets, boost=args.boost, device=device)
        h3 = train_one_agent("Rnd-SGD", m3, dl_train, dl_val, make_opt(m3),
                              loss_fn, device, args.epochs,
                              engine=rnd, update_every=args.update_every)
        h3["rnd_stats"] = rnd.stats

        all_results[f"seed_{seed}"] = {"sgd": h1, "nca_sgd": h2, "random_sgd": h3}

        print(f"\n  Seed {seed+1}:")
        print(f"    SGD:     {h1['val_acc'][-1]:.1f}%  loss={h1['val_loss'][-1]:.4f}")
        print(f"    NCA-SGD: {h2['val_acc'][-1]:.1f}%  loss={h2['val_loss'][-1]:.4f}  "
              f"(updates={h2['updates']}, null_layers={nca.stats['layers_with_null']})")
        print(f"    Rnd-SGD: {h3['val_acc'][-1]:.1f}%  loss={h3['val_loss'][-1]:.4f}  "
              f"(updates={h3['updates']})")

    # Summary
    print(f"\n{'='*70}\nFINAL SUMMARY\n{'='*70}")
    for key, label in [("sgd","SGD"), ("nca_sgd","NCA-SGD"), ("random_sgd","Rnd-SGD")]:
        accs = [all_results[f"seed_{s}"][key]["val_acc"][-1] for s in range(args.seeds)]
        print(f"  {label:10s}: {np.mean(accs):.1f}% +/- {np.std(accs):.1f}%")

    nca_beat_sgd = sum(
        1 for s in range(args.seeds)
        if all_results[f"seed_{s}"]["nca_sgd"]["val_acc"][-1] >
           all_results[f"seed_{s}"]["sgd"]["val_acc"][-1])
    nca_beat_rnd = sum(
        1 for s in range(args.seeds)
        if all_results[f"seed_{s}"]["nca_sgd"]["val_acc"][-1] >
           all_results[f"seed_{s}"]["random_sgd"]["val_acc"][-1])
    print(f"\n  NCA > SGD:     {nca_beat_sgd}/{args.seeds}")
    print(f"  NCA > Random:  {nca_beat_rnd}/{args.seeds}")

    out_path = Path(args.output)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")


class Config:
    device = "cuda"
    epochs = 20
    batch_size = 64
    lr = 0.01
    sub_dim = 50
    n_subsets = 3         # Subsets per layer per update (3 x 50 = 150 params/layer)
    update_every = 50     # Steps between direction updates
    boost = 5.0           # Boost factor (increased from 2.0)
    seeds = 3
    subset_size = 5000    # 0 = full; >0 for fast test
    output = "nca_training_v2.json"

def main():
    args = Config()
    import sys
    if not any('jupyter' in a or 'ipykernel' in a or 'colab' in a for a in sys.argv):
        try:
            import argparse
            p = argparse.ArgumentParser()
            for k, v in vars(Config()).items():
                p.add_argument(f"--{k}", type=type(v), default=v)
            args = p.parse_args()
        except SystemExit:
            pass
    if args.sub_dim % 2 != 0:
        args.sub_dim -= 1
    run_experiment(args)

if __name__ == "__main__":
    main()
