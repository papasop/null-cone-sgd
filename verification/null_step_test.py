#!/usr/bin/env python3
"""
Null Direction Step Size Test

Core question: How far can you move along a null direction before loss changes?

For each target layer:
  1. Compute null directions from 50x50 sub-Hessian
  2. Walk along null direction with step sizes epsilon = [1e-4, 1e-3, 1e-2, 0.1, 0.5, 1.0, 2.0, 5.0]
  3. Measure: delta_loss = |L(theta + eps*d) - L(theta)| / L(theta)
  4. Compare with random direction of same norm (control)

If loss stable to eps=0.1 → continual learning feasible
If loss stable only to eps=0.01 → local property only
If loss stable to eps=1.0+ → remarkably flat channel

Also measures:
  - Hessian prediction: predicted delta = 0.5 * eps^2 * d^T H d (should be ~0 for null)
  - Actual vs predicted ratio (tests higher-order effects)
  - Loss change along random direction (control)

Usage (Colab):
    main()
"""

import time
import json
from pathlib import Path
import numpy as np


# ============================================================
# 1. Null Cone Core (from verification script)
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
    """Returns (null_vecs, Q_basis, H_reg) or ([], None, H_reg)."""
    H_sym = 0.5 * (H + H.T)
    H_reg = H_sym + lam * np.eye(sub_dim)
    H_reg_norm = np.linalg.norm(H_reg)

    evals = np.linalg.eigvalsh(H_sym)
    if not (np.any(evals > 0) and np.any(evals < 0)):
        return [], None, H_reg

    try:
        H_reg_inv = np.linalg.inv(H_reg)
    except np.linalg.LinAlgError:
        return [], None, H_reg

    J = make_symplectic_J(sub_dim // 2)
    M = H_reg_inv @ J
    evals_M, evecs_M = np.linalg.eig(M)

    real_mask = np.abs(evals_M.imag) < 1e-8
    real_indices = np.where(real_mask)[0]
    if len(real_indices) == 0:
        return [], None, H_reg

    null_vecs = []
    for idx in real_indices:
        e = evecs_M[:, idx].real
        lam_e = evals_M[idx].real
        if np.linalg.norm(M @ e - lam_e * e) / (np.linalg.norm(e) + 1e-30) > 1e-6:
            continue
        e_norm = e / (np.linalg.norm(e) + 1e-30)
        if abs(e_norm @ H_reg @ e_norm) / (H_reg_norm + 1e-30) < 1e-6:
            null_vecs.append(e_norm)

    if not null_vecs:
        return [], None, H_reg

    V = np.column_stack(null_vecs)
    Q, R = np.linalg.qr(V)
    if np.min(np.abs(np.diag(R))) < 1e-12:
        Q = V[:, :1] / np.linalg.norm(V[:, 0])

    return null_vecs, Q, H_reg


# ============================================================
# 2. Step Size Test
# ============================================================

def test_step_sizes(model, loss_fn, inputs, labels, param, param_indices,
                    null_vec, H_reg, device, epsilons):
    """
    Walk along null_vec with various step sizes, measure loss change.

    Args:
        null_vec: normalized null direction in sub-Hessian coordinates
        H_reg: regularized Hessian for curvature prediction

    Returns list of dicts, one per epsilon.
    """
    import torch

    # Baseline loss
    model.eval()
    with torch.no_grad():
        outputs = model(inputs).logits
        base_loss = loss_fn(outputs, labels).item()

    # Curvature of null direction (should be ~0)
    null_curvature = float(null_vec @ H_reg @ null_vec)

    # Generate random direction for control
    rng = np.random.RandomState(42)
    rand_vec = rng.randn(len(null_vec))
    rand_vec = rand_vec / (np.linalg.norm(rand_vec) + 1e-30)
    rand_curvature = float(rand_vec @ H_reg @ rand_vec)

    results = []

    for eps in epsilons:
        row = {"epsilon": eps}

        # --- Null direction ---
        perturbation = eps * null_vec
        param_flat = param.detach().reshape(-1).clone()
        original_values = param_flat[param_indices].clone()

        # Apply perturbation
        with torch.no_grad():
            param_flat[param_indices] += torch.tensor(
                perturbation, dtype=param.dtype, device=device)
            param.data = param_flat.reshape(param.shape)

        # Measure loss
        with torch.no_grad():
            outputs = model(inputs).logits
            null_loss = loss_fn(outputs, labels).item()

        # Restore
        with torch.no_grad():
            param_flat[param_indices] = original_values
            param.data = param_flat.reshape(param.shape)

        delta_null = null_loss - base_loss
        rel_delta_null = abs(delta_null) / (abs(base_loss) + 1e-30)
        predicted_delta_null = 0.5 * eps**2 * null_curvature

        row["null_loss"] = null_loss
        row["null_delta"] = delta_null
        row["null_rel_delta"] = rel_delta_null
        row["null_predicted"] = predicted_delta_null
        row["null_curvature"] = null_curvature

        # --- Random direction (control) ---
        perturbation_rand = eps * rand_vec
        param_flat = param.detach().reshape(-1).clone()
        original_values = param_flat[param_indices].clone()

        with torch.no_grad():
            param_flat[param_indices] += torch.tensor(
                perturbation_rand, dtype=param.dtype, device=device)
            param.data = param_flat.reshape(param.shape)

        with torch.no_grad():
            outputs = model(inputs).logits
            rand_loss = loss_fn(outputs, labels).item()

        with torch.no_grad():
            param_flat[param_indices] = original_values
            param.data = param_flat.reshape(param.shape)

        delta_rand = rand_loss - base_loss
        rel_delta_rand = abs(delta_rand) / (abs(base_loss) + 1e-30)
        predicted_delta_rand = 0.5 * eps**2 * rand_curvature

        row["rand_loss"] = rand_loss
        row["rand_delta"] = delta_rand
        row["rand_rel_delta"] = rel_delta_rand
        row["rand_predicted"] = predicted_delta_rand
        row["rand_curvature"] = rand_curvature

        # Ratio: how much flatter is null vs random?
        row["flatness_ratio"] = rel_delta_rand / (rel_delta_null + 1e-30)

        results.append(row)

    return results, base_loss


# ============================================================
# 3. Main Experiment
# ============================================================

def run_step_test(args):
    import torch
    import torch.nn as nn
    import torchvision
    import torchvision.transforms as transforms
    from transformers import ViTForImageClassification

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")
    print(f"Model: ViT-Tiny, sub_dim: {args.sub_dim}")
    print(f"Epsilons: {args.epsilons}")
    print(f"Layers to test: {args.n_layers}")
    print(f"Null directions per layer: {args.n_dirs}")
    print("=" * 70)

    # Disable SDPA
    if hasattr(torch.backends, 'cuda'):
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)

    # Data
    print("\nLoading CIFAR-100...")
    mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
    tf = transforms.Compose([
        transforms.Resize(224), transforms.ToTensor(),
        transforms.Normalize(mean, std)])
    ds = torchvision.datasets.CIFAR100('./data', True, download=True, transform=tf)
    ds_sub = torch.utils.data.Subset(ds, list(range(args.subset_size)))
    dl = torch.utils.data.DataLoader(ds_sub, args.batch_size, shuffle=False,
                                      num_workers=2, pin_memory=True)
    loss_fn = nn.CrossEntropyLoss()

    # Get a fixed batch for all measurements
    images, labels = next(iter(dl))
    images, labels = images.to(device), labels.to(device)

    # Model (pretrained, no training)
    torch.manual_seed(0)
    model = ViTForImageClassification.from_pretrained(
        'WinKawaks/vit-tiny-patch16-224',
        num_labels=100, ignore_mismatched_sizes=True,
        attn_implementation="eager")
    model.to(device)
    model.eval()

    # Target layers
    layer_indices = [0, 5, 11][:args.n_layers]
    target_layers = {}
    for li in layer_indices:
        block = model.vit.encoder.layer[li]
        target_layers[f"attn_L{li}_q"] = block.attention.attention.query.weight
        target_layers[f"mlp_L{li}_f1"] = block.intermediate.dense.weight

    print(f"Testing {len(target_layers)} layers: {list(target_layers.keys())}")

    all_results = {}
    epsilons = args.epsilons

    for name, param in target_layers.items():
        print(f"\n--- {name} ({param.numel()} params) ---")

        num_params = param.numel()
        dim = min(args.sub_dim, num_params)
        if dim % 2 != 0:
            dim -= 1

        rng = np.random.RandomState(hash(name) % (2**31))
        indices = torch.tensor(
            sorted(rng.choice(num_params, size=dim, replace=False)),
            dtype=torch.long, device=device)

        param.requires_grad_(True)

        try:
            H = compute_sub_hessian(model, loss_fn, images, labels,
                                     indices, param, device)
            null_vecs, Q, H_reg = find_null_directions(H, dim)
        except Exception as e:
            print(f"  Error computing Hessian: {e}")
            param.requires_grad_(False)
            continue

        param.requires_grad_(False)

        if not null_vecs:
            print(f"  No null directions found")
            continue

        print(f"  Found {len(null_vecs)} null directions")

        # Test up to n_dirs null directions
        layer_results = []
        for di in range(min(args.n_dirs, len(null_vecs))):
            nvec = null_vecs[di]
            curvature = nvec @ H_reg @ nvec
            print(f"\n  Direction {di+1}: curvature={curvature:.2e}")

            results, base_loss = test_step_sizes(
                model, loss_fn, images, labels, param, indices,
                nvec, H_reg, device, epsilons)

            print(f"  Base loss: {base_loss:.6f}")
            print(f"  {'eps':>8s} | {'null Δ':>12s} {'null rel%':>10s} | "
                  f"{'rand Δ':>12s} {'rand rel%':>10s} | {'flat ratio':>10s}")
            print(f"  {'-'*72}")

            for r in results:
                print(f"  {r['epsilon']:8.4f} | "
                      f"{r['null_delta']:+12.6f} {r['null_rel_delta']*100:9.4f}% | "
                      f"{r['rand_delta']:+12.6f} {r['rand_rel_delta']*100:9.4f}% | "
                      f"{r['flatness_ratio']:10.1f}x")

            layer_results.append({
                "direction": di,
                "curvature": float(curvature),
                "base_loss": base_loss,
                "steps": results,
            })

        all_results[name] = layer_results

    # ============================================================
    # Summary
    # ============================================================
    print(f"\n{'='*70}")
    print("STEP SIZE SUMMARY")
    print(f"{'='*70}")

    # Find the largest epsilon where null direction keeps loss within 1%
    print(f"\n  Largest epsilon with <1% loss change (null vs random):\n")
    print(f"  {'Layer':<20s} {'Dir':>4s} {'Null ε_max':>12s} {'Rand ε_max':>12s} {'Ratio':>8s}")
    print(f"  {'-'*60}")

    for name, layer_results in all_results.items():
        for lr in layer_results:
            null_max = 0
            rand_max = 0
            for r in lr["steps"]:
                if r["null_rel_delta"] < 0.01:
                    null_max = r["epsilon"]
                if r["rand_rel_delta"] < 0.01:
                    rand_max = r["epsilon"]
            ratio = null_max / (rand_max + 1e-30)
            print(f"  {name:<20s} {lr['direction']:>4d} {null_max:>12.4f} {rand_max:>12.4f} {ratio:>7.1f}x")

    # Verdict
    print(f"\n  INTERPRETATION:")
    all_null_max = []
    for layer_results in all_results.values():
        for lr in layer_results:
            for r in lr["steps"]:
                if r["null_rel_delta"] < 0.01:
                    all_null_max.append(r["epsilon"])

    if all_null_max:
        max_safe = max(all_null_max)
        median_safe = sorted(all_null_max)[len(all_null_max)//2]
        print(f"  Max safe epsilon (null, <1% loss): {max_safe}")
        print(f"  Median safe epsilon: {median_safe}")

        if max_safe >= 1.0:
            print(f"  >>> NULL DIRECTIONS ARE REMARKABLY FLAT")
            print(f"  >>> Continual learning along null cone is FEASIBLE")
        elif max_safe >= 0.1:
            print(f"  >>> Null directions are moderately flat")
            print(f"  >>> Continual learning MAY be feasible with small steps")
        elif max_safe >= 0.01:
            print(f"  >>> Null directions are locally flat only")
            print(f"  >>> Continual learning along null cone is UNLIKELY to work")
        else:
            print(f"  >>> Null flatness is negligible at practical step sizes")

    # Save
    out_path = Path(args.output)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")


# ============================================================
# 4. Entry Point
# ============================================================

class Config:
    device = "cuda"
    batch_size = 64
    sub_dim = 50
    subset_size = 256       # Small batch for fast Hessian
    n_layers = 3            # Test layers at depth 0, 5, 11
    n_dirs = 3              # Test 3 null directions per layer
    epsilons = [1e-4, 1e-3, 1e-2, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    output = "null_step_test.json"

def main():
    args = Config()
    import sys
    if not any('jupyter' in a or 'ipykernel' in a or 'colab' in a for a in sys.argv):
        try:
            import argparse
            p = argparse.ArgumentParser()
            for k, v in vars(Config()).items():
                if isinstance(v, list):
                    p.add_argument(f"--{k}", type=float, nargs='+', default=v)
                else:
                    p.add_argument(f"--{k}", type=type(v), default=v)
            args = p.parse_args()
        except SystemExit:
            pass
    if args.sub_dim % 2 != 0:
        args.sub_dim -= 1
    run_step_test(args)

if __name__ == "__main__":
    main()
