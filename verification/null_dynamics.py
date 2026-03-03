#!/usr/bin/env python3
"""
Null Cone Dynamics Tracker

Core question: Does SGD naturally align with null directions during training?

Measurements at each probe step:
  1. null_dim:      Number of null directions in 50x50 sub-Hessian
  2. align_cos:     Cosine similarity between SGD gradient and null subspace
  3. null_frac:     Fraction of gradient norm in null subspace ||g_null|| / ||g||
  4. curvature:     Curvature cost of actual SGD step: g^T H_reg g / ||g||^2
  5. null_curv:     Curvature cost of null-projected step (should be ~0)
  6. indef_ratio:   Fraction of negative eigenvalues in H_sym (indefiniteness strength)

If align_cos increases during training → SGD naturally discovers null directions
If null_frac increases → SGD increasingly moves along null cone
If curvature decreases while null_frac increases → null alignment reduces curvature cost

Models: ViT-Tiny on CIFAR-100 (to match NCA-SGD experiment)

Usage (Colab):
    Config.probe_every = 10   # measure every 10 steps
    Config.epochs = 20
    main()
"""

import time
import json
from pathlib import Path
import numpy as np


# ============================================================
# 1. Null Cone Analysis (from verification script)
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


def analyze_null_structure(H, sub_dim, lam=1e-6):
    """
    Full analysis of null cone structure in a sub-Hessian.

    Returns dict with:
      - indefinite: bool
      - indef_ratio: fraction of negative eigenvalues
      - num_null: number of null directions found
      - Q: orthonormal null basis (or None)
      - H_reg: regularized Hessian
      - H_reg_norm: Frobenius norm of H_reg
    """
    H_sym = 0.5 * (H + H.T)
    H_reg = H_sym + lam * np.eye(sub_dim)
    H_reg_norm = np.linalg.norm(H_reg)

    evals = np.linalg.eigvalsh(H_sym)
    n_neg = np.sum(evals < 0)
    n_pos = np.sum(evals > 0)
    indefinite = (n_neg > 0) and (n_pos > 0)
    indef_ratio = n_neg / len(evals)

    result = {
        "indefinite": indefinite,
        "indef_ratio": float(indef_ratio),
        "num_null": 0,
        "Q": None,
        "H_reg": H_reg,
        "H_reg_norm": float(H_reg_norm),
    }

    if not indefinite:
        return result

    try:
        H_reg_inv = np.linalg.inv(H_reg)
    except np.linalg.LinAlgError:
        return result

    J = make_symplectic_J(sub_dim // 2)
    M = H_reg_inv @ J
    evals_M, evecs_M = np.linalg.eig(M)

    real_mask = np.abs(evals_M.imag) < 1e-8
    real_indices = np.where(real_mask)[0]
    if len(real_indices) == 0:
        return result

    null_vecs = []
    for idx in real_indices:
        e = evecs_M[:, idx].real
        lam_e = evals_M[idx].real
        if np.linalg.norm(M @ e - lam_e * e) / (np.linalg.norm(e) + 1e-30) > 1e-6:
            continue
        e_norm = e / (np.linalg.norm(e) + 1e-30)
        if abs(e_norm @ H_reg @ e_norm) / (H_reg_norm + 1e-30) < 1e-6:
            null_vecs.append(e_norm)

    if null_vecs:
        V = np.column_stack(null_vecs)
        Q, R = np.linalg.qr(V)
        # Check for near-singular R
        if np.min(np.abs(np.diag(R))) < 1e-12:
            Q = None
        result["Q"] = Q
        result["num_null"] = len(null_vecs)

    return result


# ============================================================
# 2. Gradient-Null Alignment Measurement
# ============================================================

def measure_alignment(grad_sub, Q, H_reg, H_reg_norm):
    """
    Measure alignment between gradient and null subspace.

    Args:
        grad_sub: gradient in sub-Hessian coordinates (sub_dim,)
        Q: null subspace basis (sub_dim, k)
        H_reg: regularized Hessian
        H_reg_norm: norm of H_reg

    Returns dict with:
        null_frac: ||g_null|| / ||g||
        align_cos: max |cos(g, q_i)| over null basis vectors
        curvature_full: g^T H_reg g / (||g||^2 * ||H_reg||)
        curvature_null: g_null^T H_reg g_null / (||g_null||^2 * ||H_reg||)
    """
    g = grad_sub.astype(np.float64)
    g_norm = np.linalg.norm(g)
    if g_norm < 1e-30:
        return {"null_frac": 0, "align_cos": 0,
                "curvature_full": 0, "curvature_null": 0}

    # Project onto null subspace
    g_null = Q @ (Q.T @ g)
    g_null_norm = np.linalg.norm(g_null)
    null_frac = g_null_norm / g_norm

    # Max cosine with individual null directions
    k = Q.shape[1]
    cos_vals = [abs(g @ Q[:, i]) / g_norm for i in range(k)]
    align_cos = max(cos_vals) if cos_vals else 0.0

    # Curvature cost of full gradient
    g_unit = g / g_norm
    curvature_full = abs(g_unit @ H_reg @ g_unit) / (H_reg_norm + 1e-30)

    # Curvature cost of null-projected gradient
    if g_null_norm > 1e-30:
        g_null_unit = g_null / g_null_norm
        curvature_null = abs(g_null_unit @ H_reg @ g_null_unit) / (H_reg_norm + 1e-30)
    else:
        curvature_null = 0.0

    return {
        "null_frac": float(null_frac),
        "align_cos": float(align_cos),
        "curvature_full": float(curvature_full),
        "curvature_null": float(curvature_null),
    }


# ============================================================
# 3. Probe Function
# ============================================================

def probe_null_dynamics(model, loss_fn, inputs, labels, target_layers,
                        sub_dim, device, lam=1e-6):
    """
    Run one full probe: for each target layer, compute null structure
    and measure gradient alignment.

    Returns list of per-layer measurement dicts.
    """
    import torch
    measurements = []

    for name, param in target_layers.items():
        num_params = param.numel()
        dim = min(sub_dim, num_params)
        if dim % 2 != 0:
            dim -= 1
        if dim < 4:
            continue

        rng = np.random.RandomState(hash(name) % (2**31))
        indices = torch.tensor(
            sorted(rng.choice(num_params, size=dim, replace=False)),
            dtype=torch.long, device=device)

        param.requires_grad_(True)

        try:
            # Compute sub-Hessian
            H = compute_sub_hessian(model, loss_fn, inputs, labels,
                                     indices, param, device)

            # Analyze null structure
            info = analyze_null_structure(H, dim, lam)

            # Get gradient in sub-Hessian coordinates
            model.zero_grad()
            outputs = model(inputs).logits
            loss = loss_fn(outputs, labels)
            loss.backward()
            grad_flat = param.grad.reshape(-1)
            grad_sub = grad_flat[indices].detach().cpu().numpy()

            m = {
                "layer": name,
                "indefinite": info["indefinite"],
                "indef_ratio": info["indef_ratio"],
                "num_null": info["num_null"],
            }

            if info["Q"] is not None and info["num_null"] > 0:
                align = measure_alignment(
                    grad_sub, info["Q"], info["H_reg"], info["H_reg_norm"])
                m.update(align)
            else:
                m.update({"null_frac": 0, "align_cos": 0,
                          "curvature_full": 0, "curvature_null": 0})

            measurements.append(m)

        except Exception as e:
            measurements.append({
                "layer": name, "error": str(e),
                "indefinite": False, "num_null": 0,
                "null_frac": 0, "align_cos": 0,
                "curvature_full": 0, "curvature_null": 0,
                "indef_ratio": 0,
            })

        param.requires_grad_(False)

    return measurements


# ============================================================
# 4. Training + Tracking Loop
# ============================================================

def get_probe_layers(model):
    """Select layers to probe: 3 attention + 3 MLP across depth."""
    targets = {}
    for li in [0, 5, 11]:
        block = model.vit.encoder.layer[li]
        targets[f"attn_L{li}_q"] = block.attention.attention.query.weight
        targets[f"mlp_L{li}_f1"] = block.intermediate.dense.weight
    return targets


def run_tracking(args):
    import torch
    import torch.nn as nn
    import torchvision
    import torchvision.transforms as transforms
    from transformers import ViTForImageClassification

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")
    print(f"Epochs: {args.epochs}, LR: {args.lr}")
    print(f"Probe every: {args.probe_every} steps, sub_dim: {args.sub_dim}")
    print("=" * 70)

    # Disable SDPA
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
        ds_train = torch.utils.data.Subset(ds_train,
            list(range(min(args.subset_size, len(ds_train)))))
        ds_val = torch.utils.data.Subset(ds_val,
            list(range(min(args.subset_size // 5, len(ds_val)))))
        print(f"Subset: {len(ds_train)} train, {len(ds_val)} val")

    dl_train = torch.utils.data.DataLoader(ds_train, args.batch_size, shuffle=True,
                                            num_workers=2, pin_memory=True)
    dl_val = torch.utils.data.DataLoader(ds_val, args.batch_size, shuffle=False,
                                          num_workers=2, pin_memory=True)
    loss_fn = nn.CrossEntropyLoss()

    # Model
    torch.manual_seed(args.seed)
    model = ViTForImageClassification.from_pretrained(
        'WinKawaks/vit-tiny-patch16-224',
        num_labels=100, ignore_mismatched_sizes=True,
        attn_implementation="eager")
    model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                 momentum=0.9, weight_decay=1e-4)

    probe_layers = get_probe_layers(model)
    print(f"Probing {len(probe_layers)} layers: {list(probe_layers.keys())}")

    # Tracking storage
    all_probes = []       # list of {step, epoch, measurements, train_loss}
    epoch_metrics = []    # list of {epoch, val_acc, val_loss}

    step = 0
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        n_batches = 0

        for images, labels in dl_train:
            images, labels = images.to(device), labels.to(device)

            # === PROBE ===
            if step % args.probe_every == 0:
                model.eval()
                with torch.enable_grad():
                    measurements = probe_null_dynamics(
                        model, loss_fn, images, labels,
                        probe_layers, args.sub_dim, device)
                model.train()

                # Aggregate
                avg_null_frac = np.mean([m["null_frac"] for m in measurements])
                avg_align = np.mean([m["align_cos"] for m in measurements])
                avg_null_dim = np.mean([m["num_null"] for m in measurements])
                avg_curv = np.mean([m["curvature_full"] for m in measurements])
                avg_indef = np.mean([m["indef_ratio"] for m in measurements])
                n_indef = sum(1 for m in measurements if m["indefinite"])

                probe_record = {
                    "step": step,
                    "epoch": epoch,
                    "avg_null_frac": float(avg_null_frac),
                    "avg_align_cos": float(avg_align),
                    "avg_null_dim": float(avg_null_dim),
                    "avg_curvature": float(avg_curv),
                    "avg_indef_ratio": float(avg_indef),
                    "n_indefinite": n_indef,
                    "n_layers": len(measurements),
                    "per_layer": measurements,
                }
                all_probes.append(probe_record)

                if step % (args.probe_every * 5) == 0:
                    print(f"  [Probe] step={step} ep={epoch} | "
                          f"null_frac={avg_null_frac:.4f} "
                          f"align={avg_align:.4f} "
                          f"null_dim={avg_null_dim:.1f} "
                          f"curv={avg_curv:.6f} "
                          f"indef={n_indef}/{len(measurements)}")

            # === TRAIN STEP ===
            optimizer.zero_grad()
            outputs = model(images).logits
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1
            step += 1

        # Validation
        model.eval()
        vloss = correct = total = 0
        with torch.no_grad():
            for imgs, labs in dl_val:
                imgs, labs = imgs.to(device), labs.to(device)
                out = model(imgs).logits
                vloss += loss_fn(out, labs).item()
                correct += out.argmax(1).eq(labs).sum().item()
                total += labs.size(0)

        avg_val = vloss / len(dl_val)
        acc = 100.0 * correct / total
        avg_train = running_loss / n_batches

        epoch_metrics.append({
            "epoch": epoch, "train_loss": avg_train,
            "val_loss": avg_val, "val_acc": acc,
        })

        # Latest probe stats
        latest = all_probes[-1] if all_probes else {}
        nf = latest.get("avg_null_frac", 0)
        al = latest.get("avg_align_cos", 0)
        nd = latest.get("avg_null_dim", 0)

        print(f"  Ep {epoch+1}/{args.epochs}: "
              f"train={avg_train:.4f} val={avg_val:.4f} acc={acc:.1f}% | "
              f"null_frac={nf:.4f} align={al:.4f} null_dim={nd:.1f}")

    # ============================================================
    # Summary
    # ============================================================
    print(f"\n{'='*70}")
    print("DYNAMICS SUMMARY")
    print(f"{'='*70}")

    if len(all_probes) >= 4:
        # Compare first quarter vs last quarter
        n = len(all_probes)
        q1 = all_probes[:n//4]
        q4 = all_probes[3*n//4:]

        metrics = ["avg_null_frac", "avg_align_cos", "avg_null_dim", "avg_curvature"]
        labels = ["Null fraction", "Alignment cos", "Null dimensions", "Curvature cost"]

        print(f"\n  {'Metric':<20s} {'Early (Q1)':>12s} {'Late (Q4)':>12s} {'Change':>12s}")
        print(f"  {'-'*56}")
        for metric, label in zip(metrics, labels):
            early = np.mean([p[metric] for p in q1])
            late = np.mean([p[metric] for p in q4])
            change = late - early
            pct = 100 * change / (abs(early) + 1e-30)
            print(f"  {label:<20s} {early:>12.6f} {late:>12.6f} {change:>+12.6f} ({pct:+.1f}%)")

        # Key test: does null_frac increase?
        early_nf = np.mean([p["avg_null_frac"] for p in q1])
        late_nf = np.mean([p["avg_null_frac"] for p in q4])
        if late_nf > early_nf * 1.1:
            print(f"\n  >>> SGD gradient INCREASINGLY ALIGNS with null cone (+{100*(late_nf/early_nf-1):.0f}%)")
            print(f"  >>> This suggests SGD naturally discovers and follows null directions.")
        elif late_nf < early_nf * 0.9:
            print(f"\n  >>> SGD gradient DECREASINGLY ALIGNS with null cone ({100*(late_nf/early_nf-1):.0f}%)")
            print(f"  >>> Null directions exist but SGD does not naturally exploit them.")
        else:
            print(f"\n  >>> Null alignment is STABLE throughout training (±10%).")
            print(f"  >>> Null directions are persistent but not increasingly exploited.")

    # Save
    results = {
        "config": vars(args) if hasattr(args, '__dict__') else str(args),
        "probes": all_probes,
        "epochs": epoch_metrics,
    }
    out_path = Path(args.output)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")
    print(f"Total probes: {len(all_probes)}")

    return results


# ============================================================
# 5. Entry Point
# ============================================================

class Config:
    device = "cuda"
    epochs = 20
    batch_size = 64
    lr = 0.01
    sub_dim = 50
    probe_every = 10      # Probe null structure every N steps
    seed = 0
    subset_size = 5000    # 0 = full dataset
    output = "null_dynamics.json"

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
    run_tracking(args)

if __name__ == "__main__":
    main()
