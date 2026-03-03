#!/usr/bin/env python3
"""
Null Direction Step Size Test — TRAINED MODEL

Key insight: Pretrained ViT landscape is globally flat, so null vs random
difference is small. After training on CIFAR-100, landscape sharpens and
null directions should show much larger relative advantage.

Protocol:
  1. Train ViT-Tiny on CIFAR-100 (5K subset, 20 epochs SGD) — same as NCA experiment
  2. Test step sizes on the TRAINED model
  3. Compare with pretrained model results

Hypothesis: After training, curvature increases on non-null directions
but null directions remain flat → flatness_ratio should be >> 1.

Usage (Colab):
    main()
"""

import time
import json
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
    import torch
    model.eval()
    with torch.no_grad():
        outputs = model(inputs).logits
        base_loss = loss_fn(outputs, labels).item()

    null_curvature = float(null_vec @ H_reg @ null_vec)

    rng = np.random.RandomState(42)
    rand_vec = rng.randn(len(null_vec))
    rand_vec = rand_vec / (np.linalg.norm(rand_vec) + 1e-30)
    rand_curvature = float(rand_vec @ H_reg @ rand_vec)

    # Also test the max-curvature direction (eigenvector of H_reg with largest |eigenvalue|)
    evals_H, evecs_H = np.linalg.eigh(H_reg)
    max_idx = np.argmax(np.abs(evals_H))
    max_curv_vec = evecs_H[:, max_idx]
    max_curv_vec = max_curv_vec / (np.linalg.norm(max_curv_vec) + 1e-30)
    max_curvature = float(max_curv_vec @ H_reg @ max_curv_vec)

    results = []
    for eps in epsilons:
        row = {"epsilon": eps}

        # --- Null direction ---
        param_flat = param.detach().reshape(-1).clone()
        orig = param_flat[param_indices].clone()
        with torch.no_grad():
            param_flat[param_indices] += torch.tensor(
                eps * null_vec, dtype=param.dtype, device=device)
            param.data = param_flat.reshape(param.shape)
            null_loss = loss_fn(model(inputs).logits, labels).item()
            param_flat[param_indices] = orig
            param.data = param_flat.reshape(param.shape)

        row["null_delta"] = null_loss - base_loss
        row["null_rel"] = abs(null_loss - base_loss) / (abs(base_loss) + 1e-30)

        # --- Random direction ---
        param_flat = param.detach().reshape(-1).clone()
        orig = param_flat[param_indices].clone()
        with torch.no_grad():
            param_flat[param_indices] += torch.tensor(
                eps * rand_vec, dtype=param.dtype, device=device)
            param.data = param_flat.reshape(param.shape)
            rand_loss = loss_fn(model(inputs).logits, labels).item()
            param_flat[param_indices] = orig
            param.data = param_flat.reshape(param.shape)

        row["rand_delta"] = rand_loss - base_loss
        row["rand_rel"] = abs(rand_loss - base_loss) / (abs(base_loss) + 1e-30)

        # --- Max curvature direction ---
        param_flat = param.detach().reshape(-1).clone()
        orig = param_flat[param_indices].clone()
        with torch.no_grad():
            param_flat[param_indices] += torch.tensor(
                eps * max_curv_vec, dtype=param.dtype, device=device)
            param.data = param_flat.reshape(param.shape)
            max_loss = loss_fn(model(inputs).logits, labels).item()
            param_flat[param_indices] = orig
            param.data = param_flat.reshape(param.shape)

        row["max_delta"] = max_loss - base_loss
        row["max_rel"] = abs(max_loss - base_loss) / (abs(base_loss) + 1e-30)

        # Ratios
        row["null_vs_rand"] = row["rand_rel"] / (row["null_rel"] + 1e-30)
        row["null_vs_max"] = row["max_rel"] / (row["null_rel"] + 1e-30)

        results.append(row)

    return results, base_loss, {
        "null_curvature": null_curvature,
        "rand_curvature": rand_curvature,
        "max_curvature": max_curvature,
        "max_eigenvalue": float(evals_H[max_idx]),
    }


# ============================================================
# 3. Training
# ============================================================

def train_model(model, dl_train, dl_val, loss_fn, device, epochs, lr):
    import torch
    optimizer = torch.optim.SGD(model.parameters(), lr=lr,
                                 momentum=0.9, weight_decay=1e-4)
    for epoch in range(epochs):
        model.train()
        rloss = nb = 0
        for imgs, labs in dl_train:
            imgs, labs = imgs.to(device), labs.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(imgs).logits, labs)
            loss.backward()
            optimizer.step()
            rloss += loss.item()
            nb += 1

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for imgs, labs in dl_val:
                imgs, labs = imgs.to(device), labs.to(device)
                correct += model(imgs).logits.argmax(1).eq(labs).sum().item()
                total += labs.size(0)
        acc = 100 * correct / total
        print(f"  Ep {epoch+1}/{epochs}: train={rloss/nb:.4f} acc={acc:.1f}%")

    return model


# ============================================================
# 4. Main
# ============================================================

def run_experiment(args):
    import torch
    import torch.nn as nn
    import torchvision
    import torchvision.transforms as transforms
    from transformers import ViTForImageClassification

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print("=" * 70)

    if hasattr(torch.backends, 'cuda'):
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)

    # Data
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

    dl_train = torch.utils.data.DataLoader(ds_train, args.batch_size, shuffle=True,
                                            num_workers=2, pin_memory=True)
    dl_val = torch.utils.data.DataLoader(ds_val, args.batch_size, shuffle=False,
                                          num_workers=2, pin_memory=True)
    loss_fn = nn.CrossEntropyLoss()

    # Fixed test batch (from training set for Hessian consistency)
    test_imgs, test_labs = next(iter(dl_train))
    test_imgs, test_labs = test_imgs.to(device), test_labs.to(device)

    # ============================================================
    # Phase 1: Test PRETRAINED model
    # ============================================================
    print("\n" + "=" * 70)
    print("PHASE 1: PRETRAINED MODEL")
    print("=" * 70)

    torch.manual_seed(0)
    model_pre = ViTForImageClassification.from_pretrained(
        'WinKawaks/vit-tiny-patch16-224',
        num_labels=100, ignore_mismatched_sizes=True,
        attn_implementation="eager").to(device)
    model_pre.eval()

    pre_results = run_step_test_on_model(
        model_pre, loss_fn, test_imgs, test_labs, device, args)

    # ============================================================
    # Phase 2: Train model
    # ============================================================
    print("\n" + "=" * 70)
    print("PHASE 2: TRAINING (20 epochs)")
    print("=" * 70)

    torch.manual_seed(0)
    model_trained = ViTForImageClassification.from_pretrained(
        'WinKawaks/vit-tiny-patch16-224',
        num_labels=100, ignore_mismatched_sizes=True,
        attn_implementation="eager").to(device)

    model_trained = train_model(model_trained, dl_train, dl_val, loss_fn,
                                 device, args.epochs, args.lr)

    # ============================================================
    # Phase 3: Test TRAINED model
    # ============================================================
    print("\n" + "=" * 70)
    print("PHASE 3: TRAINED MODEL")
    print("=" * 70)

    post_results = run_step_test_on_model(
        model_trained, loss_fn, test_imgs, test_labs, device, args)

    # ============================================================
    # Phase 4: Comparison
    # ============================================================
    print("\n" + "=" * 70)
    print("COMPARISON: PRETRAINED vs TRAINED")
    print("=" * 70)

    eps_targets = [0.1, 1.0, 5.0, 10.0]

    print(f"\n  {'':20s} |{'--- PRETRAINED ---':^36s}|{'--- TRAINED ---':^36s}")
    print(f"  {'Layer':<13s} {'eps':>6s} | {'null%':>8s} {'rand%':>8s} {'max%':>8s} | "
          f"{'null%':>8s} {'rand%':>8s} {'max%':>8s} | {'Δratio':>7s}")
    print(f"  {'-'*95}")

    for layer_name in pre_results:
        if layer_name not in post_results:
            continue
        pre_layer = pre_results[layer_name]
        post_layer = post_results[layer_name]
        if not pre_layer or not post_layer:
            continue

        # Use first direction
        pre_steps = pre_layer[0]["steps"]
        post_steps = post_layer[0]["steps"]

        for target_eps in eps_targets:
            pre_row = next((r for r in pre_steps if abs(r["epsilon"] - target_eps) < 0.001), None)
            post_row = next((r for r in post_steps if abs(r["epsilon"] - target_eps) < 0.001), None)
            if pre_row and post_row:
                pre_ratio = pre_row["null_vs_rand"]
                post_ratio = post_row["null_vs_rand"]
                delta_ratio = post_ratio / (pre_ratio + 1e-30)
                print(f"  {layer_name:<13s} {target_eps:>6.1f} | "
                      f"{pre_row['null_rel']*100:>7.3f}% {pre_row['rand_rel']*100:>7.3f}% "
                      f"{pre_row['max_rel']*100:>7.3f}% | "
                      f"{post_row['null_rel']*100:>7.3f}% {post_row['rand_rel']*100:>7.3f}% "
                      f"{post_row['max_rel']*100:>7.3f}% | "
                      f"{delta_ratio:>6.1f}x")

    # Verdict
    print(f"\n  INTERPRETATION:")
    print(f"  If trained 'Δratio' >> 1: training sharpens non-null directions")
    print(f"    → null cone advantage INCREASES with training")
    print(f"  If trained 'Δratio' ≈ 1: training sharpens all directions equally")
    print(f"    → null cone advantage is CONSTANT")
    print(f"  If trained 'Δratio' < 1: training flattens non-null more than null")
    print(f"    → null cone advantage DECREASES with training")

    # Save
    all_data = {"pretrained": pre_results, "trained": post_results}
    out_path = Path(args.output)
    with open(out_path, "w") as f:
        json.dump(all_data, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")


def run_step_test_on_model(model, loss_fn, images, labels, device, args):
    import torch
    model.eval()

    layer_indices = [0, 5, 11][:args.n_layers]
    target_layers = {}
    for li in layer_indices:
        block = model.vit.encoder.layer[li]
        target_layers[f"attn_L{li}_q"] = block.attention.attention.query.weight
        target_layers[f"mlp_L{li}_f1"] = block.intermediate.dense.weight

    epsilons = args.epsilons
    all_results = {}

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
            print(f"  Error: {e}")
            param.requires_grad_(False)
            continue
        param.requires_grad_(False)

        if not null_vecs:
            print(f"  No null directions found")
            all_results[name] = []
            continue

        print(f"  Found {len(null_vecs)} null directions")
        layer_results = []

        for di in range(min(args.n_dirs, len(null_vecs))):
            nvec = null_vecs[di]
            results, base_loss, curvatures = test_step_sizes(
                model, loss_fn, images, labels, param, indices,
                nvec, H_reg, device, epsilons)

            print(f"\n  Dir {di+1}: null_curv={curvatures['null_curvature']:.2e} "
                  f"rand_curv={curvatures['rand_curvature']:.2e} "
                  f"max_curv={curvatures['max_curvature']:.2e}")
            print(f"  Base loss: {base_loss:.6f}")
            print(f"  {'eps':>8s} | {'null%':>9s} {'rand%':>9s} {'max%':>9s} | "
                  f"{'n/r':>6s} {'n/m':>6s}")
            print(f"  {'-'*58}")

            for r in results:
                print(f"  {r['epsilon']:8.4f} | "
                      f"{r['null_rel']*100:8.4f}% {r['rand_rel']*100:8.4f}% "
                      f"{r['max_rel']*100:8.4f}% | "
                      f"{r['null_vs_rand']:5.1f}x {r['null_vs_max']:5.1f}x")

            layer_results.append({
                "direction": di,
                "curvatures": curvatures,
                "base_loss": base_loss,
                "steps": results,
            })

        all_results[name] = layer_results

    return all_results


class Config:
    device = "cuda"
    epochs = 20
    batch_size = 64
    lr = 0.01
    sub_dim = 50
    subset_size = 5000
    n_layers = 3
    n_dirs = 2            # 2 per layer (faster than 3, enough to compare)
    epsilons = [1e-4, 1e-3, 1e-2, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    output = "null_step_trained.json"

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
    run_experiment(args)

if __name__ == "__main__":
    main()
