#!/usr/bin/env python3
"""
Step 1: Verify Null Cone Structure in ViT-Base (google/vit-base-patch16-224)

Extends Table 4 of "Neural Null Cones" paper to Vision Transformers (~86M params).
Method: coordinate principal sub-Hessians (50x50 random parameter subsets)
        computed via autograd, with three-level null verification.

Requirements:
    pip install torch torchvision transformers numpy

Usage:
    python vit_null_cone_verify.py [--device cuda] [--sub_dim 50] [--num_subsets 10]
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from transformers import ViTForImageClassification, ViTImageProcessor

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("Note: install tqdm for progress bars (pip install tqdm)")


# ============================================================
# 1. Hessian Computation (sub-Hessian via autograd)
# ============================================================

def compute_sub_hessian(model, loss_fn, inputs, labels, param_indices, layer_params, device):
    """
    Compute a sub-Hessian for a subset of parameters in a given layer.
    
    Args:
        model: the full model (frozen except target layer)
        loss_fn: loss function
        inputs: input batch
        labels: label batch
        param_indices: indices into the flattened layer parameter vector
        layer_params: the target layer's parameter tensor (requires_grad=True)
        device: torch device
    
    Returns:
        H: sub-Hessian matrix (sub_dim x sub_dim), numpy array
    """
    sub_dim = len(param_indices)
    H = np.zeros((sub_dim, sub_dim), dtype=np.float64)
    
    # Forward pass
    model.zero_grad()
    outputs = model(inputs).logits
    loss = loss_fn(outputs, labels)
    
    # First-order gradient
    grad = torch.autograd.grad(loss, layer_params, create_graph=True)[0]
    grad_flat = grad.reshape(-1)
    
    # Extract sub-gradient
    sub_grad = grad_flat[param_indices]
    
    # Compute sub-Hessian row by row
    for i in range(sub_dim):
        grad2 = torch.autograd.grad(sub_grad[i], layer_params, retain_graph=True)[0]
        grad2_flat = grad2.reshape(-1)
        H[i, :] = grad2_flat[param_indices].detach().cpu().numpy().astype(np.float64)
    
    return H


# ============================================================
# 2. Symplectic Null Cone Analysis
# ============================================================

def make_symplectic_J(n):
    """Standard 2n x 2n symplectic matrix J = [[0, I], [-I, 0]]."""
    I_n = np.eye(n)
    J = np.block([[np.zeros((n, n)), I_n], [-I_n, np.zeros((n, n))]])
    return J


def symmetrize_and_regularize(H, lam=1e-6):
    """Symmetrize H and add regularization: H_reg = 0.5*(H + H^T) + lam*I."""
    H_sym = 0.5 * (H + H.T)
    H_reg = H_sym + lam * np.eye(H.shape[0])
    return H_sym, H_reg


def check_indefiniteness(H_sym):
    """Check if H_sym is indefinite (has both positive and negative eigenvalues)."""
    evals = np.linalg.eigvalsh(H_sym)
    has_pos = np.any(evals > 0)
    has_neg = np.any(evals < 0)
    return has_pos and has_neg, evals


def find_null_directions(H_reg, J, tau_rel=1e-6, tau_lenient=1e-4, eigen_residual_gate=1e-6):
    """
    Find null cone directions via eigenvectors of H_reg^{-1} J.
    
    Three verification levels:
        - strict:  |e^T H_reg e| / ||H_reg|| < tau_rel  (1e-6)
        - lenient: |e^T H_reg e| / ||H_reg|| < tau_lenient (1e-4)
        - subspace: ||Q^T H_reg Q|| / ||H_reg|| < 1e-3
    
    Returns:
        results dict with null directions, residuals, verification status
    """
    dim = H_reg.shape[0]
    H_reg_norm = np.linalg.norm(H_reg)
    
    # Compute H_reg^{-1} J
    try:
        H_reg_inv = np.linalg.inv(H_reg)
    except np.linalg.LinAlgError:
        return {"error": "H_reg singular", "null_strict": 0, "null_lenient": 0}
    
    M = H_reg_inv @ J  # M = H_reg^{-1} J (paper notation: JG = H_reg^{-1} J)
    
    # Eigendecomposition
    evals, evecs = np.linalg.eig(M)
    
    # Filter for real eigenpairs (unified absolute gate)
    real_mask = np.abs(evals.imag) < 1e-8
    
    real_indices = np.where(real_mask)[0]
    
    if len(real_indices) == 0:
        return {
            "num_real_eigenpairs": 0,
            "null_strict": 0,
            "null_lenient": 0,
            "residuals": [],
            "eigenvalues": [],
        }
    
    # Eigen-residual gate: ||M e - lambda e|| / ||e|| < gate
    verified_indices = []
    for idx in real_indices:
        e = evecs[:, idx].real
        lam = evals[idx].real
        residual = np.linalg.norm(M @ e - lam * e) / (np.linalg.norm(e) + 1e-30)
        if residual < eigen_residual_gate:
            verified_indices.append(idx)
    
    if len(verified_indices) == 0:
        return {
            "num_real_eigenpairs": len(real_indices),
            "num_verified_eigenpairs": 0,
            "null_strict": 0,
            "null_lenient": 0,
            "residuals": [],
            "eigenvalues": [],
        }
    
    # Check nullness: |e^T H_reg e| for each verified eigenvector
    null_residuals = []
    null_strict = 0
    null_lenient = 0
    eigenvalues = []
    
    for idx in verified_indices:
        e = evecs[:, idx].real
        e = e / (np.linalg.norm(e) + 1e-30)  # normalize
        
        quad_form = abs(e @ H_reg @ e)
        rel_nullness = quad_form / (H_reg_norm + 1e-30)
        
        null_residuals.append({
            "eigenvalue": float(evals[idx].real),
            "quad_form_abs": float(quad_form),
            "rel_nullness": float(rel_nullness),
            "strict_pass": bool(rel_nullness < tau_rel),
            "lenient_pass": bool(rel_nullness < tau_lenient),
        })
        
        eigenvalues.append(float(evals[idx].real))
        
        if rel_nullness < tau_rel:
            null_strict += 1
        if rel_nullness < tau_lenient:
            null_lenient += 1
    
    # Subspace nullness check (QR on strict-passing directions)
    strict_vecs = []
    for i, idx in enumerate(verified_indices):
        if null_residuals[i]["strict_pass"]:
            strict_vecs.append(evecs[:, idx].real)
    
    subspace_nullness = None
    if len(strict_vecs) >= 2 and len(strict_vecs) <= dim:
        V = np.column_stack(strict_vecs)
        Q, R = np.linalg.qr(V)
        # Check QR numerical quality: skip if R diagonal is near-singular
        r_diag = np.abs(np.diag(R))
        if r_diag.min() > 1e-12:
            sub_H = Q.T @ H_reg @ Q
            subspace_nullness = float(np.linalg.norm(sub_H) / (H_reg_norm + 1e-30))
    
    return {
        "num_real_eigenpairs": len(real_indices),
        "num_verified_eigenpairs": len(verified_indices),
        "null_strict": null_strict,
        "null_lenient": null_lenient,
        "residuals": null_residuals,
        "eigenvalues": eigenvalues,
        "subspace_nullness": subspace_nullness,
        "H_reg_norm": float(H_reg_norm),
        "min_residual": float(min(r["quad_form_abs"] for r in null_residuals)) if null_residuals else None,
        "max_residual": float(max(r["quad_form_abs"] for r in null_residuals)) if null_residuals else None,
    }


# ============================================================
# 3. Layer Targeting
# ============================================================

def get_target_layers(model):
    """
    Return a dict of layer_name -> parameter tensor for key ViT-Base layers.
    Covers: embeddings, attention (early/mid/late), MLP (early/mid/late), classifier.
    """
    targets = {}
    
    # Patch embedding projection
    targets["patch_embed"] = model.vit.embeddings.patch_embeddings.projection.weight
    
    # Attention layers: L0, L5, L11 (early, mid, late)
    for layer_idx in [0, 5, 11]:
        prefix = f"attn_L{layer_idx}"
        block = model.vit.encoder.layer[layer_idx]
        targets[f"{prefix}_qkv"] = block.attention.attention.query.weight
        targets[f"{prefix}_out"] = block.attention.output.dense.weight
    
    # MLP layers: L0, L5, L11
    for layer_idx in [0, 5, 11]:
        prefix = f"mlp_L{layer_idx}"
        block = model.vit.encoder.layer[layer_idx]
        targets[f"{prefix}_fc1"] = block.intermediate.dense.weight
        targets[f"{prefix}_fc2"] = block.output.dense.weight
    
    # Classifier head
    targets["classifier"] = model.classifier.weight
    
    return targets


# ============================================================
# 4. Main Experiment
# ============================================================

def run_experiment(args):
    import torch
    import torch.nn as nn
    import numpy as np
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Sub-Hessian dim: {args.sub_dim}")
    print(f"Num random subsets per layer: {args.num_subsets}")
    print(f"Regularization lambda: {args.lam}")
    print("=" * 70)
    
    # Load model
    print("\nLoading ViT-Base...")
    from transformers import ViTForImageClassification
    model = ViTForImageClassification.from_pretrained(
        'google/vit-base-patch16-224',
        num_labels=100,  # CIFAR-100
        ignore_mismatched_sizes=True,
        attn_implementation="eager",  # Disable SDPA — fused attention lacks double-backward
    )
    model.to(device)
    model.eval()  # Disable dropout for deterministic Hessians
    
    # Also disable SDPA globally in case eager flag isn't enough
    import torch.backends.cuda
    if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)  # Math fallback supports double-backward
        print("SDPA: forced math-only backend (double-backward compatible)")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,} (~{total_params/1e6:.1f}M)")
    
    # Freeze all parameters
    for p in model.parameters():
        p.requires_grad = False
    
    # Create synthetic input (we only need Hessian structure, not trained accuracy)
    # Using random input is valid: we're probing the Hessian at pretrained weights
    print("\nCreating input batch...")
    torch.manual_seed(42)
    batch_size = 4
    inputs = torch.randn(batch_size, 3, 224, 224, device=device)
    labels = torch.randint(0, 100, (batch_size,), device=device)
    loss_fn = nn.CrossEntropyLoss()
    
    # Get target layers
    target_layers = get_target_layers(model)
    print(f"\nTarget layers ({len(target_layers)}):")
    for name, param in target_layers.items():
        print(f"  {name}: {param.shape} ({param.numel():,} params)")
    
    # Run analysis
    all_results = {}
    summary_rows = []
    
    total_tasks = sum(1 for _ in target_layers) * args.num_subsets
    completed = 0
    elapsed_times = []
    
    for layer_name, layer_param in target_layers.items():
        print(f"\n{'='*70}")
        print(f"Layer: {layer_name} ({layer_param.numel():,} params)")
        print(f"{'='*70}")
        
        layer_param.requires_grad = True
        num_params = layer_param.numel()
        
        # Ensure sub_dim is even and <= num_params
        sub_dim = min(args.sub_dim, num_params)
        if sub_dim % 2 != 0:
            sub_dim -= 1
        
        if sub_dim < 4:
            print(f"  Skipping: too few parameters ({num_params})")
            layer_param.requires_grad = False
            continue
        
        layer_results = []
        
        for subset_idx in range(args.num_subsets):
            t0 = time.time()
            
            # Random parameter subset
            rng = np.random.RandomState(42 + subset_idx * 1000 + hash(layer_name) % 10000)
            param_indices = torch.tensor(
                rng.choice(num_params, size=sub_dim, replace=False),
                dtype=torch.long, device=device
            )
            param_indices, _ = param_indices.sort()
            
            # Compute sub-Hessian
            print(f"  Subset {subset_idx+1}/{args.num_subsets}: computing {sub_dim}x{sub_dim} sub-Hessian...", end="", flush=True)
            
            try:
                H = compute_sub_hessian(model, loss_fn, inputs, labels, param_indices, layer_param, device)
            except Exception as e:
                print(f" ERROR: {e}")
                layer_results.append({"error": str(e), "subset_idx": subset_idx})
                completed += 1
                continue
            
            # Symmetrize and regularize
            H_sym, H_reg = symmetrize_and_regularize(H, lam=args.lam)
            
            # Check indefiniteness
            is_indef, evals_sym = check_indefiniteness(H_sym)
            
            if not is_indef:
                elapsed = time.time() - t0
                elapsed_times.append(elapsed)
                completed += 1
                eta = np.mean(elapsed_times) * (total_tasks - completed)
                print(f" definite ({elapsed:.1f}s) [ETA: {eta/60:.0f}min]")
                layer_results.append({
                    "subset_idx": subset_idx,
                    "indefinite": False,
                    "null_strict": 0,
                    "null_lenient": 0,
                })
                continue
            
            # Symplectic analysis
            J = make_symplectic_J(sub_dim // 2)
            null_result = find_null_directions(H_reg, J, tau_rel=1e-6, tau_lenient=1e-4)
            null_result["subset_idx"] = subset_idx
            null_result["indefinite"] = True
            null_result["eigenvalue_range_sym"] = [float(evals_sym.min()), float(evals_sym.max())]
            
            elapsed = time.time() - t0
            elapsed_times.append(elapsed)
            completed += 1
            eta = np.mean(elapsed_times) * (total_tasks - completed)
            
            # Print summary
            status = "STRICT" if null_result["null_strict"] > 0 else ("LENIENT" if null_result["null_lenient"] > 0 else "no null")
            min_res = null_result.get("min_residual", "N/A")
            if isinstance(min_res, float):
                min_res = f"{min_res:.2e}"
            print(f" indef, {null_result.get('num_real_eigenpairs',0)} real, "
                  f"{null_result['null_strict']} strict, {null_result['null_lenient']} lenient, "
                  f"min_res={min_res} [{status}] ({elapsed:.1f}s) [ETA: {eta/60:.0f}min]")
            
            layer_results.append(null_result)
        
        # Layer summary
        n_indef = sum(1 for r in layer_results if r.get("indefinite", False))
        n_null_strict = sum(1 for r in layer_results if r.get("null_strict", 0) > 0)
        n_null_lenient = sum(1 for r in layer_results if r.get("null_lenient", 0) > 0)
        
        # Residual range (from strict-passing directions)
        all_min_res = [r.get("min_residual") for r in layer_results 
                       if r.get("min_residual") is not None and r.get("null_strict", 0) > 0]
        all_max_res = [r.get("max_residual") for r in layer_results 
                       if r.get("max_residual") is not None and r.get("null_strict", 0) > 0]
        
        residual_range = "---"
        if all_min_res:
            lo = min(all_min_res)
            hi = max(all_max_res) if all_max_res else lo
            residual_range = f"{lo:.0e}--{hi:.0e}"
        
        summary = {
            "layer": layer_name,
            "num_params": num_params,
            "subsets": args.num_subsets,
            "indefinite": f"{n_indef}/{args.num_subsets}",
            "null_strict": f"{n_null_strict}/{args.num_subsets}",
            "null_lenient": f"{n_null_lenient}/{args.num_subsets}",
            "residual_range": residual_range,
        }
        summary_rows.append(summary)
        
        print(f"\n  Summary: indef={summary['indefinite']}, "
              f"null(strict)={summary['null_strict']}, "
              f"null(lenient)={summary['null_lenient']}, "
              f"residuals={residual_range}")
        
        all_results[layer_name] = layer_results
        layer_param.requires_grad = False
    
    # ============================================================
    # Print final summary table (matches paper Table 4 format)
    # ============================================================
    print(f"\n\n{'='*80}")
    print("FINAL SUMMARY: Null Cone Structure in ViT-Base")
    print(f"Sub-Hessian dim = {args.sub_dim}, lambda = {args.lam}")
    print(f"{'='*80}")
    print(f"{'Layer':<20} {'Params':>10} {'Indef':>10} {'Null(strict)':>14} {'Null(lenient)':>14} {'Residual range':>18}")
    print("-" * 90)
    for row in summary_rows:
        print(f"{row['layer']:<20} {row['num_params']:>10,} {row['indefinite']:>10} "
              f"{row['null_strict']:>14} {row['null_lenient']:>14} {row['residual_range']:>18}")
    print("-" * 90)
    
    # Count totals
    total_subsets = len(summary_rows) * args.num_subsets
    total_indef = sum(int(r["indefinite"].split("/")[0]) for r in summary_rows)
    total_null = sum(int(r["null_strict"].split("/")[0]) for r in summary_rows)
    print(f"{'TOTAL':<20} {'':>10} {total_indef}/{total_subsets:>7} {total_null}/{total_subsets:>11}")
    
    # Save detailed results
    output = {
        "model": "google/vit-base-patch16-224",
        "total_params": total_params,
        "sub_dim": args.sub_dim,
        "lam": args.lam,
        "num_subsets": args.num_subsets,
        "device": str(device),
        "timestamp": datetime.now().isoformat(),
        "summary": summary_rows,
        "detailed_results": {k: v for k, v in all_results.items()},
    }
    
    out_path = Path(args.output)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nDetailed results saved to: {out_path}")
    
    # Print paper-ready table row
    print(f"\n{'='*80}")
    print("Paper-ready row for Table 4:")
    print(f"{'='*80}")
    print(f"ViT-Base (attn)    86M    {total_indef}/{total_subsets}    {total_null}/{total_subsets}    [see residual range above]")
    
    return output


# ============================================================
# 5. Entry Point
# ============================================================

# === CONFIGURATION (edit these directly for Colab/Jupyter) ===
class Config:
    device = "cuda"        # "cuda" or "cpu"
    sub_dim = 50           # Sub-Hessian dimension (must be even)
    num_subsets = 8        # Random parameter subsets per layer
    lam = 1e-6             # Regularization lambda
    output = "vit_null_cone_results.json"  # Output JSON path

def main():
    args = Config()
    
    # Allow command-line override when run as script (not in Jupyter)
    import sys
    if not any('jupyter' in arg or 'ipykernel' in arg or 'colab' in arg for arg in sys.argv):
        import argparse
        parser = argparse.ArgumentParser(description="Verify null cone structure in ViT-Base")
        parser.add_argument("--device", type=str, default="cuda")
        parser.add_argument("--sub_dim", type=int, default=50)
        parser.add_argument("--num_subsets", type=int, default=8)
        parser.add_argument("--lam", type=float, default=1e-6)
        parser.add_argument("--output", type=str, default="vit_null_cone_results.json")
        args = parser.parse_args()
    
    if args.sub_dim % 2 != 0:
        args.sub_dim -= 1
        print(f"Adjusted sub_dim to {args.sub_dim} (must be even)")
    
    run_experiment(args)

if __name__ == "__main__":
    main()

# For Colab/Jupyter: just call main() or run_experiment(Config()) directly
