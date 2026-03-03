#!/usr/bin/env python3
"""
Universal Null Cone Verification for Transformer Models

Verifies null cone structure in loss Hessians across architectures:
  - ViT-Base  (vision, 86M)
  - BERT-Base (encoder, 110M)
  - GPT-2     (causal LM, 124M)

Method: coordinate principal sub-Hessians (50x50) via autograd,
        with three-level null verification (strict/lenient/subspace).

Requirements:
    pip install torch transformers numpy

Usage (command line):
    python null_cone_verify.py --model vit
    python null_cone_verify.py --model bert
    python null_cone_verify.py --model gpt2

Usage (Colab/Jupyter):
    Config.model = "bert"   # or "vit", "gpt2"
    main()
"""

import time
import json
from datetime import datetime
from pathlib import Path

import numpy as np


# ============================================================
# 1. Hessian Computation (sub-Hessian via autograd)
# ============================================================

def compute_sub_hessian(model, loss_fn, inputs, labels, param_indices,
                        layer_params, device, input_kwargs=None):
    """
    Compute a sub-Hessian for a subset of parameters in a given layer.

    Args:
        model: the full model (frozen except target layer)
        loss_fn: loss function
        inputs: input tensor (input_ids for text, pixel_values for vision)
        labels: label tensor
        param_indices: indices into the flattened layer parameter vector
        layer_params: the target layer's parameter tensor (requires_grad=True)
        device: torch device
        input_kwargs: optional extra kwargs for model forward (e.g. attention_mask)

    Returns:
        H: sub-Hessian matrix (sub_dim x sub_dim), numpy float64
    """
    import torch
    sub_dim = len(param_indices)
    H = np.zeros((sub_dim, sub_dim), dtype=np.float64)

    model.zero_grad()
    if input_kwargs:
        outputs = model(inputs, **input_kwargs)
    else:
        outputs = model(inputs)

    # Handle different output formats
    logits = outputs.logits if hasattr(outputs, 'logits') else outputs
    loss = loss_fn(logits, labels)

    grad = torch.autograd.grad(loss, layer_params, create_graph=True)[0]
    grad_flat = grad.reshape(-1)
    sub_grad = grad_flat[param_indices]

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
    return np.block([[np.zeros((n, n)), I_n], [-I_n, np.zeros((n, n))]])


def symmetrize_and_regularize(H, lam=1e-6):
    """H_reg = 0.5*(H + H^T) + lam*I."""
    H_sym = 0.5 * (H + H.T)
    H_reg = H_sym + lam * np.eye(H.shape[0])
    return H_sym, H_reg


def check_indefiniteness(H_sym):
    """Check if H_sym has both positive and negative eigenvalues."""
    evals = np.linalg.eigvalsh(H_sym)
    return (np.any(evals > 0) and np.any(evals < 0)), evals


def find_null_directions(H_reg, J, tau_rel=1e-6, tau_lenient=1e-4, eigen_residual_gate=1e-6):
    """
    Find null cone directions via eigenvectors of M = H_reg^{-1} J.

    Three verification levels:
        strict:  |e^T H_reg e| / ||H_reg|| < tau_rel
        lenient: |e^T H_reg e| / ||H_reg|| < tau_lenient
        subspace: ||Q^T H_reg Q|| / ||H_reg|| < 1e-3
    """
    dim = H_reg.shape[0]
    H_reg_norm = np.linalg.norm(H_reg)

    try:
        H_reg_inv = np.linalg.inv(H_reg)
    except np.linalg.LinAlgError:
        return {"error": "H_reg singular", "null_strict": 0, "null_lenient": 0}

    M = H_reg_inv @ J  # M = H_reg^{-1} J
    evals, evecs = np.linalg.eig(M)

    # Filter for real eigenpairs (unified absolute gate)
    real_mask = np.abs(evals.imag) < 1e-8
    real_indices = np.where(real_mask)[0]

    if len(real_indices) == 0:
        return {
            "num_real_eigenpairs": 0, "null_strict": 0, "null_lenient": 0,
            "residuals": [], "eigenvalues": [],
        }

    # Eigen-residual gate
    verified_indices = []
    for idx in real_indices:
        e = evecs[:, idx].real
        lam = evals[idx].real
        res = np.linalg.norm(M @ e - lam * e) / (np.linalg.norm(e) + 1e-30)
        if res < eigen_residual_gate:
            verified_indices.append(idx)

    if len(verified_indices) == 0:
        return {
            "num_real_eigenpairs": len(real_indices),
            "num_verified_eigenpairs": 0,
            "null_strict": 0, "null_lenient": 0,
            "residuals": [], "eigenvalues": [],
        }

    # Nullness check
    null_residuals = []
    null_strict = null_lenient = 0
    eigenvalues = []

    for idx in verified_indices:
        e = evecs[:, idx].real
        e = e / (np.linalg.norm(e) + 1e-30)
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

    # Subspace nullness (QR on strict-passing directions)
    strict_vecs = [evecs[:, verified_indices[i]].real
                   for i in range(len(verified_indices))
                   if null_residuals[i]["strict_pass"]]

    subspace_nullness = None
    if len(strict_vecs) >= 2 and len(strict_vecs) <= dim:
        V = np.column_stack(strict_vecs)
        Q, R = np.linalg.qr(V)
        r_diag = np.abs(np.diag(R))
        if r_diag.min() > 1e-12:
            sub_H = Q.T @ H_reg @ Q
            subspace_nullness = float(np.linalg.norm(sub_H) / (H_reg_norm + 1e-30))

    return {
        "num_real_eigenpairs": len(real_indices),
        "num_verified_eigenpairs": len(verified_indices),
        "null_strict": null_strict, "null_lenient": null_lenient,
        "residuals": null_residuals, "eigenvalues": eigenvalues,
        "subspace_nullness": subspace_nullness,
        "H_reg_norm": float(H_reg_norm),
        "min_residual": float(min(r["quad_form_abs"] for r in null_residuals)) if null_residuals else None,
        "max_residual": float(max(r["quad_form_abs"] for r in null_residuals)) if null_residuals else None,
    }


# ============================================================
# 3. Model Registry: load model, create inputs, get target layers
# ============================================================

def setup_vit(device):
    """ViT-Base for image classification."""
    import torch
    import torch.nn as nn
    from transformers import ViTForImageClassification

    model = ViTForImageClassification.from_pretrained(
        'google/vit-base-patch16-224',
        num_labels=100, ignore_mismatched_sizes=True,
        attn_implementation="eager",
    )
    model.to(device).eval()

    # Disable SDPA fused backends for double-backward
    import torch.backends.cuda
    if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        print("SDPA: forced math-only backend")

    for p in model.parameters():
        p.requires_grad = False

    # Inputs
    torch.manual_seed(42)
    inputs = torch.randn(4, 3, 224, 224, device=device)
    labels = torch.randint(0, 100, (4,), device=device)

    # Target layers
    targets = {}
    targets["patch_embed"] = model.vit.embeddings.patch_embeddings.projection.weight
    for li in [0, 5, 11]:
        block = model.vit.encoder.layer[li]
        targets[f"attn_L{li}_query"] = block.attention.attention.query.weight
        targets[f"attn_L{li}_out"] = block.attention.output.dense.weight
        targets[f"mlp_L{li}_fc1"] = block.intermediate.dense.weight
        targets[f"mlp_L{li}_fc2"] = block.output.dense.weight
    targets["classifier"] = model.classifier.weight

    return model, inputs, labels, nn.CrossEntropyLoss(), targets, None


def setup_bert(device):
    """BERT-Base for sequence classification."""
    import torch
    import torch.nn as nn
    from transformers import BertForSequenceClassification, BertTokenizer

    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=10,
        attn_implementation="eager",
    )
    model.to(device).eval()

    # Disable SDPA
    import torch.backends.cuda
    if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        print("SDPA: forced math-only backend")

    for p in model.parameters():
        p.requires_grad = False

    # Inputs: random token IDs (Hessian structure, not accuracy)
    torch.manual_seed(42)
    seq_len = 64
    input_ids = torch.randint(100, 30000, (4, seq_len), device=device)
    attention_mask = torch.ones(4, seq_len, dtype=torch.long, device=device)
    labels = torch.randint(0, 10, (4,), device=device)

    input_kwargs = {"attention_mask": attention_mask}

    # Target layers: embeddings, attention (L0, L5, L11), MLP (L0, L5, L11), classifier
    targets = {}
    targets["word_embed"] = model.bert.embeddings.word_embeddings.weight

    for li in [0, 5, 11]:
        block = model.bert.encoder.layer[li]
        targets[f"attn_L{li}_query"] = block.attention.self.query.weight
        targets[f"attn_L{li}_key"] = block.attention.self.key.weight
        targets[f"attn_L{li}_out"] = block.attention.output.dense.weight
        targets[f"mlp_L{li}_fc1"] = block.intermediate.dense.weight
        targets[f"mlp_L{li}_fc2"] = block.output.dense.weight

    targets["classifier"] = model.classifier.weight

    return model, input_ids, labels, nn.CrossEntropyLoss(), targets, input_kwargs


def setup_gpt2(device):
    """GPT-2 for causal language modeling."""
    import torch
    import torch.nn as nn
    from transformers import GPT2LMHeadModel

    model = GPT2LMHeadModel.from_pretrained(
        'gpt2',
        attn_implementation="eager",
    )
    model.to(device).eval()

    import torch.backends.cuda
    if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        print("SDPA: forced math-only backend")

    for p in model.parameters():
        p.requires_grad = False

    # Inputs
    torch.manual_seed(42)
    seq_len = 64
    input_ids = torch.randint(100, 50000, (4, seq_len), device=device)
    # For causal LM, labels = shifted input_ids (handled internally by GPT2LMHeadModel)
    labels = input_ids.clone()

    # Target layers
    targets = {}
    targets["wte"] = model.transformer.wte.weight  # token embedding

    for li in [0, 5, 11]:
        block = model.transformer.h[li]
        targets[f"attn_L{li}_qkv"] = block.attn.c_attn.weight
        targets[f"attn_L{li}_out"] = block.attn.c_proj.weight
        targets[f"mlp_L{li}_fc1"] = block.mlp.c_fc.weight
        targets[f"mlp_L{li}_fc2"] = block.mlp.c_proj.weight

    targets["lm_head"] = model.lm_head.weight

    return model, input_ids, labels, nn.CrossEntropyLoss(), targets, {"labels": labels}


MODEL_REGISTRY = {
    "vit": ("ViT-Base (86M)", setup_vit),
    "bert": ("BERT-Base (110M)", setup_bert),
    "gpt2": ("GPT-2 (124M)", setup_gpt2),
}


# ============================================================
# 4. Main Experiment Loop
# ============================================================

def run_experiment(args):
    import torch
    import torch.nn as nn

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model_key = args.model.lower()

    if model_key not in MODEL_REGISTRY:
        print(f"Unknown model '{model_key}'. Available: {list(MODEL_REGISTRY.keys())}")
        return

    model_name, setup_fn = MODEL_REGISTRY[model_key]

    print(f"Model: {model_name}")
    print(f"Device: {device}")
    print(f"Sub-Hessian dim: {args.sub_dim}")
    print(f"Num random subsets per layer: {args.num_subsets}")
    print(f"Regularization lambda: {args.lam}")
    print("=" * 70)

    # Setup
    print(f"\nLoading {model_name}...")
    model, inputs, labels, loss_fn, target_layers, input_kwargs = setup_fn(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,} (~{total_params/1e6:.1f}M)")

    print(f"\nTarget layers ({len(target_layers)}):")
    for name, param in target_layers.items():
        print(f"  {name}: {param.shape} ({param.numel():,} params)")

    # Run analysis
    all_results = {}
    summary_rows = []
    total_tasks = len(target_layers) * args.num_subsets
    completed = 0
    elapsed_times = []

    for layer_name, layer_param in target_layers.items():
        print(f"\n{'='*70}")
        print(f"Layer: {layer_name} ({layer_param.numel():,} params)")
        print(f"{'='*70}")

        layer_param.requires_grad = True
        num_params = layer_param.numel()

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

            rng = np.random.RandomState(42 + subset_idx * 1000 + hash(layer_name) % 10000)
            param_indices = torch.tensor(
                rng.choice(num_params, size=sub_dim, replace=False),
                dtype=torch.long, device=device
            )
            param_indices, _ = param_indices.sort()

            print(f"  Subset {subset_idx+1}/{args.num_subsets}: computing {sub_dim}x{sub_dim} sub-Hessian...",
                  end="", flush=True)

            try:
                H = compute_sub_hessian(
                    model, loss_fn, inputs, labels,
                    param_indices, layer_param, device,
                    input_kwargs=input_kwargs
                )
            except Exception as e:
                elapsed = time.time() - t0
                elapsed_times.append(elapsed)
                completed += 1
                print(f" ERROR: {e}")
                layer_results.append({"error": str(e), "subset_idx": subset_idx})
                continue

            H_sym, H_reg = symmetrize_and_regularize(H, lam=args.lam)
            is_indef, evals_sym = check_indefiniteness(H_sym)

            if not is_indef:
                elapsed = time.time() - t0
                elapsed_times.append(elapsed)
                completed += 1
                eta = np.mean(elapsed_times) * (total_tasks - completed)
                print(f" definite ({elapsed:.1f}s) [ETA: {eta/60:.0f}min]")
                layer_results.append({
                    "subset_idx": subset_idx, "indefinite": False,
                    "null_strict": 0, "null_lenient": 0,
                })
                continue

            J = make_symplectic_J(sub_dim // 2)
            null_result = find_null_directions(H_reg, J, tau_rel=1e-6, tau_lenient=1e-4)
            null_result["subset_idx"] = subset_idx
            null_result["indefinite"] = True
            null_result["eigenvalue_range_sym"] = [float(evals_sym.min()), float(evals_sym.max())]

            elapsed = time.time() - t0
            elapsed_times.append(elapsed)
            completed += 1
            eta = np.mean(elapsed_times) * (total_tasks - completed)

            status = ("STRICT" if null_result["null_strict"] > 0
                      else ("LENIENT" if null_result["null_lenient"] > 0 else "no null"))
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
            "layer": layer_name, "num_params": num_params,
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
    # Final summary
    # ============================================================
    print(f"\n\n{'='*80}")
    print(f"FINAL SUMMARY: Null Cone Structure in {model_name}")
    print(f"Sub-Hessian dim = {args.sub_dim}, lambda = {args.lam}")
    print(f"{'='*80}")
    print(f"{'Layer':<22} {'Params':>12} {'Indef':>8} {'Null(strict)':>14} {'Null(lenient)':>14} {'Residual range':>18}")
    print("-" * 92)
    for row in summary_rows:
        print(f"{row['layer']:<22} {row['num_params']:>12,} {row['indefinite']:>8} "
              f"{row['null_strict']:>14} {row['null_lenient']:>14} {row['residual_range']:>18}")
    print("-" * 92)

    total_subsets = len(summary_rows) * args.num_subsets
    total_indef = sum(int(r["indefinite"].split("/")[0]) for r in summary_rows)
    total_null = sum(int(r["null_strict"].split("/")[0]) for r in summary_rows)
    print(f"{'TOTAL':<22} {'':>12} {total_indef}/{total_subsets:>5} {total_null}/{total_subsets:>11}")

    # Save
    output = {
        "model": model_name, "model_key": model_key,
        "total_params": total_params,
        "sub_dim": args.sub_dim, "lam": args.lam,
        "num_subsets": args.num_subsets, "device": str(device),
        "timestamp": datetime.now().isoformat(),
        "summary": summary_rows,
        "detailed_results": {k: v for k, v in all_results.items()},
    }

    out_path = Path(args.output)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nDetailed results saved to: {out_path}")

    return output


# ============================================================
# 5. Entry Point
# ============================================================

class Config:
    model = "bert"         # "vit", "bert", or "gpt2"
    device = "cuda"
    sub_dim = 50
    num_subsets = 8
    lam = 1e-6
    output = "null_cone_results.json"

def main():
    args = Config()

    # Auto-set output filename
    if args.output == "null_cone_results.json":
        args.output = f"{args.model}_null_cone_results.json"

    # Allow command-line override when not in Jupyter
    import sys
    if not any('jupyter' in arg or 'ipykernel' in arg or 'colab' in arg for arg in sys.argv):
        try:
            import argparse
            parser = argparse.ArgumentParser(description="Universal null cone verification")
            parser.add_argument("--model", type=str, default="bert",
                                choices=["vit", "bert", "gpt2"])
            parser.add_argument("--device", type=str, default="cuda")
            parser.add_argument("--sub_dim", type=int, default=50)
            parser.add_argument("--num_subsets", type=int, default=8)
            parser.add_argument("--lam", type=float, default=1e-6)
            parser.add_argument("--output", type=str, default="null_cone_results.json")
            args = parser.parse_args()
            if args.output == "null_cone_results.json":
                args.output = f"{args.model}_null_cone_results.json"
        except SystemExit:
            pass  # Fallback to Config defaults

    if args.sub_dim % 2 != 0:
        args.sub_dim -= 1
        print(f"Adjusted sub_dim to {args.sub_dim} (must be even)")

    run_experiment(args)

if __name__ == "__main__":
    main()
