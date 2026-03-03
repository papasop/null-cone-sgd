#!/usr/bin/env python3
"""
NCCL v2: Dual-Head Null Cone Continual Learning

v1 problem: shared 100-class classifier. Updating outputs for Task B
directly corrupts Task A outputs, masking backbone protection.

v2 fix:
  - head_a: 50 classes (0-49), FROZEN after Phase 1
  - head_b: 50 classes (50-99), NEW, trained freely in Phase 2
  - Labels remapped: Task A uses 0-49, Task B uses 0-49 (local indices)

This isolates backbone protection: any Task A accuracy drop is purely
from backbone representation changes, not classifier head corruption.

Usage (Colab):
    main()
"""

import time
import json
from pathlib import Path
import numpy as np
import copy


# ============================================================
# 1. Null Cone Core
# ============================================================

def make_symplectic_J(n):
    I_n = np.eye(n)
    return np.block([[np.zeros((n, n)), I_n], [-I_n, np.zeros((n, n))]])


def compute_sub_hessian(model_fn, loss_fn, inputs, labels, param_indices,
                        layer_params, device):
    import torch
    sub_dim = len(param_indices)
    H = np.zeros((sub_dim, sub_dim), dtype=np.float64)
    outputs = model_fn(inputs)
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
        return [], None
    try:
        H_reg_inv = np.linalg.inv(H_reg)
    except np.linalg.LinAlgError:
        return [], None
    J = make_symplectic_J(sub_dim // 2)
    M = H_reg_inv @ J
    evals_M, evecs_M = np.linalg.eig(M)
    real_mask = np.abs(evals_M.imag) < 1e-8
    real_indices = np.where(real_mask)[0]
    if len(real_indices) == 0:
        return [], None
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
        return [], None
    V = np.column_stack(null_vecs)
    Q, R = np.linalg.qr(V)
    if np.min(np.abs(np.diag(R))) < 1e-12:
        return null_vecs, None
    return null_vecs, Q


# ============================================================
# 2. Dual-Head Model
# ============================================================

class DualHeadViT:
    """
    ViT backbone + two independent classifier heads.
    head_a: Linear(hidden, 50) for classes 0-49
    head_b: Linear(hidden, 50) for classes 50-99
    """
    def __init__(self, base_model, device):
        import torch
        import torch.nn as nn
        self.device = device
        self.backbone = base_model.vit
        hidden = base_model.config.hidden_size  # 192

        # head_a initialized from pretrained classifier rows 0-49
        self.head_a = nn.Linear(hidden, 50).to(device)
        with torch.no_grad():
            self.head_a.weight.copy_(base_model.classifier.weight[:50])
            self.head_a.bias.copy_(base_model.classifier.bias[:50])

        # head_b freshly initialized
        self.head_b = nn.Linear(hidden, 50).to(device)
        nn.init.xavier_uniform_(self.head_b.weight)
        nn.init.zeros_(self.head_b.bias)

        self.backbone.to(device)

    def _features(self, x):
        return self.backbone(x).last_hidden_state[:, 0]  # CLS token

    def forward_a(self, x):
        return self.head_a(self._features(x))

    def forward_b(self, x):
        return self.head_b(self._features(x))

    def freeze_head_a(self):
        for p in self.head_a.parameters():
            p.requires_grad_(False)

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad_(False)

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad_(True)

    def train(self):
        self.backbone.train()
        self.head_a.train()
        self.head_b.train()

    def eval(self):
        self.backbone.eval()
        self.head_a.eval()
        self.head_b.eval()

    def backbone_state(self):
        return copy.deepcopy(self.backbone.state_dict())

    def head_a_state(self):
        return copy.deepcopy(self.head_a.state_dict())

    def head_b_state(self):
        return copy.deepcopy(self.head_b.state_dict())

    def load_backbone(self, sd):
        self.backbone.load_state_dict(sd)

    def load_head_a(self, sd):
        self.head_a.load_state_dict(sd)

    def load_head_b(self, sd):
        self.head_b.load_state_dict(sd)

    def trainable_params_for_phase2(self, method):
        """Return parameters to optimize in Phase 2."""
        if method == "freeze":
            return list(self.head_b.parameters())
        else:
            return (list(self.backbone.parameters()) +
                    list(self.head_b.parameters()))


# ============================================================
# 3. Data: class-split with label remapping
# ============================================================

def make_split_loaders(subset_size, batch_size):
    import torch
    import torchvision
    import torchvision.transforms as transforms

    mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
    tf_train = transforms.Compose([
        transforms.Resize(224), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), transforms.Normalize(mean, std)])
    tf_val = transforms.Compose([
        transforms.Resize(224), transforms.ToTensor(),
        transforms.Normalize(mean, std)])

    ds_train_full = torchvision.datasets.CIFAR100('./data', True, download=True, transform=tf_train)
    ds_val_full = torchvision.datasets.CIFAR100('./data', False, download=True, transform=tf_val)

    class RemappedSubset(torch.utils.data.Dataset):
        """Subset filtered by class range, with labels remapped to 0-based."""
        def __init__(self, dataset, class_range, max_samples=0):
            self.dataset = dataset
            self.offset = min(class_range)
            targets = dataset.targets if hasattr(dataset, 'targets') else \
                      [dataset[i][1] for i in range(len(dataset))]
            self.indices = [i for i, t in enumerate(targets) if t in class_range]
            if max_samples > 0:
                self.indices = self.indices[:max_samples]

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, idx):
            img, label = self.dataset[self.indices[idx]]
            return img, label - self.offset  # Remap: 0-49 or 50-99 → 0-49

    classes_a = set(range(0, 50))
    classes_b = set(range(50, 100))

    train_a = RemappedSubset(ds_train_full, classes_a, subset_size)
    train_b = RemappedSubset(ds_train_full, classes_b, subset_size)
    val_a = RemappedSubset(ds_val_full, classes_a, 0)  # Full val set
    val_b = RemappedSubset(ds_val_full, classes_b, 0)

    dl = lambda ds, shuffle=True: torch.utils.data.DataLoader(
        ds, batch_size, shuffle=shuffle, num_workers=2, pin_memory=True)

    print(f"  Task A: {len(train_a)} train, {len(val_a)} val (classes 0-49 → labels 0-49)")
    print(f"  Task B: {len(train_b)} train, {len(val_b)} val (classes 50-99 → labels 0-49)")

    return dl(train_a), dl(val_a, False), dl(train_b), dl(val_b, False)


# ============================================================
# 4. NCCL Engine (uses forward_a for Hessian)
# ============================================================

class NCCLEngine:
    def __init__(self, dual_model, task_a_loader, loss_fn,
                 sub_dim=50, n_subsets=3, update_every=25, device='cuda'):
        import torch
        self.model = dual_model
        self.task_a_loader = task_a_loader
        self.task_a_iter = iter(task_a_loader)
        self.loss_fn = loss_fn
        self.sub_dim = sub_dim
        self.n_subsets = n_subsets
        self.update_every = update_every
        self.device = device

        # Get target backbone layers
        self.target_layers = self._get_target_layers()
        self.projections = {name: [] for name in self.target_layers}
        self.stats = {"updates": 0, "null_layers": 0, "total_null_dirs": 0}

    def _get_target_layers(self):
        targets = {}
        for li in range(12):
            block = self.model.backbone.encoder.layer[li]
            targets[f"attn_L{li}_q"] = block.attention.attention.query.weight
            targets[f"attn_L{li}_o"] = block.attention.output.dense.weight
            targets[f"mlp_L{li}_f1"] = block.intermediate.dense.weight
            targets[f"mlp_L{li}_f2"] = block.output.dense.weight
        return targets

    def _get_task_a_batch(self):
        import torch
        try:
            imgs, labs = next(self.task_a_iter)
        except StopIteration:
            self.task_a_iter = iter(self.task_a_loader)
            imgs, labs = next(self.task_a_iter)
        return imgs.to(self.device), labs.to(self.device)

    def update_null_directions(self):
        import torch
        self.stats["updates"] += 1
        imgs, labs = self._get_task_a_batch()

        for name, param in self.target_layers.items():
            self.projections[name] = []
            num_params = param.numel()
            dim = min(self.sub_dim, num_params)
            if dim % 2 != 0:
                dim -= 1
            if dim < 4:
                continue

            param.requires_grad_(True)

            for s in range(self.n_subsets):
                rng = np.random.RandomState(
                    int(time.time() * 1000 + s * 7919) % (2**31))
                indices = torch.tensor(
                    sorted(rng.choice(num_params, size=dim, replace=False)),
                    dtype=torch.long, device=self.device)

                try:
                    H = compute_sub_hessian(
                        self.model.forward_a, self.loss_fn, imgs, labs,
                        indices, param, self.device)
                    null_vecs, Q = find_null_directions(H, dim)
                except Exception:
                    null_vecs, Q = [], None

                if Q is not None:
                    self.projections[name].append((indices, Q))
                    self.stats["null_layers"] += 1
                    self.stats["total_null_dirs"] += len(null_vecs)

            param.requires_grad_(False)

    def project_gradients(self):
        """Project backbone gradients onto null subspace of Task A."""
        import torch

        for name, param in self.target_layers.items():
            if param.grad is None:
                continue
            projections = self.projections[name]
            if not projections:
                param.grad.zero_()
                continue

            grad_flat = param.grad.reshape(-1)
            for indices, Q in projections:
                sub_grad = grad_flat[indices].detach().cpu().numpy().astype(np.float64)
                g_null = Q @ (Q.T @ sub_grad)
                with torch.no_grad():
                    grad_flat[indices] = torch.tensor(
                        g_null, dtype=grad_flat.dtype, device=self.device)


# ============================================================
# 5. EWC Engine (backbone only)
# ============================================================

class EWCEngine:
    def __init__(self, dual_model, task_a_loader, loss_fn,
                 ewc_lambda=1000, n_samples=500, device='cuda'):
        import torch
        self.model = dual_model
        self.ewc_lambda = ewc_lambda
        self.device = device

        # Store backbone params after Task A
        self.theta_star = {}
        for name, param in dual_model.backbone.named_parameters():
            self.theta_star[name] = param.detach().clone()

        # Compute Fisher on backbone only
        print("    Computing Fisher (backbone only)...")
        self.fisher = {}
        for name, param in dual_model.backbone.named_parameters():
            self.fisher[name] = torch.zeros_like(param)

        dual_model.eval()
        count = 0
        for imgs, labs in task_a_loader:
            if count >= n_samples:
                break
            imgs, labs = imgs.to(device), labs.to(device)
            for p in dual_model.backbone.parameters():
                if p.grad is not None:
                    p.grad.zero_()
            out = dual_model.forward_a(imgs)
            loss = loss_fn(out, labs)
            loss.backward()
            for name, param in dual_model.backbone.named_parameters():
                if param.grad is not None:
                    self.fisher[name] += param.grad.detach() ** 2
            count += imgs.size(0)

        for name in self.fisher:
            self.fisher[name] /= count
        print(f"    Fisher computed on {count} samples (backbone only)")

    def penalty(self):
        loss = 0.0
        for name, param in self.model.backbone.named_parameters():
            if name in self.fisher:
                loss += (self.fisher[name] *
                         (param - self.theta_star[name]) ** 2).sum()
        return 0.5 * self.ewc_lambda * loss


# ============================================================
# 6. Evaluation
# ============================================================

def evaluate(dual_model, loader, loss_fn, device, task='a', label=""):
    import torch
    dual_model.eval()
    correct = total = 0
    total_loss = 0.0
    forward_fn = dual_model.forward_a if task == 'a' else dual_model.forward_b
    with torch.no_grad():
        for imgs, labs in loader:
            imgs, labs = imgs.to(device), labs.to(device)
            out = forward_fn(imgs)
            total_loss += loss_fn(out, labs).item()
            correct += out.argmax(1).eq(labs).sum().item()
            total += labs.size(0)
    acc = 100.0 * correct / total
    avg_loss = total_loss / len(loader)
    if label:
        print(f"    {label}: acc={acc:.1f}% loss={avg_loss:.4f}")
    return acc, avg_loss


# ============================================================
# 7. Training Functions
# ============================================================

def train_task_a(dual_model, dl_train, dl_val, loss_fn, device, epochs, lr):
    """Train backbone + head_a on Task A."""
    import torch
    print("\n  Training Task A (backbone + head_a)...")
    params = list(dual_model.backbone.parameters()) + \
             list(dual_model.head_a.parameters())
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=1e-4)

    for ep in range(epochs):
        dual_model.train()
        rloss = nb = 0
        for imgs, labs in dl_train:
            imgs, labs = imgs.to(device), labs.to(device)
            optimizer.zero_grad()
            loss = loss_fn(dual_model.forward_a(imgs), labs)
            loss.backward()
            optimizer.step()
            rloss += loss.item()
            nb += 1
        acc, _ = evaluate(dual_model, dl_val, loss_fn, device, 'a')
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"    Ep {ep+1}/{epochs}: train={rloss/nb:.4f} val_A={acc:.1f}%")


def train_phase2(dual_model, method, dl_train_b, dl_train_a, dl_val_a, dl_val_b,
                 loss_fn, device, epochs, lr, ewc_engine=None, nccl_engine=None):
    """Phase 2: learn Task B while preserving Task A."""
    import torch

    label = method.upper()
    print(f"\n  [{label}] Training Task B...")

    # Freeze head_a for all methods
    dual_model.freeze_head_a()

    # Freeze backbone for Freeze method
    if method == "freeze":
        dual_model.freeze_backbone()

    params = dual_model.trainable_params_for_phase2(method)
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=1e-4)

    step = 0
    for ep in range(epochs):
        dual_model.train()
        rloss = nb = 0
        for imgs, labs in dl_train_b:
            imgs, labs = imgs.to(device), labs.to(device)

            # NCCL: update null directions periodically
            if method == "nccl" and nccl_engine and step % nccl_engine.update_every == 0:
                dual_model.eval()
                with torch.enable_grad():
                    nccl_engine.update_null_directions()
                dual_model.train()

            optimizer.zero_grad()
            out = dual_model.forward_b(imgs)
            loss = loss_fn(out, labs)

            # EWC: add penalty
            if method == "ewc" and ewc_engine:
                loss = loss + ewc_engine.penalty()

            loss.backward()

            # NCCL: project backbone gradients
            if method == "nccl" and nccl_engine:
                nccl_engine.project_gradients()

            optimizer.step()
            rloss += loss.item()
            nb += 1
            step += 1

        if (ep + 1) % 5 == 0 or ep == 0:
            acc_a, _ = evaluate(dual_model, dl_val_a, loss_fn, device, 'a')
            acc_b, _ = evaluate(dual_model, dl_val_b, loss_fn, device, 'b')
            extra = ""
            if nccl_engine:
                extra = f" | upd={nccl_engine.stats['updates']} null={nccl_engine.stats['null_layers']}"
            print(f"    Ep {ep+1}/{epochs}: loss={rloss/nb:.4f} "
                  f"A={acc_a:.1f}% B={acc_b:.1f}%{extra}")

    # Unfreeze for next method
    if method == "freeze":
        dual_model.unfreeze_backbone()


# ============================================================
# 8. Main Experiment
# ============================================================

def run_experiment(args):
    import torch
    import torch.nn as nn
    from transformers import ViTForImageClassification

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Architecture: Dual-head ViT-Tiny")
    print(f"Task A: {args.epochs_a} epochs, Task B: {args.epochs_b} epochs")
    print(f"EWC lambda: {args.ewc_lambda}")
    print(f"NCCL: update_every={args.update_every}, n_subsets={args.n_subsets}")
    print("=" * 70)

    if hasattr(torch.backends, 'cuda'):
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)

    print("\nLoading CIFAR-100 (class split, remapped labels)...")
    dl_train_a, dl_val_a, dl_train_b, dl_val_b = make_split_loaders(
        args.subset_size, args.batch_size)
    loss_fn = nn.CrossEntropyLoss()

    def make_dual_model():
        torch.manual_seed(args.seed)
        base = ViTForImageClassification.from_pretrained(
            'WinKawaks/vit-tiny-patch16-224',
            num_labels=100, ignore_mismatched_sizes=True,
            attn_implementation="eager")
        return DualHeadViT(base, device)

    # ============================================================
    # Phase 1: Train on Task A
    # ============================================================
    print("\n" + "=" * 70)
    print("PHASE 1: TASK A TRAINING (backbone + head_a)")
    print("=" * 70)

    model_base = make_dual_model()
    train_task_a(model_base, dl_train_a, dl_val_a, loss_fn, device,
                 args.epochs_a, args.lr)
    acc_a_base, _ = evaluate(model_base, dl_val_a, loss_fn, device, 'a',
                              "Task A (after training)")

    # Save state
    backbone_state = model_base.backbone_state()
    head_a_state = model_base.head_a_state()

    results = {"task_a_baseline": acc_a_base, "methods": {}}

    # ============================================================
    # Phase 2: Learn Task B with each method
    # ============================================================
    print("\n" + "=" * 70)
    print("PHASE 2: TASK B LEARNING (head_a FROZEN)")
    print("=" * 70)

    methods = ["naive", "ewc", "nccl", "freeze"]

    for method in methods:
        print(f"\n--- {method.upper()} ---")
        dm = make_dual_model()
        dm.load_backbone(backbone_state)
        dm.load_head_a(head_a_state)
        # head_b is fresh for each method

        ewc_engine = None
        nccl_engine = None

        if method == "ewc":
            ewc_engine = EWCEngine(dm, dl_train_a, loss_fn,
                                    ewc_lambda=args.ewc_lambda, device=device)

        if method == "nccl":
            nccl_engine = NCCLEngine(dm, dl_train_a, loss_fn,
                                      sub_dim=args.sub_dim,
                                      n_subsets=args.n_subsets,
                                      update_every=args.update_every,
                                      device=device)

        train_phase2(dm, method, dl_train_b, dl_train_a, dl_val_a, dl_val_b,
                     loss_fn, device, args.epochs_b, args.lr_b,
                     ewc_engine, nccl_engine)

        acc_a, _ = evaluate(dm, dl_val_a, loss_fn, device, 'a', "Task A")
        acc_b, _ = evaluate(dm, dl_val_b, loss_fn, device, 'b', "Task B")
        forgetting = acc_a_base - acc_a

        results["methods"][method] = {
            "task_a": acc_a, "task_b": acc_b, "forgetting": forgetting,
        }
        if nccl_engine:
            results["methods"][method]["nccl_stats"] = nccl_engine.stats

    # ============================================================
    # Summary
    # ============================================================
    print(f"\n{'='*70}")
    print("FINAL RESULTS (DUAL-HEAD)")
    print(f"{'='*70}")
    print(f"\n  Task A baseline: {acc_a_base:.1f}%")
    print(f"\n  {'Method':<12s} {'Task A':>8s} {'Task B':>8s} {'Forget':>8s} {'Total':>8s}")
    print(f"  {'-'*48}")

    for method in methods:
        r = results["methods"][method]
        total = r["task_a"] + r["task_b"]
        print(f"  {method.upper():<12s} {r['task_a']:>7.1f}% {r['task_b']:>7.1f}% "
              f"{r['forgetting']:>+7.1f}% {total:>7.1f}%")

    print(f"\n  KEY COMPARISON:")
    nccl_r = results["methods"]["nccl"]
    ewc_r = results["methods"]["ewc"]
    naive_r = results["methods"]["naive"]
    freeze_r = results["methods"]["freeze"]

    print(f"  NCCL vs Naive forgetting: {nccl_r['forgetting']:+.1f}% vs {naive_r['forgetting']:+.1f}%"
          f" (NCCL saves {naive_r['forgetting'] - nccl_r['forgetting']:.1f}%)")
    print(f"  NCCL vs EWC forgetting:   {nccl_r['forgetting']:+.1f}% vs {ewc_r['forgetting']:+.1f}%"
          f" (NCCL saves {ewc_r['forgetting'] - nccl_r['forgetting']:.1f}%)")
    print(f"  NCCL vs Freeze Task B:    {nccl_r['task_b']:.1f}% vs {freeze_r['task_b']:.1f}%"
          f" (NCCL gains {nccl_r['task_b'] - freeze_r['task_b']:+.1f}%)")

    print(f"\n  VERDICT (dual-head isolates backbone effect):")
    if nccl_r['forgetting'] < ewc_r['forgetting']:
        print(f"  >>> NCCL backbone protection > EWC backbone protection")
        if nccl_r['task_b'] > freeze_r['task_b']:
            print(f"  >>> NCCL learns Task B better than Freeze → null subspace has learning capacity")
        print(f"  >>> Null cone constraint is a viable continual learning method")
    else:
        print(f"  >>> EWC backbone protection >= NCCL backbone protection")
        print(f"  >>> Null cone constraint does not outperform EWC in dual-head setting")

    out_path = Path(args.output.replace(".json", f"_seed{args.seed}.json")
                    if hasattr(args, 'seeds') and len(args.seeds) > 1
                    else args.output)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")

    return results


class Config:
    device = "cuda"
    seed = 0
    seeds = [0, 1, 2]     # Multi-seed validation
    batch_size = 64
    lr = 0.01
    lr_b = 0.005
    epochs_a = 20
    epochs_b = 10
    sub_dim = 50
    n_subsets = 3
    update_every = 25
    ewc_lambda = 1000
    subset_size = 2500
    output = "nccl_v2_multiseed.json"

def main():
    args = Config()
    import sys
    if not any('jupyter' in a or 'ipykernel' in a or 'colab' in a for a in sys.argv):
        try:
            import argparse
            p = argparse.ArgumentParser()
            for k, v in vars(Config()).items():
                if isinstance(v, list):
                    p.add_argument(f"--{k}", type=int, nargs='+', default=v)
                else:
                    p.add_argument(f"--{k}", type=type(v), default=v)
            args = p.parse_args()
        except SystemExit:
            pass
    if args.sub_dim % 2 != 0:
        args.sub_dim -= 1

    # Multi-seed loop
    all_seed_results = []
    methods = ["naive", "ewc", "nccl", "freeze"]

    for seed in args.seeds:
        print(f"\n{'#'*70}")
        print(f"# SEED {seed}")
        print(f"{'#'*70}")
        args.seed = seed
        result = run_experiment(args)
        all_seed_results.append(result)

    if len(args.seeds) >= 2:
        # ============================================================
        # Multi-seed summary
        # ============================================================
        print(f"\n{'='*70}")
        print(f"MULTI-SEED SUMMARY ({len(args.seeds)} seeds: {args.seeds})")
        print(f"{'='*70}")

        baselines = [r["task_a_baseline"] for r in all_seed_results]
        print(f"\n  Task A baseline: {np.mean(baselines):.1f}% ± {np.std(baselines):.1f}%")

        print(f"\n  {'Method':<12s} {'Task A':>14s} {'Task B':>14s} {'Forget':>14s} {'Total':>14s}")
        print(f"  {'-'*70}")

        summary = {}
        for method in methods:
            task_a = [r["methods"][method]["task_a"] for r in all_seed_results]
            task_b = [r["methods"][method]["task_b"] for r in all_seed_results]
            forget = [r["methods"][method]["forgetting"] for r in all_seed_results]
            total = [a + b for a, b in zip(task_a, task_b)]

            summary[method] = {
                "task_a": task_a, "task_b": task_b,
                "forgetting": forget, "total": total,
                "task_a_mean": np.mean(task_a), "task_a_std": np.std(task_a),
                "task_b_mean": np.mean(task_b), "task_b_std": np.std(task_b),
                "forget_mean": np.mean(forget), "forget_std": np.std(forget),
                "total_mean": np.mean(total), "total_std": np.std(total),
            }

            print(f"  {method.upper():<12s} "
                  f"{np.mean(task_a):>5.1f}±{np.std(task_a):>4.1f}%  "
                  f"{np.mean(task_b):>5.1f}±{np.std(task_b):>4.1f}%  "
                  f"{np.mean(forget):>+5.1f}±{np.std(forget):>4.1f}%  "
                  f"{np.mean(total):>5.1f}±{np.std(total):>4.1f}%")

        # Per-seed detail
        print(f"\n  Per-seed NCCL forgetting: {summary['nccl']['forgetting']}")
        print(f"  Per-seed NCCL Task B:     {summary['nccl']['task_b']}")

        # Consistency check
        nccl_wins_naive = sum(1 for i in range(len(args.seeds))
                              if summary['nccl']['forgetting'][i] < summary['naive']['forgetting'][i])
        nccl_wins_ewc = sum(1 for i in range(len(args.seeds))
                            if summary['nccl']['forgetting'][i] < summary['ewc']['forgetting'][i])
        print(f"\n  NCCL < Naive forgetting: {nccl_wins_naive}/{len(args.seeds)} seeds")
        print(f"  NCCL < EWC forgetting:   {nccl_wins_ewc}/{len(args.seeds)} seeds")

        if nccl_wins_naive == len(args.seeds) and nccl_wins_ewc == len(args.seeds):
            print(f"  >>> NCCL CONSISTENTLY WINS across all {len(args.seeds)} seeds")
        elif nccl_wins_naive > len(args.seeds) // 2:
            print(f"  >>> NCCL wins majority but not all seeds")
        else:
            print(f"  >>> NCCL advantage is NOT consistent")

        # Save combined
        combined = {"seeds": args.seeds, "per_seed": all_seed_results, "summary": summary}
        with open(args.output, "w") as f:
            json.dump(combined, f, indent=2, default=str)
        print(f"\nSaved: {args.output}")

if __name__ == "__main__":
    main()
