"""
NCCL 3-Task Continual Learning Experiment
==========================================
Extension of nccl_v2.py to 3 sequential tasks with tri-head architecture.

Tasks: CIFAR-100 split into 3:
  Task A: classes 0-33   (34 classes, head_a)
  Task B: classes 34-66  (33 classes, head_b)
  Task C: classes 67-99  (33 classes, head_c)

Key design for NCCL:
  Phase 2 (learn B): project onto null(H_A)
  Phase 3 (learn C): project onto null(H_A) ∩ null(H_B)
  This tests whether null subspace intersection shrinks too fast.

Methods: Naive, EWC, NCCL, Freeze
Multi-seed: 3 seeds by default.
"""

import numpy as np
import copy
import time
import json
from pathlib import Path


# ============================================================
# 1. Sub-Hessian computation (same as v2)
# ============================================================

def compute_sub_hessian(forward_fn, loss_fn, imgs, labs, param_indices, param, device):
    import torch
    sub_dim = len(param_indices)
    H = np.zeros((sub_dim, sub_dim), dtype=np.float64)
    out = forward_fn(imgs)
    loss = loss_fn(out, labs)
    g1 = torch.autograd.grad(loss, param, create_graph=True)[0]
    g1_flat = g1.reshape(-1)
    for i in range(sub_dim):
        idx = param_indices[i]
        g2 = torch.autograd.grad(g1_flat[idx], param, retain_graph=True)[0]
        g2_flat = g2.reshape(-1)
        for j in range(sub_dim):
            H[i, j] = g2_flat[param_indices[j]].item()
    return H


def make_symplectic_J(n):
    I_n = np.eye(n)
    return np.block([[np.zeros((n, n)), I_n], [-I_n, np.zeros((n, n))]])


def find_null_directions(H, sub_dim, lam=1e-6):
    H_sym = 0.5 * (H + H.T)
    H_reg = H_sym + lam * np.eye(sub_dim)
    H_reg_norm = np.linalg.norm(H_reg)
    if H_reg_norm < 1e-30:
        return [], None
    H_reg_inv = np.linalg.inv(H_reg)
    J = make_symplectic_J(sub_dim // 2)
    M = H_reg_inv @ J
    evals_M, evecs_M = np.linalg.eig(M)
    real_mask = np.abs(evals_M.imag) < 1e-8
    real_indices = np.where(real_mask)[0]
    null_vecs = []
    for idx in real_indices:
        e = evecs_M[:, idx].real
        e_norm = e / (np.linalg.norm(e) + 1e-30)
        if abs(e_norm @ H_reg @ e_norm) / (H_reg_norm + 1e-30) < 1e-6:
            null_vecs.append(e_norm)
    if len(null_vecs) < 2:
        return null_vecs, None
    V = np.column_stack(null_vecs)
    Q, R = np.linalg.qr(V)
    return null_vecs, Q


# ============================================================
# 2. Tri-Head ViT Architecture
# ============================================================

class TriHeadViT:
    """
    ViT backbone + three independent classifier heads.
    head_a: Linear(hidden, 34) for classes 0-33
    head_b: Linear(hidden, 33) for classes 34-66
    head_c: Linear(hidden, 33) for classes 67-99
    """
    def __init__(self, base_model, device):
        import torch
        import torch.nn as nn
        self.device = device
        self.backbone = base_model.vit
        hidden = base_model.config.hidden_size  # 192

        # head_a from pretrained rows 0-33
        self.head_a = nn.Linear(hidden, 34).to(device)
        with torch.no_grad():
            self.head_a.weight.copy_(base_model.classifier.weight[:34])
            self.head_a.bias.copy_(base_model.classifier.bias[:34])

        # head_b and head_c freshly initialized
        self.head_b = nn.Linear(hidden, 33).to(device)
        nn.init.xavier_uniform_(self.head_b.weight)
        nn.init.zeros_(self.head_b.bias)

        self.head_c = nn.Linear(hidden, 33).to(device)
        nn.init.xavier_uniform_(self.head_c.weight)
        nn.init.zeros_(self.head_c.bias)

        self.backbone.to(device)

    def _features(self, x):
        return self.backbone(x).last_hidden_state[:, 0]

    def forward_a(self, x): return self.head_a(self._features(x))
    def forward_b(self, x): return self.head_b(self._features(x))
    def forward_c(self, x): return self.head_c(self._features(x))

    def freeze_head(self, head_name):
        head = getattr(self, f'head_{head_name}')
        for p in head.parameters():
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
        self.head_c.train()

    def eval(self):
        self.backbone.eval()
        self.head_a.eval()
        self.head_b.eval()
        self.head_c.eval()

    def backbone_state(self):
        return copy.deepcopy(self.backbone.state_dict())

    def head_state(self, name):
        return copy.deepcopy(getattr(self, f'head_{name}').state_dict())

    def load_backbone(self, sd):
        self.backbone.load_state_dict(sd)

    def load_head(self, name, sd):
        getattr(self, f'head_{name}').load_state_dict(sd)


# ============================================================
# 3. Data: 3-way class split
# ============================================================

def make_3task_loaders(subset_size, batch_size):
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

    ds_train = torchvision.datasets.CIFAR100('./data', True, download=True, transform=tf_train)
    ds_val = torchvision.datasets.CIFAR100('./data', False, download=True, transform=tf_val)

    class RemappedSubset(torch.utils.data.Dataset):
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
            return img, label - self.offset

    classes_a = set(range(0, 34))
    classes_b = set(range(34, 67))
    classes_c = set(range(67, 100))

    dl = lambda ds, shuffle=True: torch.utils.data.DataLoader(
        ds, batch_size, shuffle=shuffle, num_workers=2, pin_memory=True)

    loaders = {}
    for name, cls_set in [('a', classes_a), ('b', classes_b), ('c', classes_c)]:
        train = RemappedSubset(ds_train, cls_set, subset_size)
        val = RemappedSubset(ds_val, cls_set, 0)
        loaders[f'train_{name}'] = dl(train)
        loaders[f'val_{name}'] = dl(val, False)
        print(f"  Task {name.upper()}: {len(train)} train, {len(val)} val "
              f"(classes {min(cls_set)}-{max(cls_set)} -> labels 0-{len(cls_set)-1})")

    return loaders


# ============================================================
# 4. NCCL Engine (multi-task null intersection)
# ============================================================

class NCCLEngine3Task:
    """
    For Phase 2: null subspace of Task A Hessian
    For Phase 3: intersection of null(H_A) and null(H_B)
    """
    def __init__(self, model, loaders_for_hessian, forward_fns, loss_fn,
                 sub_dim=50, n_subsets=3, update_every=25, device='cuda'):
        import torch
        self.model = model
        self.loaders = loaders_for_hessian  # list of dataloaders
        self.loader_iters = [iter(dl) for dl in loaders_for_hessian]
        self.forward_fns = forward_fns       # list of forward functions
        self.loss_fn = loss_fn
        self.sub_dim = sub_dim
        self.n_subsets = n_subsets
        self.update_every = update_every
        self.device = device

        self.target_layers = self._get_target_layers()
        self.projections = {name: [] for name in self.target_layers}
        self.stats = {"updates": 0, "null_layers": 0, "total_null_dirs": 0,
                      "null_intersection_dims": []}

    def _get_target_layers(self):
        targets = {}
        for li in range(12):
            block = self.model.backbone.encoder.layer[li]
            targets[f"attn_L{li}_q"] = block.attention.attention.query.weight
            targets[f"attn_L{li}_o"] = block.attention.output.dense.weight
            targets[f"mlp_L{li}_f1"] = block.intermediate.dense.weight
            targets[f"mlp_L{li}_f2"] = block.output.dense.weight
        return targets

    def _get_batch(self, task_idx):
        import torch
        try:
            imgs, labs = next(self.loader_iters[task_idx])
        except StopIteration:
            self.loader_iters[task_idx] = iter(self.loaders[task_idx])
            imgs, labs = next(self.loader_iters[task_idx])
        return imgs.to(self.device), labs.to(self.device)

    def _find_null_for_task(self, forward_fn, imgs, labs, param, indices, dim):
        try:
            H = compute_sub_hessian(forward_fn, self.loss_fn, imgs, labs,
                                    indices, param, self.device)
            null_vecs, Q = find_null_directions(H, dim)
            return null_vecs, Q
        except Exception:
            return [], None

    def _intersect_null_subspaces(self, Q_list):
        """
        Find intersection of multiple null subspaces.
        Sum of projection matrices: vectors in intersection have eigenvalue = num_tasks.
        """
        if not Q_list or any(Q is None for Q in Q_list):
            return None
        if len(Q_list) == 1:
            return Q_list[0]

        n = Q_list[0].shape[0]
        P_sum = sum(Q @ Q.T for Q in Q_list)
        eigvals, eigvecs = np.linalg.eigh(P_sum)

        num_tasks = len(Q_list)
        tol = 0.5
        mask = eigvals > (num_tasks - tol)
        if not np.any(mask):
            return None

        V = eigvecs[:, mask]
        Q_int, _ = np.linalg.qr(V)
        return Q_int

    def update_null_directions(self):
        import torch
        self.stats["updates"] += 1
        batches = [self._get_batch(i) for i in range(len(self.loaders))]

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

                Q_list = []
                for t_idx, (imgs, labs) in enumerate(batches):
                    null_vecs, Q = self._find_null_for_task(
                        self.forward_fns[t_idx], imgs, labs, param, indices, dim)
                    if Q is not None:
                        Q_list.append(Q)

                if len(Q_list) == len(self.loaders):
                    Q_int = self._intersect_null_subspaces(Q_list)
                    if Q_int is not None and Q_int.shape[1] >= 1:
                        self.projections[name].append((indices, Q_int))
                        self.stats["null_layers"] += 1
                        self.stats["total_null_dirs"] += Q_int.shape[1]
                        self.stats["null_intersection_dims"].append(Q_int.shape[1])

            param.requires_grad_(False)

    def project_gradients(self):
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
# 5. EWC Engine (multi-task Fisher accumulation)
# ============================================================

class EWCEngine3Task:
    def __init__(self, model, loaders, forward_fns, loss_fn,
                 ewc_lambda=1000, n_samples=500, device='cuda'):
        import torch
        self.model = model
        self.ewc_lambda = ewc_lambda
        self.device = device
        self.fisher = {}
        self.theta_star = {}

        for name, param in model.backbone.named_parameters():
            self.theta_star[name] = param.data.clone()

        for loader, forward_fn in zip(loaders, forward_fns):
            self._accumulate_fisher(loader, forward_fn, loss_fn, n_samples)

    def _accumulate_fisher(self, loader, forward_fn, loss_fn, n_samples):
        import torch
        self.model.eval()
        count = 0
        for imgs, labs in loader:
            if count >= n_samples:
                break
            imgs, labs = imgs.to(self.device), labs.to(self.device)
            self.model.backbone.zero_grad()
            out = forward_fn(imgs)
            loss = loss_fn(out, labs)
            loss.backward()
            for name, param in self.model.backbone.named_parameters():
                if param.grad is not None:
                    if name not in self.fisher:
                        self.fisher[name] = torch.zeros_like(param)
                    self.fisher[name] += param.grad.detach() ** 2
            count += imgs.size(0)
        for name in self.fisher:
            self.fisher[name] /= max(count, 1)

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

def evaluate(model, loader, loss_fn, device, task='a', label=""):
    import torch
    model.eval()
    correct = total = 0
    total_loss = 0.0
    forward_fn = getattr(model, f'forward_{task}')
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

def train_task(model, task_name, dl_train, dl_val, loss_fn, device, epochs, lr):
    import torch
    print(f"\n  Training Task {task_name.upper()} (backbone + head_{task_name})...")
    forward_fn = getattr(model, f'forward_{task_name}')
    head = getattr(model, f'head_{task_name}')
    params = list(model.backbone.parameters()) + list(head.parameters())
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=1e-4)

    for ep in range(epochs):
        model.train()
        rloss = nb = 0
        for imgs, labs in dl_train:
            imgs, labs = imgs.to(device), labs.to(device)
            optimizer.zero_grad()
            loss = loss_fn(forward_fn(imgs), labs)
            loss.backward()
            optimizer.step()
            rloss += loss.item()
            nb += 1
        if (ep + 1) % 5 == 0 or ep == 0 or ep == epochs - 1:
            acc, _ = evaluate(model, dl_val, loss_fn, device, task_name)
            print(f"    Ep {ep+1}/{epochs}: train={rloss/nb:.4f} val_{task_name.upper()}={acc:.1f}%")


def train_new_task(model, method, new_task, dl_train_new,
                   dl_vals, loss_fn, device, epochs, lr,
                   ewc_engine=None, nccl_engine=None):
    import torch
    forward_fn = getattr(model, f'forward_{new_task}')
    head = getattr(model, f'head_{new_task}')

    if method == "freeze":
        model.freeze_backbone()
        params = list(head.parameters())
    else:
        params = list(model.backbone.parameters()) + list(head.parameters())

    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=1e-4)
    step = 0

    for ep in range(epochs):
        model.train()
        rloss = nb = 0
        for imgs, labs in dl_train_new:
            imgs, labs = imgs.to(device), labs.to(device)
            optimizer.zero_grad()
            loss = loss_fn(forward_fn(imgs), labs)
            if ewc_engine and method == "ewc":
                loss = loss + ewc_engine.penalty()
            loss.backward()

            if method == "nccl" and nccl_engine:
                if step % nccl_engine.update_every == 0:
                    nccl_engine.update_null_directions()
                nccl_engine.project_gradients()

            optimizer.step()
            rloss += loss.item()
            nb += 1
            step += 1

        extra = ""
        if nccl_engine and method == "nccl":
            extra = (f" | upd={nccl_engine.stats['updates']} "
                     f"null={nccl_engine.stats['null_layers']}")
            if nccl_engine.stats['null_intersection_dims']:
                avg_dim = np.mean(nccl_engine.stats['null_intersection_dims'])
                extra += f" avg_int_dim={avg_dim:.1f}"

        if (ep + 1) % 5 == 0 or ep == 0 or ep == epochs - 1:
            accs = {}
            for t, dl_v in dl_vals.items():
                acc, _ = evaluate(model, dl_v, loss_fn, device, t)
                accs[t] = acc
            acc_str = " ".join(f"{t.upper()}={accs[t]:.1f}%" for t in sorted(accs))
            print(f"    Ep {ep+1}/{epochs}: loss={rloss/nb:.4f} {acc_str}{extra}")

    if method == "freeze":
        model.unfreeze_backbone()


# ============================================================
# 8. Main Experiment
# ============================================================

def run_experiment(args):
    import torch
    import torch.nn as nn
    from transformers import ViTForImageClassification

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Architecture: Tri-head ViT-Tiny (3 tasks)")
    print(f"Epochs: A={args.epochs_a}, B={args.epochs_b}, C={args.epochs_c}")
    print(f"EWC lambda: {args.ewc_lambda}")
    print(f"NCCL: update_every={args.update_every}, n_subsets={args.n_subsets}")
    print("=" * 70)

    if hasattr(torch.backends, 'cuda'):
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)

    print("\nLoading CIFAR-100 (3-way class split)...")
    loaders = make_3task_loaders(args.subset_size, args.batch_size)
    loss_fn = nn.CrossEntropyLoss()

    def make_model():
        torch.manual_seed(args.seed)
        base = ViTForImageClassification.from_pretrained(
            'WinKawaks/vit-tiny-patch16-224',
            num_labels=100, ignore_mismatched_sizes=True,
            attn_implementation="eager")
        return TriHeadViT(base, device)

    # ============================================================
    # Phase 1: Train on Task A
    # ============================================================
    print("\n" + "=" * 70)
    print("PHASE 1: TASK A TRAINING (backbone + head_a)")
    print("=" * 70)

    model_base = make_model()
    train_task(model_base, 'a', loaders['train_a'], loaders['val_a'],
               loss_fn, device, args.epochs_a, args.lr)
    acc_a_base, _ = evaluate(model_base, loaders['val_a'], loss_fn, device, 'a',
                              "Task A baseline")

    backbone_after_a = model_base.backbone_state()
    head_a_state = model_base.head_state('a')

    # ============================================================
    # Phase 2: Learn Task B (head_a frozen)
    # ============================================================
    print("\n" + "=" * 70)
    print("PHASE 2: TASK B LEARNING (head_a FROZEN)")
    print("=" * 70)

    methods = ["naive", "ewc", "nccl", "freeze"]
    phase2_states = {}

    for method in methods:
        print(f"\n--- {method.upper()} ---")
        m = make_model()
        m.load_backbone(backbone_after_a)
        m.load_head('a', head_a_state)
        m.freeze_head('a')

        ewc_engine = nccl_engine = None
        if method == "ewc":
            ewc_engine = EWCEngine3Task(
                m, [loaders['train_a']], [m.forward_a], loss_fn,
                ewc_lambda=args.ewc_lambda, device=device)
        if method == "nccl":
            # Phase 2: single-task null (same as v2)
            nccl_engine = NCCLEngine3Task(
                m, [loaders['train_a']], [m.forward_a], loss_fn,
                sub_dim=args.sub_dim, n_subsets=args.n_subsets,
                update_every=args.update_every, device=device)

        dl_vals = {'a': loaders['val_a'], 'b': loaders['val_b']}
        train_new_task(m, method, 'b', loaders['train_b'],
                       dl_vals, loss_fn, device, args.epochs_b,
                       args.lr_b, ewc_engine, nccl_engine)

        acc_a_p2, _ = evaluate(m, loaders['val_a'], loss_fn, device, 'a', "Task A after P2")
        acc_b_p2, _ = evaluate(m, loaders['val_b'], loss_fn, device, 'b', "Task B after P2")

        phase2_states[method] = {
            'backbone': m.backbone_state(),
            'head_b': m.head_state('b'),
            'acc_a_p2': acc_a_p2,
            'acc_b_p2': acc_b_p2,
            'nccl_p2_stats': nccl_engine.stats if nccl_engine else None,
        }

    # ============================================================
    # Phase 3: Learn Task C (head_a, head_b frozen)
    # ============================================================
    print("\n" + "=" * 70)
    print("PHASE 3: TASK C LEARNING (head_a, head_b FROZEN)")
    print("=" * 70)

    results = {
        "task_a_baseline": acc_a_base,
        "methods": {}
    }

    for method in methods:
        print(f"\n--- {method.upper()} ---")
        m = make_model()
        m.load_backbone(phase2_states[method]['backbone'])
        m.load_head('a', head_a_state)
        m.load_head('b', phase2_states[method]['head_b'])
        m.freeze_head('a')
        m.freeze_head('b')

        acc_b_base = phase2_states[method]['acc_b_p2']

        ewc_engine = nccl_engine = None
        if method == "ewc":
            ewc_engine = EWCEngine3Task(
                m, [loaders['train_a'], loaders['train_b']],
                [m.forward_a, m.forward_b], loss_fn,
                ewc_lambda=args.ewc_lambda, device=device)
        if method == "nccl":
            # Phase 3: null intersection of Task A AND Task B
            nccl_engine = NCCLEngine3Task(
                m, [loaders['train_a'], loaders['train_b']],
                [m.forward_a, m.forward_b], loss_fn,
                sub_dim=args.sub_dim, n_subsets=args.n_subsets,
                update_every=args.update_every, device=device)

        dl_vals = {'a': loaders['val_a'], 'b': loaders['val_b'], 'c': loaders['val_c']}
        train_new_task(m, method, 'c', loaders['train_c'],
                       dl_vals, loss_fn, device, args.epochs_c,
                       args.lr_b, ewc_engine, nccl_engine)

        acc_a_final, _ = evaluate(m, loaders['val_a'], loss_fn, device, 'a', "Task A final")
        acc_b_final, _ = evaluate(m, loaders['val_b'], loss_fn, device, 'b', "Task B final")
        acc_c_final, _ = evaluate(m, loaders['val_c'], loss_fn, device, 'c', "Task C final")

        forget_a = acc_a_base - acc_a_final
        forget_b = acc_b_base - acc_b_final

        results["methods"][method] = {
            "acc_a_after_p2": phase2_states[method]['acc_a_p2'],
            "acc_b_after_p2": phase2_states[method]['acc_b_p2'],
            "acc_a_final": acc_a_final,
            "acc_b_final": acc_b_final,
            "acc_c_final": acc_c_final,
            "forget_a_total": forget_a,
            "forget_b_p3": forget_b,
            "total": acc_a_final + acc_b_final + acc_c_final,
        }
        if nccl_engine:
            results["methods"][method]["nccl_p3_stats"] = nccl_engine.stats
        if phase2_states[method]['nccl_p2_stats']:
            results["methods"][method]["nccl_p2_stats"] = phase2_states[method]['nccl_p2_stats']

    # ============================================================
    # Summary
    # ============================================================
    print(f"\n{'='*70}")
    print("FINAL RESULTS (TRI-HEAD, 3 TASKS)")
    print(f"{'='*70}")
    print(f"\n  Task A baseline: {acc_a_base:.1f}%")

    print(f"\n  {'Method':<10s} {'A(P2)':>7s} {'B(P2)':>7s} {'A(fin)':>7s} "
          f"{'B(fin)':>7s} {'C(fin)':>7s} {'fA':>6s} {'fB':>6s} {'Total':>7s}")
    print(f"  {'-'*68}")

    for method in methods:
        r = results["methods"][method]
        print(f"  {method.upper():<10s} {r['acc_a_after_p2']:>6.1f}% {r['acc_b_after_p2']:>6.1f}% "
              f"{r['acc_a_final']:>6.1f}% {r['acc_b_final']:>6.1f}% {r['acc_c_final']:>6.1f}% "
              f"{r['forget_a_total']:>+5.1f}% {r['forget_b_p3']:>+5.1f}% {r['total']:>6.1f}%")

    # NCCL intersection diagnostics
    for method in ["nccl"]:
        r = results["methods"].get(method, {})
        p3_stats = r.get("nccl_p3_stats", {})
        dims = p3_stats.get("null_intersection_dims", [])
        if dims:
            print(f"\n  NCCL Phase 3 null∩ dims: mean={np.mean(dims):.1f} "
                  f"min={min(dims)} max={max(dims)} ({len(dims)} subsets)")
            print(f"  NCCL Phase 3 updates={p3_stats['updates']} "
                  f"null_layers={p3_stats['null_layers']}")
        else:
            print(f"\n  ⚠️ NCCL Phase 3: NO null intersection found")

    print(f"\n  KEY: fA = Task A forgetting (baseline→final), fB = Task B forgetting (P2→final)")

    out_path = Path(args.output.replace(".json", f"_seed{args.seed}.json")
                    if hasattr(args, 'seeds') and len(args.seeds) > 1
                    else args.output)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")

    return results


# ============================================================
# 9. Config and Main
# ============================================================

class Config:
    device = "cuda"
    seed = 0
    seeds = [0, 1, 2]
    batch_size = 64
    lr = 0.01
    lr_b = 0.005
    epochs_a = 20
    epochs_b = 10
    epochs_c = 10
    sub_dim = 50
    n_subsets = 3
    update_every = 25
    ewc_lambda = 1000
    subset_size = 2500
    output = "nccl_3task.json"


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
        print(f"\n{'='*70}")
        print(f"MULTI-SEED SUMMARY ({len(args.seeds)} seeds: {args.seeds})")
        print(f"{'='*70}")

        baselines = [r["task_a_baseline"] for r in all_seed_results]
        print(f"\n  Task A baseline: {np.mean(baselines):.1f} ± {np.std(baselines):.1f}%")

        print(f"\n  {'Method':<10s} {'A(fin)':>14s} {'B(fin)':>14s} {'C(fin)':>14s} "
              f"{'fA':>14s} {'fB':>14s} {'Total':>14s}")
        print(f"  {'-'*90}")

        for method in methods:
            a_fin = [r["methods"][method]["acc_a_final"] for r in all_seed_results]
            b_fin = [r["methods"][method]["acc_b_final"] for r in all_seed_results]
            c_fin = [r["methods"][method]["acc_c_final"] for r in all_seed_results]
            fa = [r["methods"][method]["forget_a_total"] for r in all_seed_results]
            fb = [r["methods"][method]["forget_b_p3"] for r in all_seed_results]
            total = [r["methods"][method]["total"] for r in all_seed_results]

            print(f"  {method.upper():<10s} "
                  f"{np.mean(a_fin):>5.1f}±{np.std(a_fin):>4.1f}%  "
                  f"{np.mean(b_fin):>5.1f}±{np.std(b_fin):>4.1f}%  "
                  f"{np.mean(c_fin):>5.1f}±{np.std(c_fin):>4.1f}%  "
                  f"{np.mean(fa):>+5.1f}±{np.std(fa):>4.1f}%  "
                  f"{np.mean(fb):>+5.1f}±{np.std(fb):>4.1f}%  "
                  f"{np.mean(total):>5.1f}±{np.std(total):>4.1f}%")

        # NCCL wins check
        nccl_fa = [r["methods"]["nccl"]["forget_a_total"] for r in all_seed_results]
        naive_fa = [r["methods"]["naive"]["forget_a_total"] for r in all_seed_results]
        ewc_fa = [r["methods"]["ewc"]["forget_a_total"] for r in all_seed_results]
        wins_naive = sum(1 for n, na in zip(nccl_fa, naive_fa) if n < na)
        wins_ewc = sum(1 for n, e in zip(nccl_fa, ewc_fa) if n < e)
        print(f"\n  NCCL < Naive on fA: {wins_naive}/{len(args.seeds)} seeds")
        print(f"  NCCL < EWC on fA:   {wins_ewc}/{len(args.seeds)} seeds")

        # Null intersection diagnostics
        for i, r in enumerate(all_seed_results):
            p3_stats = r["methods"].get("nccl", {}).get("nccl_p3_stats", {})
            dims = p3_stats.get("null_intersection_dims", [])
            if dims:
                print(f"  Seed {args.seeds[i]} null∩: mean={np.mean(dims):.1f} "
                      f"min={min(dims)} max={max(dims)}")
            else:
                print(f"  Seed {args.seeds[i]} null∩: EMPTY")

        combined = {"seeds": args.seeds, "per_seed": all_seed_results}
        with open(args.output, "w") as f:
            json.dump(combined, f, indent=2, default=str)
        print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
