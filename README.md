NullConeOptimizer: PyTorch Optimizer for Null Cone Exploitation
================================================================

Two modes:
  - 'nca': NCA-SGD (boost null directions, keep remainder) for training
  - 'nccl': NCCL (project onto null subspace only) for continual learning

Paper: "Neural Null Cones: Zero-Curvature Channels in Loss Landscapes
        from Symplectic Hessian Decomposition" (Li, 2025)
"""

import torch
from torch.optim import Optimizer
import numpy as np
import time


def _make_symplectic_J(n):
    I_n = np.eye(n)
    return np.block([[np.zeros((n, n)), I_n], [-I_n, np.zeros((n, n))]])


def _compute_sub_hessian(forward_fn, loss_fn, imgs, labs, param_indices, param, device):
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


def _find_null_directions(H, sub_dim, lam=1e-6):
    H_sym = 0.5 * (H + H.T)
    H_reg = H_sym + lam * np.eye(sub_dim)
    H_reg_norm = np.linalg.norm(H_reg)
    if H_reg_norm < 1e-30:
        return [], None
    H_reg_inv = np.linalg.inv(H_reg)
    J = _make_symplectic_J(sub_dim // 2)
    M = H_reg_inv @ J
    evals, evecs = np.linalg.eig(M)
    real_mask = np.abs(evals.imag) < 1e-8
    real_indices = np.where(real_mask)[0]
    null_vecs = []
    for idx in real_indices:
        e = evecs[:, idx].real
        e_norm = e / (np.linalg.norm(e) + 1e-30)
        if abs(e_norm @ H_reg @ e_norm) / (H_reg_norm + 1e-30) < 1e-6:
            null_vecs.append(e_norm)
    if len(null_vecs) < 2:
        return null_vecs, None
    V = np.column_stack(null_vecs)
    Q, _ = np.linalg.qr(V)
    return null_vecs, Q


def _intersect_null_subspaces(Q_list):
    if not Q_list or any(Q is None for Q in Q_list):
        return None
    if len(Q_list) == 1:
        return Q_list[0]
    P_sum = sum(Q @ Q.T for Q in Q_list)
    eigvals, eigvecs = np.linalg.eigh(P_sum)
    num_tasks = len(Q_list)
    mask = eigvals > (num_tasks - 0.5)
    if not np.any(mask):
        return None
    V = eigvecs[:, mask]
    Q_int, _ = np.linalg.qr(V)
    return Q_int


class _NullEngine:
    def __init__(self, target_params, forward_fns, loaders,
                 loss_fn, sub_dim=50, n_subsets=3, lam=1e-6, device='cuda'):
        self.target_params = target_params
        self.forward_fns = forward_fns
        self.loaders = loaders
        self.loader_iters = [iter(dl) for dl in loaders] if loaders else []
        self.loss_fn = loss_fn
        self.sub_dim = sub_dim
        self.n_subsets = n_subsets
        self.lam = lam
        self.device = device
        self.projections = {name: [] for name in target_params}
        self.stats = {"updates": 0, "null_layers": 0, "total_null_dirs": 0,
                      "intersection_dims": []}

    def _get_batch(self, task_idx):
        try:
            imgs, labs = next(self.loader_iters[task_idx])
        except StopIteration:
            self.loader_iters[task_idx] = iter(self.loaders[task_idx])
            imgs, labs = next(self.loader_iters[task_idx])
        return imgs.to(self.device), labs.to(self.device)

    def update(self, forward_fn=None, inputs=None, labels=None):
        self.stats["updates"] += 1
        if self.loaders:
            batches = [(self.forward_fns[i], *self._get_batch(i))
                       for i in range(len(self.loaders))]
        elif forward_fn is not None and inputs is not None:
            batches = [(forward_fn, inputs, labels)]
        else:
            return

        for name, param in self.target_params.items():
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
                for fwd_fn, imgs, labs in batches:
                    try:
                        H = _compute_sub_hessian(
                            fwd_fn, self.loss_fn, imgs, labs,
                            indices, param, self.device)
                        _, Q = _find_null_directions(H, dim, self.lam)
                    except Exception:
                        Q = None
                    if Q is not None:
                        Q_list.append(Q)
                if len(Q_list) == len(batches) and len(Q_list) > 0:
                    Q_final = Q_list[0] if len(Q_list) == 1 else _intersect_null_subspaces(Q_list)
                    if Q_final is not None and Q_final.shape[1] >= 1:
                        self.projections[name].append((indices, Q_final))
                        self.stats["null_layers"] += 1
                        self.stats["total_null_dirs"] += Q_final.shape[1]
                        if len(Q_list) > 1:
                            self.stats["intersection_dims"].append(Q_final.shape[1])
            param.requires_grad_(False)


class NullConeOptimizer(Optimizer):
    """
    SGD with null cone exploitation.

    Modes:
        'nca':  g = boost * g_null + g_rem   (training acceleration)
        'nccl': g = g_null                    (continual learning)

    Args:
        params: parameters to optimize
        lr, momentum, weight_decay: standard SGD args
        mode: 'nca' or 'nccl'
        boost: null boost factor (NCA only, default: 2.0)
        model: the model (required)
        loss_fn: loss function (required)
        forward_fn: forward function for Hessian (NCA mode, default: model)
        prev_forward_fns: list of forward fns for old tasks (NCCL mode)
        prev_loaders: list of dataloaders for old tasks (NCCL mode)
        target_param_names: substrings to filter which params get projection
                           (default: all). E.g. ['backbone']
        sub_dim: sub-Hessian size (default: 50, must be even)
        n_subsets: random subsets per layer per update (default: 3)
        update_every: steps between Hessian recomputation (default: 25)
        lam: H_reg regularization (default: 1e-6)
    """

    def __init__(self, params, lr=0.01, momentum=0.9, weight_decay=1e-4,
                 mode='nca', boost=2.0,
                 model=None, loss_fn=None, forward_fn=None,
                 prev_forward_fns=None, prev_loaders=None,
                 target_param_names=None,
                 sub_dim=50, n_subsets=3, update_every=25, lam=1e-6):

        if mode not in ('nca', 'nccl'):
            raise ValueError("mode must be 'nca' or 'nccl'")
        if sub_dim % 2 != 0:
            sub_dim -= 1
        if model is None or loss_fn is None:
            raise ValueError("model and loss_fn required")

        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super().__init__(params, defaults)

        self.mode = mode
        self.boost = boost
        self.model = model
        self.loss_fn = loss_fn
        self.forward_fn = forward_fn or model
        self.update_every = update_every
        self._step_count = 0

        device = next(model.parameters()).device
        target_params = {}
        for name, p in model.named_parameters():
            if target_param_names is None:
                target_params[name] = p
            elif any(sub in name for sub in target_param_names):
                target_params[name] = p

        if mode == 'nccl' and prev_forward_fns and prev_loaders:
            self.engine = _NullEngine(
                target_params, prev_forward_fns, prev_loaders,
                loss_fn, sub_dim, n_subsets, lam, device)
        else:
            self.engine = _NullEngine(
                target_params, [], [],
                loss_fn, sub_dim, n_subsets, lam, device)

    @property
    def stats(self):
        return self.engine.stats

    @torch.no_grad()
    def step(self, closure=None, inputs=None, labels=None):
        """
        Single optimization step.

        Args:
            closure: optional loss closure
            inputs: input batch for Hessian (required for NCA mode)
            labels: label batch for Hessian (required for NCA mode)
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._step_count += 1

        if self._step_count % self.update_every == 1 or self._step_count == 1:
            with torch.enable_grad():
                if self.mode == 'nca' and inputs is not None:
                    self.engine.update(forward_fn=self.forward_fn,
                                       inputs=inputs, labels=labels)
                elif self.mode == 'nccl' and self.engine.loaders:
                    self.engine.update()

        self._apply_null_projection()

        for group in self.param_groups:
            wd = group['weight_decay']
            mom = group['momentum']
            lr = group['lr']
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad
                if wd != 0:
                    d_p = d_p.add(p, alpha=wd)
                if mom != 0:
                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        buf = state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = state['momentum_buffer']
                        buf.mul_(mom).add_(d_p)
                    d_p = buf
                p.add_(d_p, alpha=-lr)

        return loss

    def _apply_null_projection(self):
        for name, param in self.engine.target_params.items():
            if param.grad is None:
                continue
            projections = self.engine.projections.get(name, [])
            if self.mode == 'nccl' and not projections:
                param.grad.zero_()
                continue
            if not projections:
                continue
            grad_flat = param.grad.reshape(-1)
            for indices, Q in projections:
                sub_grad = grad_flat[indices].detach().cpu().numpy().astype(np.float64)
                g_null = Q @ (Q.T @ sub_grad)
                if self.mode == 'nca':
                    g_rem = sub_grad - g_null
                    modified = self.boost * g_null + g_rem
                else:
                    modified = g_null
                grad_flat[indices] = torch.tensor(
                    modified, dtype=grad_flat.dtype, device=param.device)


# ============================================================
# Convenience constructors
# ============================================================

def nca_sgd(params, lr=0.01, momentum=0.9, weight_decay=1e-4,
            boost=2.0, model=None, loss_fn=None, forward_fn=None,
            sub_dim=50, n_subsets=3, update_every=25, **kwargs):
    """
    NCA-SGD for training acceleration.

        optimizer = nca_sgd(model.parameters(), lr=0.01,
                            model=model, loss_fn=loss_fn)
        for imgs, labs in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(imgs), labs)
            loss.backward()
            optimizer.step(inputs=imgs, labels=labs)
    """
    return NullConeOptimizer(
        params, lr=lr, momentum=momentum, weight_decay=weight_decay,
        mode='nca', boost=boost, model=model, loss_fn=loss_fn,
        forward_fn=forward_fn, sub_dim=sub_dim, n_subsets=n_subsets,
        update_every=update_every, **kwargs)


def nccl_sgd(params, lr=0.005, momentum=0.9, weight_decay=1e-4,
             model=None, loss_fn=None,
             prev_forward_fns=None, prev_loaders=None,
             target_param_names=None,
             sub_dim=50, n_subsets=3, update_every=25, **kwargs):
    """
    NCCL for continual learning.

        optimizer = nccl_sgd(model.parameters(), lr=0.005,
                             model=model, loss_fn=loss_fn,
                             prev_forward_fns=[model.forward_a],
                             prev_loaders=[task_a_loader],
                             target_param_names=['backbone'])
        for imgs, labs in task_b_loader:
            optimizer.zero_grad()
            loss = loss_fn(model.forward_b(imgs), labs)
            loss.backward()
            optimizer.step()
    """
    return NullConeOptimizer(
        params, lr=lr, momentum=momentum, weight_decay=weight_decay,
        mode='nccl', model=model, loss_fn=loss_fn,
        prev_forward_fns=prev_forward_fns, prev_loaders=prev_loaders,
        target_param_names=target_param_names,
        sub_dim=sub_dim, n_subsets=n_subsets, update_every=update_every,
        **kwargs)
