# null-cone-sgd

**Zero-curvature channels in neural network loss landscapes, exploited as a PyTorch optimizer.**

Neural network Hessians contain hidden null directions -- eigenvectors of H_reg^{-1}J (symplectic decomposition) along which gradient updates incur zero second-order penalty. This repo provides:

1. **`null_cone_optimizer.py`** -- Drop-in PyTorch optimizer with two modes:
   - **NCA-SGD**: Boost gradients along null directions for faster training (+5.7% on CIFAR-100)
   - **NCCL**: Project gradients onto null subspace for continual learning (1.6% forgetting after 3 tasks)

2. **Experiment scripts** reproducing all paper results

## Quick Start

### NCA-SGD (Training Acceleration)

```python
from null_cone_optimizer import nca_sgd

optimizer = nca_sgd(
    model.parameters(), lr=0.01,
    model=model, loss_fn=nn.CrossEntropyLoss(),
    boost=2.0, update_every=25,
)

for imgs, labs in loader:
    optimizer.zero_grad()
    loss = loss_fn(model(imgs), labs)
    loss.backward()
    optimizer.step(inputs=imgs, labels=labs)
```

### NCCL (Continual Learning)

```python
from null_cone_optimizer import nccl_sgd

optimizer = nccl_sgd(
    model.parameters(), lr=0.005,
    model=model, loss_fn=nn.CrossEntropyLoss(),
    prev_forward_fns=[model.forward_a],
    prev_loaders=[task_a_loader],
    target_param_names=['backbone'],
)

for imgs, labs in task_b_loader:
    optimizer.zero_grad()
    loss = loss_fn(model.forward_b(imgs), labs)
    loss.backward()
    optimizer.step()
```

## Key Results

### Training (ViT-Tiny, CIFAR-100 5K subset, 2 seeds)

| Method | Accuracy | vs SGD |
|--------|----------|--------|
| **NCA-SGD** | **65.8%** | **+5.7%** |
| SGD | 60.1% | -- |
| Random-SGD | 57.3% | -2.8% |

NCA-SGD converges 6.7x faster (60% accuracy at epoch 3 vs epoch 20).

### Continual Learning -- 2 Tasks (ViT-Tiny, dual-head, 3 seeds)

| Method | Task A | Task B | Forgetting | Total |
|--------|--------|--------|------------|-------|
| **NCCL** | **59.1+/-1.1%** | **58.1+/-0.3%** | **-2.0+/-1.3%** | **117.3+/-1.1%** |
| Naive | 53.3+/-1.1% | 61.0+/-0.7% | -7.9+/-2.2% | 114.3+/-0.6% |
| EWC | 52.4+/-1.6% | 60.9+/-0.5% | -8.8+/-0.8% | 113.3+/-1.7% |

### Continual Learning -- 3 Tasks (ViT-Tiny, tri-head, 3 seeds)

| Method | A(fin) | B(fin) | C(fin) | Forget A | Forget B | Total |
|--------|--------|--------|--------|----------|----------|-------|
| **NCCL** | **66.5+/-0.6%** | **58.8+/-1.2%** | **70.8+/-0.9%** | **-1.6+/-0.6%** | **-9.1+/-1.8%** | **196.1+/-0.3%** |
| Naive | 54.6+/-1.8% | 56.1+/-1.0% | 72.0+/-2.0% | -13.5+/-1.0% | -15.4+/-0.7% | 182.6+/-2.7% |
| EWC | 56.5+/-2.4% | 58.0+/-2.5% | 73.6+/-1.1% | -11.6+/-2.4% | -13.2+/-2.3% | 188.1+/-3.4% |

Null subspace intersection retains ~5 directions per layer after 2 tasks (from ~11). The geometry does not collapse.

## How It Works

1. Sample a 50x50 sub-Hessian H from a parameter layer
2. Regularize: H_reg = 0.5(H + H^T) + lambda * I
3. Build symplectic matrix: M = H_reg^{-1} J where J = [[0, I], [-I, 0]]
4. Find real eigenvectors of M satisfying |e^T H_reg e| / ||H_reg|| < 1e-6
5. These are **null cone directions** -- zero curvature cost
6. **NCA-SGD**: g_new = boost * g_null + g_rem (boost null, keep rest)
7. **NCCL**: g_new = g_null (only null component, protects old tasks)

For multiple previous tasks, NCCL uses the intersection of null subspaces.

## Repo Structure

```
null-cone-sgd/
+-- null_cone_optimizer.py        # PyTorch optimizer (NCA-SGD + NCCL)
+-- experiments/
|   +-- nca_training_v2.py        # Training experiment (Table 5)
|   +-- nccl_v2.py                # 2-task continual learning (Table 7)
|   +-- nccl_3task.py             # 3-task continual learning (Table 8)
+-- verification/
|   +-- null_cone_verify.py       # Basic null cone verification
|   +-- vit_null_cone_verify.py   # ViT/BERT/GPT-2 scaling
|   +-- null_dynamics.py          # SGD dynamics tracking (Table 6)
|   +-- null_step_test.py         # Pretrained model null test
|   +-- null_step_trained.py      # Trained model null test
+-- README.md
```

## Requirements

```
torch >= 2.0
transformers
torchvision
numpy
```

## Optimizer Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `mode` | `'nca'` | `'nca'` for training, `'nccl'` for continual learning |
| `boost` | `2.0` | Null direction amplification (NCA mode only) |
| `sub_dim` | `50` | Sub-Hessian dimension (must be even) |
| `n_subsets` | `3` | Random parameter subsets per layer |
| `update_every` | `25` | Steps between Hessian recomputation |
| `target_param_names` | `None` | Filter params by name substring (e.g. `['backbone']`) |
| `prev_forward_fns` | `None` | Forward functions of old tasks (NCCL mode) |
| `prev_loaders` | `None` | Dataloaders of old tasks (NCCL mode) |
| `lam` | `1e-6` | Regularization for H_reg |

## Verified At Scale

| Model | Parameters | Null Residual |
|-------|-----------|---------------|
| MLP-22 | 22 | 1e-15 |
| LeNet-5 | 61K | 1e-16 |
| GPT-2 | 124M | 1e-26 |
| ViT-Base | 86M | 1e-20 |
| BERT-Base | 110M | 1e-19 |

432 numerical tests, 393 pass. 39 failures are documented boundary cases.

## Citation

```bibtex
@article{li2026nullcones,
  title={Neural Null Cones: Zero-Curvature Channels in Loss Landscapes
         from Symplectic Hessian Decomposition},
  author={Li, Y.Y.N.},
  year={2026}
}
```

## License

MIT
