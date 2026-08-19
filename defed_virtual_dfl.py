"""
DeFed-GMM-DaDiL Virtual Sample DFL Training
===========================================
Downstream pipeline that consumes the defed_dictionary.npz produced by
defed_gmm_dadil.py:

    1. Load defed_dictionary.npz:
       - Global atom dictionary (medoid's K atoms, saved as atom{k}_*).
       - Each client's private alpha (barycentric coordinates), saved as
         clientN_alpha.
    2. Per client, reconstruct its GMM as the free-support Wasserstein
       barycenter of the GLOBAL atoms weighted by THAT client's alpha.
       This gives a personalized 256-d feature-space GMM (C=4 components,
       one per class) approximating the original domain GMM.
    3. Sample virtual 256-d features from each client's reconstructed GMM:
       component ~ Categorical(pi), feature ~ N(mu[c], diag(var[c])),
       label = argmax(V[c]).  Sample count is matched to the real client
       size (read from train_set/clientN/data.npy) so the virtual train
       set is the same scale as the real one.
    4. Train a small MLP (256 -> 128 -> num_classes) on the virtual
       features in a dynamic-topology DFL setting, mirroring
       DeFed_Dynamic.py: 8 clients, 2 random peers per round, synchronous
       gossip averaging with self_weight=1/3.
    5. Extract 256-d features from the held-out test_set signals via the
       existing federated-trained backbone
       (backbone_results/dfl_cnn_backbone.pth).
    6. Evaluate each client's MLP on the test features.  Pick the medoid
       client as the representative, report per-client accuracies, mean
       accuracy, best-client confusion matrix.

Usage:
    python defed_virtual_dfl.py
    python defed_virtual_dfl.py --rounds 50 --local-epochs 3 --device cuda
    python defed_virtual_dfl.py --virtual-samples-per-client 500
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Import utilities from defed_gmm_dadil.py (same directory)
from defed_gmm_dadil import (
    AtomGMM,
    free_support_barycenter,
    set_seed,
    get_dtype,
)
# model.py is in the same directory (and also in the parent directory)
from model import load_feature_extractor


# ============================================================================
# Defaults
# ============================================================================

DEFAULT_CONFIG = {
    # Inputs / outputs
    "defed_dict_path": str(Path(__file__).parent / "defed_gmm_dadil_results" / "defed_dictionary.npz"),
    "backbone_path":   str(Path(__file__).parent / "backbone_results" / "dfl_cnn_backbone.pth"),
    "train_data_root": r"E:\FL\Data\Data_iid_nomalicious\train_set",   # only used to match sample sizes
    "test_data_root":  r"E:\FL\Data\Data_iid_nomalicious\test_set",
    "save_dir":        str(Path(__file__).parent / "defed_virtual_dfl_results"),

    # Dictionary structure (must match defed_gmm_dadil_results/config.json)
    "K": 3,
    "C": 4,
    "num_classes": 4,
    "feature_dim": 256,

    # Barycenter reconstruction (mirrors defed_gmm_dadil defaults)
    "barycenter_iters": 8,
    "beta_class": 1.0,
    "var_floor": 1e-3,
    "eps": 1e-9,

    # Virtual feature sampling
    "virtual_samples_per_client": None,   # None = match real client size
    "virtual_seed": 42,

    # MLP
    "mlp_hidden_dim": 128,
    "mlp_dropout": 0.3,

    # DFL training (mirrors DeFed_Dynamic.py defaults)
    "num_clients": 8,
    "rounds": 30,
    "local_epochs": 2,
    "batch_size": 64,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "neighbors_per_client": 2,
    "self_weight": 1.0 / 3.0,
    "keep_bn_local": False,

    # Misc
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "dtype": "float64",
    "seed": 42,
    "num_workers": 0,
}

FAULT_NAMES = {0: "Normal", 1: "Inner", 2: "Roller", 3: "Outer"}


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                     labels: Sequence[int]) -> np.ndarray:
    """Small NumPy equivalent of sklearn.metrics.confusion_matrix."""
    index = {int(label): i for i, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for true_label, pred_label in zip(y_true, y_pred):
        if int(true_label) in index and int(pred_label) in index:
            matrix[index[int(true_label)], index[int(pred_label)]] += 1
    return matrix


# ============================================================================
# MLP classifier on 256-d features
# ============================================================================
class BearingMLP(nn.Module):
    """Simple MLP for 256-d feature -> num_classes classification.

    Architecture mirrors a stripped-down BearingCNN head:
        Linear(D -> H) -> BN -> ReLU -> Dropout -> Linear(H -> num_classes)

    Input : (B, feature_dim)  float
    Output: (B, num_classes)  logits
    """

    def __init__(self, feature_dim: int = 256, hidden_dim: int = 128,
                 num_classes: int = 4, dropout: float = 0.3):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        self.fc1 = nn.Linear(feature_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, feature_dim)
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        return self.fc2(x)


# ============================================================================
# Load defed_dictionary.npz
# ============================================================================
def load_dictionary_metadata(npz_path: str) -> Dict:
    """Read and validate the structural contract saved by dictionary learning."""
    with np.load(npz_path, allow_pickle=False) as d:
        required = ("K", "C", "num_classes", "feature_dim", "client_names")
        missing = [key for key in required if key not in d]
        if missing:
            raise ValueError(f"dictionary metadata is missing: {missing}")
        return {
            "K": int(d["K"].item()),
            "C": int(d["C"].item()),
            "num_classes": int(d["num_classes"].item()),
            "feature_dim": int(d["feature_dim"].item()),
            "client_names": [str(name) for name in d["client_names"].tolist()],
        }


def load_global_atoms(npz_path: str, K: int,
                      device: torch.device, dtype: torch.dtype) -> List[AtomGMM]:
    """Load the medoid's K atoms (the 'global' atom dictionary).

    These are the consensus atoms at the top level of defed_dictionary.npz
    (atom0_pi/mu/var/V, atom1_*, atom2_*).  All clients share this dictionary;
    each client personalizes via its own alpha.
    """
    atoms: List[AtomGMM] = []
    with np.load(npz_path, allow_pickle=False) as d:
        for k in range(K):
            atoms.append(AtomGMM.from_array(
                pi=d[f"atom{k}_pi"],
                mu=d[f"atom{k}_mu"],
                var=d[f"atom{k}_var"],
                V=d[f"atom{k}_V"],
                device=device, dtype=dtype, requires_grad=False,
            ))
    return atoms


def load_client_alphas(npz_path: str,
                       client_names: Sequence[str]) -> Dict[str, np.ndarray]:
    """Load each client's private alpha (barycentric coordinates over the K atoms)."""
    with np.load(npz_path, allow_pickle=False) as d:
        return {
            name: d[f"{name}_alpha"].astype(np.float64)
            for name in client_names
        }


# ============================================================================
# Reconstruct per-client GMM via free-support barycenter
# ============================================================================
def reconstruct_client_gmm(global_atoms: Sequence[AtomGMM],
                           alpha: np.ndarray,
                           config: Dict,
                           device: torch.device,
                           dtype: torch.dtype) -> AtomGMM:
    """Reconstruct one client's GMM as the free-support Wasserstein barycenter
    of the global atoms weighted by the client's private alpha.

    This is the DeFed-GMM-DaDiL 'decode' step: alpha (private) + global atoms
    (shared) -> personalized GMM B(alpha, P_global) in 256-d feature space.

    Returns:
        AtomGMM with C=4 components (one per class), in 256-d feature space.
    """
    if alpha.ndim != 1 or len(alpha) != len(global_atoms):
        raise ValueError(
            f"alpha length ({len(alpha)}) must match global atom count "
            f"({len(global_atoms)})"
        )
    if not np.isfinite(alpha).all() or (alpha < 0).any() or alpha.sum() <= 0:
        raise ValueError("alpha must be finite, non-negative, and sum to > 0")
    alpha_t = torch.as_tensor(alpha, dtype=dtype, device=device)
    # Sanitize alpha (it should already be on the simplex, but be defensive)
    alpha_t = torch.clamp(alpha_t, min=1e-12)
    alpha_t = alpha_t / alpha_t.sum()

    B = free_support_barycenter(
        gmms=global_atoms,
        weights=alpha_t,
        n_iters=config["barycenter_iters"],
        beta=config["beta_class"],
        var_floor=config["var_floor"],
        eps=config["eps"],
        init_idx=0,
        differentiable=False,   # decoding only, no autograd needed
    )
    return B


# ============================================================================
# Sample virtual features from a reconstructed GMM
# ============================================================================
def sample_virtual_features(recon_gmm: AtomGMM,
                            n_samples: int,
                            seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Sample n_samples from the reconstructed GMM in 256-d feature space.

    Sampling:
        component index c ~ Categorical(pi)
        feature x | c    ~ N(mu[c], diag(var[c]))      (diagonal covariance)
        label  y | c     = argmax(V[c])                (class assignment vector)

    Returns:
        features: (n_samples, feature_dim) float32
        labels:   (n_samples,) int64  in [0, num_classes)
    """
    rng = np.random.default_rng(seed)

    pi = recon_gmm.pi.detach().cpu().numpy().astype(np.float64)
    pi = np.maximum(pi, 1e-12)
    pi = pi / pi.sum()
    mu  = recon_gmm.mu.detach().cpu().numpy().astype(np.float64)    # (C, D)
    var = recon_gmm.var.detach().cpu().numpy().astype(np.float64)   # (C, D)
    V   = recon_gmm.V.detach().cpu().numpy().astype(np.float64)     # (C, n_class)

    C, D = mu.shape
    comp_labels = np.argmax(V, axis=1)   # (C,) class label per component

    # Sample component indices from Categorical(pi)
    comp_idx = rng.choice(C, size=n_samples, p=pi)

    features = np.zeros((n_samples, D), dtype=np.float32)
    labels = np.zeros(n_samples, dtype=np.int64)
    for c in range(C):
        mask = (comp_idx == c)
        n_c = int(mask.sum())
        if n_c == 0:
            continue
        std = np.sqrt(np.maximum(var[c], 1e-12))
        samples = rng.standard_normal(size=(n_c, D)) * std + mu[c]
        features[mask] = samples.astype(np.float32)
        labels[mask] = comp_labels[c]

    # Shuffle (so DataLoader batches don't cluster by class)
    perm = rng.permutation(n_samples)
    return features[perm], labels[perm]


# ============================================================================
# Match real client sizes
# ============================================================================
def get_real_client_sizes(train_root: str, num_clients: int) -> Dict[str, int]:
    """Read sample counts from real train_set to match virtual sample sizes.

    Only reads array shapes via mmap (no full load), so this is fast even
    for large datasets.
    """
    sizes = {}
    for i in range(num_clients):
        name = f"client{i}"
        path = os.path.join(train_root, name, "data.npy")
        if os.path.isfile(path):
            X = np.load(path, mmap_mode="r")
            sizes[name] = int(X.shape[0])
        else:
            sizes[name] = 410   # fallback if train_set not present
    return sizes


# ============================================================================
# Test-set feature extraction via saved backbone
# ============================================================================
def per_sample_zscore(x: np.ndarray) -> np.ndarray:
    """Per-sample z-score normalization along the signal axis.

    Mirrors DeFed_Dynamic.py / iid_dfl.py preprocessing - critical for
    BatchNorm consistency between train and test.
    """
    if x.ndim == 2:
        axis = 1
    elif x.ndim == 3:
        axis = 2
    else:
        raise ValueError(f"Expected ndim 2 or 3, got {x.ndim}")
    mean = x.mean(axis=axis, keepdims=True)
    std = x.std(axis=axis, keepdims=True)
    return ((x - mean) / np.maximum(std, 1e-6)).astype(np.float32)


def extract_test_features(test_root: str, extractor_path: str,
                          batch_size: int, device: str) -> Tuple[np.ndarray, np.ndarray]:
    """Extract 256-d features from test_set signals via the saved backbone.

    The backbone is the federated-trained BearingFeatureExtractor saved at
    backbone_results/dfl_cnn_backbone.pth (produced by DeFed_Dynamic.py).
    This is the same extractor that was used to fit the original domain GMM,
    so the test features live in the same 256-d space as the GMM and the
    virtual samples.
    """
    extractor = load_feature_extractor(extractor_path, device=device)
    X = np.load(os.path.join(test_root, "data.npy"))
    y = np.load(os.path.join(test_root, "labels.npy")).astype(np.int64)
    if X.ndim == 2:
        X = X[:, np.newaxis, :]
    X = per_sample_zscore(X)

    N = X.shape[0]
    feats = np.empty((N, extractor.feature_dim), dtype=np.float32)
    extractor.eval()
    with torch.no_grad():
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            batch = torch.from_numpy(X[start:end]).to(device).float()
            out = extractor(batch)
            feats[start:end] = out.cpu().numpy()
    return feats, y


# ============================================================================
# DFL training utilities (mirror DeFed_Dynamic.py)
# ============================================================================
def sample_dynamic_neighbors(client_names: Sequence[str],
                             neighbors_per_client: int,
                             rng: random.Random) -> Dict[str, List[str]]:
    """Sample a directed dynamic topology independently for every client.

    Identical to DeFed_Dynamic.py: each client picks `neighbors_per_client`
    distinct peers (without replacement) from the other clients.
    """
    if neighbors_per_client < 1 or neighbors_per_client >= len(client_names):
        raise ValueError(
            "neighbors_per_client must be between 1 and number_of_clients - 1"
        )
    topology: Dict[str, List[str]] = {}
    for name in client_names:
        candidates = [peer for peer in client_names if peer != name]
        topology[name] = rng.sample(candidates, neighbors_per_client)
    return topology


def train_local_mlp(model: nn.Module, loader: DataLoader,
                    device: torch.device, epochs: int,
                    lr: float, weight_decay: float
                    ) -> Tuple[Dict[str, torch.Tensor], float, float]:
    """Local SGD on the MLP for `epochs` epochs.

    Returns: (state_dict, avg_loss, train_accuracy)
    """
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_loss, total_correct, total_seen = 0.0, 0, 0

    for _ in range(epochs):
        for data, labels in loader:
            data = data.to(device, non_blocking=True).float()
            labels = labels.to(device, non_blocking=True).long()
            optimizer.zero_grad(set_to_none=True)
            logits = model(data)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            bs = labels.size(0)
            total_loss += loss.item() * bs
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_seen += bs

    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return state, total_loss / max(1, total_seen), 100.0 * total_correct / max(1, total_seen)


def is_bn_buffer(key: str) -> bool:
    return key.endswith("running_mean") or key.endswith("running_var") or \
           key.endswith("num_batches_tracked")


def mix_with_neighbors(own_state: Dict[str, torch.Tensor],
                       neighbor_states: Sequence[Dict[str, torch.Tensor]],
                       self_weight: float,
                       keep_bn_local: bool) -> Dict[str, torch.Tensor]:
    """Synchronous gossip averaging of one local state and selected peers.

    Identical to DeFed_Dynamic.py mix_with_neighbors.
    """
    if not 0.0 <= self_weight <= 1.0:
        raise ValueError("self_weight must be in [0, 1]")
    neighbor_weight = (1.0 - self_weight) / len(neighbor_states)
    mixed: Dict[str, torch.Tensor] = {}

    for key, own_value in own_state.items():
        if keep_bn_local and is_bn_buffer(key):
            mixed[key] = own_value.clone()
            continue
        if not torch.is_floating_point(own_value):
            # Integer counters (e.g. BN num_batches_tracked) cannot be averaged
            mixed[key] = own_value.clone()
            continue
        value = own_value.float().mul(self_weight)
        for peer_state in neighbor_states:
            value.add_(peer_state[key].float(), alpha=neighbor_weight)
        mixed[key] = value.to(dtype=own_value.dtype)
    return mixed


def state_distance(left: Dict[str, torch.Tensor],
                   right: Dict[str, torch.Tensor]) -> float:
    """Root-mean-square parameter distance, excluding BN integer counters."""
    squared_sum, element_count = 0.0, 0
    for key in left:
        if not torch.is_floating_point(left[key]):
            continue
        delta = left[key].double() - right[key].double()
        squared_sum += torch.sum(delta * delta).item()
        element_count += delta.numel()
    return float(np.sqrt(squared_sum / max(1, element_count)))


def choose_model_medoid(states: Dict[str, Dict[str, torch.Tensor]]
                        ) -> Tuple[str, Dict[str, float]]:
    """Choose the client state closest to all others (parameter-space medoid)."""
    names = list(states)
    mean_distances: Dict[str, float] = {}
    for name in names:
        distances = [
            state_distance(states[name], states[peer])
            for peer in names if peer != name
        ]
        mean_distances[name] = float(np.mean(distances))
    return min(mean_distances, key=mean_distances.get), mean_distances


def evaluate_mlp(model: nn.Module, features: np.ndarray, labels: np.ndarray,
                 device: torch.device, batch_size: int = 256
                 ) -> Tuple[float, np.ndarray, np.ndarray]:
    """Evaluate the MLP on (features, labels).

    Returns:
        accuracy: float (%)
        confusion_matrix: (num_classes, num_classes) ndarray
        predictions: (N,) ndarray
    """
    model.eval()
    X = torch.from_numpy(features).to(device).float()
    y = torch.from_numpy(labels).to(device).long()
    all_preds: List[np.ndarray] = []
    correct, total = 0, 0
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            end = min(start + batch_size, len(X))
            logits = model(X[start:end])
            pred = logits.argmax(dim=1)
            correct += (pred == y[start:end]).sum().item()
            total += (end - start)
            all_preds.append(pred.cpu().numpy())
    acc = 100.0 * correct / max(1, total)
    preds = np.concatenate(all_preds)
    cm = confusion_matrix(labels, preds, labels=list(range(model.num_classes)))
    return acc, cm, preds


# ============================================================================
# Main pipeline
# ============================================================================
def run_virtual_dfl(config: Dict) -> Tuple[Dict[str, nn.Module], Dict]:
    set_seed(config["seed"])
    device = torch.device(config["device"])
    dtype = get_dtype(config["dtype"])
    save_dir = Path(config["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("DeFed-GMM-DaDiL  Virtual Sample DFL Training")
    print("=" * 70)
    print(f"  defed_dict  : {config['defed_dict_path']}")
    print(f"  backbone    : {config['backbone_path']}")
    print(f"  train_root  : {config['train_data_root']}  (only for size matching)")
    print(f"  test_root   : {config['test_data_root']}")
    print(f"  save_dir    : {save_dir}")
    print(f"  K/C/cls/D   : {config['K']}/{config['C']}/{config['num_classes']}/{config['feature_dim']}")
    print(f"  num_clients : {config['num_clients']}")
    print(f"  rounds      : {config['rounds']}")
    print(f"  local_epochs: {config['local_epochs']}")
    print(f"  MLP hidden  : {config['mlp_hidden_dim']} (dropout={config['mlp_dropout']})")
    print(f"  device      : {device} / dtype={dtype}")

    # ---------------------------------------------------------------
    # Stage 1: Load defed dictionary (global atoms + per-client alpha)
    # ---------------------------------------------------------------
    print("\n[1] Load defed_dictionary.npz (global atoms + per-client alpha)")
    if not os.path.exists(config["defed_dict_path"]):
        raise FileNotFoundError(
            f"defed_dictionary.npz not found at {config['defed_dict_path']}. "
            "Run defed_gmm_dadil.py first."
        )
    metadata = load_dictionary_metadata(config["defed_dict_path"])
    for key in ("K", "C", "num_classes", "feature_dim"):
        if config[key] != metadata[key]:
            raise ValueError(
                f"dictionary {key}={metadata[key]}, but virtual DFL is configured "
                f"with {key}={config[key]}; rerun with matching settings"
            )
    if len(metadata["client_names"]) != config["num_clients"]:
        raise ValueError(
            f"dictionary contains {len(metadata['client_names'])} clients, but "
            f"num_clients={config['num_clients']}"
        )
    if metadata["C"] < metadata["num_classes"]:
        raise ValueError(
            "dictionary has fewer components per atom than classes; rerun "
            "defed_gmm_dadil.py with C >= num_classes"
        )

    global_atoms = load_global_atoms(
        config["defed_dict_path"], config["K"], device, dtype,
    )
    client_alphas = load_client_alphas(
        config["defed_dict_path"], metadata["client_names"],
    )
    print(f"  Loaded {len(global_atoms)} global atoms (each: {global_atoms[0]})")
    for name, alpha in client_alphas.items():
        print(f"  {name}: alpha={alpha.round(3)}")

    # ---------------------------------------------------------------
    # Stage 2: Reconstruct per-client GMM via free-support barycenter
    # ---------------------------------------------------------------
    print("\n[2] Reconstruct per-client GMM  (B = barycenter(alpha, global_atoms))")
    client_gmms: Dict[str, AtomGMM] = {}
    for name, alpha in client_alphas.items():
        B = reconstruct_client_gmm(global_atoms, alpha, config, device, dtype)
        client_gmms[name] = B
        pi_np = B.pi.detach().cpu().numpy().round(3)
        comp_labels = np.argmax(B.V.detach().cpu().numpy(), axis=1).tolist()
        missing_classes = sorted(set(range(config["num_classes"])) - set(comp_labels))
        if missing_classes:
            raise RuntimeError(
                f"{name} reconstructed GMM is missing class components "
                f"{missing_classes}; dictionary class assignments collapsed"
            )
        print(f"  {name}: pi={pi_np}  comp_class_labels={comp_labels}")

    # ---------------------------------------------------------------
    # Stage 3: Sample virtual features from each reconstructed GMM
    # ---------------------------------------------------------------
    print("\n[3] Sample virtual features from each client's reconstructed GMM")
    real_sizes = get_real_client_sizes(
        config["train_data_root"], config["num_clients"],
    )
    print(f"  Real client sizes: {real_sizes}")

    virtual_data: Dict[str, Dict[str, np.ndarray]] = {}
    total_virtual = 0
    for i, (name, gmm) in enumerate(client_gmms.items()):
        n = config.get("virtual_samples_per_client") or real_sizes[name]
        feats, labels = sample_virtual_features(
            gmm, n, seed=config["virtual_seed"] + i,
        )
        virtual_data[name] = {"features": feats, "labels": labels}
        total_virtual += len(feats)
        counts = np.bincount(labels, minlength=config["num_classes"]).tolist()
        print(f"  {name}: n={len(feats)}, class_counts={counts}")
    print(f"  Total virtual samples: {total_virtual}")

    # ---------------------------------------------------------------
    # Stage 4: Build DataLoaders for the virtual features
    # ---------------------------------------------------------------
    print("\n[4] Build DataLoaders for virtual features")
    loaders: Dict[str, DataLoader] = {}
    for name, d in virtual_data.items():
        ds = TensorDataset(
            torch.from_numpy(d["features"]),
            torch.from_numpy(d["labels"]),
        )
        loaders[name] = DataLoader(
            ds, batch_size=config["batch_size"], shuffle=True,
            drop_last=False, num_workers=config["num_workers"],
            pin_memory=torch.cuda.is_available(),
        )

    # ---------------------------------------------------------------
    # Stage 5: DFL training on the MLP (mirrors DeFed_Dynamic.py)
    # ---------------------------------------------------------------
    print(f"\n[5] DFL training  ({config['rounds']} rounds, dynamic topology)")
    print("-" * 70)
    client_names = list(virtual_data.keys())
    models: Dict[str, nn.Module] = {
        name: BearingMLP(
            feature_dim=config["feature_dim"],
            hidden_dim=config["mlp_hidden_dim"],
            num_classes=config["num_classes"],
            dropout=config["mlp_dropout"],
        ).to(device)
        for name in client_names
    }
    # Same initialization across clients; they diverge via local updates
    init_state = deepcopy(models[client_names[0]].state_dict())
    for name in client_names:
        models[name].load_state_dict(init_state)

    rng = random.Random(config["seed"])
    history: List[Dict] = []
    started = time.time()

    for r in range(config["rounds"]):
        round_num = r + 1
        topology = sample_dynamic_neighbors(
            client_names, config["neighbors_per_client"], rng,
        )

        # (a) Local update per client
        local_states: Dict[str, Dict[str, torch.Tensor]] = {}
        local_metrics: Dict[str, Dict[str, float]] = {}
        for name in client_names:
            state, loss, acc = train_local_mlp(
                models[name], loaders[name], device,
                config["local_epochs"], config["lr"], config["weight_decay"],
            )
            local_states[name] = state
            local_metrics[name] = {"loss": float(loss), "train_acc": float(acc)}

        # (b) Synchronous gossip: each client mixes with its 2 sampled peers
        mixed_states: Dict[str, Dict[str, torch.Tensor]] = {}
        for name in client_names:
            peer_states = [local_states[p] for p in topology[name]]
            mixed_states[name] = mix_with_neighbors(
                local_states[name], peer_states,
                config["self_weight"], config["keep_bn_local"],
            )
        for name in client_names:
            models[name].load_state_dict(mixed_states[name])

        # (c) Diagnostics
        mean_loss = float(np.mean([m["loss"] for m in local_metrics.values()]))
        mean_acc = float(np.mean([m["train_acc"] for m in local_metrics.values()]))
        history.append({
            "round": round_num,
            "topology": topology,
            "local_metrics": local_metrics,
            "mean_train_loss": mean_loss,
            "mean_train_accuracy": mean_acc,
        })
        print(f"  R{round_num:03d}/{config['rounds']}: "
              f"loss={mean_loss:.4f}, train_acc={mean_acc:.2f}%")
        print(f"    " + "; ".join(
            f"{n}->{','.join(ps)}" for n, ps in topology.items()
        ))

    # ---------------------------------------------------------------
    # Stage 6: Extract 256-d features from test_set via saved backbone
    # ---------------------------------------------------------------
    print("\n[6] Extract 256-d test features via saved backbone")
    if not os.path.exists(config["backbone_path"]):
        raise FileNotFoundError(
            f"Backbone not found at {config['backbone_path']}. "
            "Run DeFed_Dynamic.py first to produce the federated-trained backbone."
        )
    test_feats, test_labels = extract_test_features(
        config["test_data_root"], config["backbone_path"],
        config["batch_size"], config["device"],
    )
    test_counts = np.bincount(test_labels, minlength=config["num_classes"]).tolist()
    print(f"  test features: shape={test_feats.shape}, dtype={test_feats.dtype}, "
          f"labels={test_counts}")

    # ---------------------------------------------------------------
    # Stage 7: Evaluate each client MLP on the test features
    # ---------------------------------------------------------------
    print("\n[7] Final evaluation on test_set features")
    print("-" * 70)
    final_results: Dict[str, Dict[str, float]] = {}
    accs: List[float] = []
    cms: Dict[str, np.ndarray] = {}
    for name in client_names:
        acc, cm, _ = evaluate_mlp(models[name], test_feats, test_labels, device)
        final_results[name] = {"accuracy": acc}
        cms[name] = cm
        accs.append(acc)
        print(f"  {name}: test_acc={acc:.2f}%")

    print(f"\n  Client mean test_acc: {np.mean(accs):.2f}%")
    best_idx = int(np.argmax(accs))
    best_name = client_names[best_idx]
    print(f"  Best client: {best_name} ({np.max(accs):.2f}%)")

    # Medoid (parameter-space representative)
    final_states = {
        name: {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
        for name, m in models.items()
    }
    medoid, mean_distances = choose_model_medoid(final_states)
    print(f"  Medoid client (representative): {medoid}")
    medoid_acc = final_results[medoid]["accuracy"]
    print(f"  Medoid test_acc: {medoid_acc:.2f}%")

    # Best client confusion matrix
    print(f"\n  Best client ({best_name}) confusion matrix:")
    for i, row in enumerate(cms[best_name]):
        print(f"    {FAULT_NAMES.get(i, f'C{i}'):<8}: {row.tolist()}")

    # ---------------------------------------------------------------
    # Stage 8: Save results
    # ---------------------------------------------------------------
    print("\n[8] Save results")

    # Medoid MLP (representative)
    torch.save({
        "state_dict": final_states[medoid],
        "medoid_client": medoid,
        "medoid_test_accuracy": float(medoid_acc),
        "mean_test_accuracy": float(np.mean(accs)),
        "max_test_accuracy": float(np.max(accs)),
        "best_client": best_name,
        "config": config,
    }, save_dir / "virtual_dfl_medoid_mlp.pth")

    # Best MLP (highest test acc)
    torch.save({
        "state_dict": final_states[best_name],
        "best_client": best_name,
        "test_accuracy": float(np.max(accs)),
        "config": config,
    }, save_dir / "virtual_dfl_best_mlp.pth")

    # All client MLPs
    torch.save(
        {name: final_states[name] for name in client_names},
        save_dir / "virtual_dfl_all_mlps.pth",
    )

    # Metrics + history
    metrics = {
        "final_test_accuracy": final_results,
        "mean_test_accuracy": float(np.mean(accs)),
        "max_test_accuracy": float(np.max(accs)),
        "best_client": best_name,
        "medoid_client": medoid,
        "medoid_test_accuracy": float(medoid_acc),
        "mean_parameter_distances": mean_distances,
        "real_client_sizes": real_sizes,
        "virtual_sample_counts": {n: len(virtual_data[n]["features"]) for n in client_names},
        "client_alphas": {n: client_alphas[n].tolist() for n in client_names},
        "history": history,
    }

    def _to_jsonable(o):
        if isinstance(o, (np.floating, np.integer)):
            return float(o) if isinstance(o, np.floating) else int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, dict):
            return {k: _to_jsonable(v) for k, v in o.items()}
        if isinstance(o, list):
            return [_to_jsonable(x) for x in o]
        return o

    with open(save_dir / "virtual_dfl_metrics.json", "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(metrics), f, ensure_ascii=False, indent=2)
    with open(save_dir / "virtual_dfl_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2, default=str)

    print(f"  Saved medoid MLP : {save_dir / 'virtual_dfl_medoid_mlp.pth'}")
    print(f"  Saved best  MLP  : {save_dir / 'virtual_dfl_best_mlp.pth'}")
    print(f"  Saved all   MLPs : {save_dir / 'virtual_dfl_all_mlps.pth'}")
    print(f"  Saved metrics    : {save_dir / 'virtual_dfl_metrics.json'}")
    print(f"  Saved config     : {save_dir / 'virtual_dfl_config.json'}")
    print(f"  Elapsed: {time.time() - started:.1f}s")

    return models, metrics


# ============================================================================
# CLI
# ============================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="DeFed-GMM-DaDiL virtual sample DFL training"
    )
    p.add_argument("--defed-dict-path", type=str, default=None,
                   help="Path to defed_dictionary.npz")
    p.add_argument("--backbone-path", type=str, default=None,
                   help="Path to dfl_cnn_backbone.pth (for test feature extraction)")
    p.add_argument("--train-data-root", type=str, default=None,
                   help="Real train_set root (only used to match virtual sample sizes)")
    p.add_argument("--test-data-root", type=str, default=None,
                   help="Test set root (with data.npy/labels.npy)")
    p.add_argument("--save-dir", type=str, default=None)
    p.add_argument("--K", type=int, default=None)
    p.add_argument("--C", type=int, default=None)
    p.add_argument("--num-classes", type=int, default=None)
    p.add_argument("--feature-dim", type=int, default=None)
    p.add_argument("--num-clients", type=int, default=None)
    p.add_argument("--rounds", type=int, default=None, help="DFL communication rounds")
    p.add_argument("--local-epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument("--neighbors-per-client", type=int, default=None)
    p.add_argument("--self-weight", type=float, default=None)
    p.add_argument("--keep-bn-local", action="store_true",
                   help="Keep BN running stats local instead of gossip averaging")
    p.add_argument("--mlp-hidden-dim", type=int, default=None)
    p.add_argument("--mlp-dropout", type=float, default=None)
    p.add_argument("--virtual-samples-per-client", type=int, default=None,
                   help="Override real-client-size matching with a fixed count")
    p.add_argument("--barycenter-iters", type=int, default=None)
    p.add_argument("--beta-class", type=float, default=None)
    p.add_argument("--var-floor", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"])
    p.add_argument("--dtype", type=str, default=None, choices=["float32", "float64"])
    return p.parse_args()


def load_config(args: argparse.Namespace) -> Dict:
    config = DEFAULT_CONFIG.copy()
    # CLI overrides (only those explicitly set)
    for key in list(config.keys()):
        attr = key.replace("-", "_")
        v = getattr(args, attr, None)
        if v is not None or attr == "keep_bn_local":
            if attr == "keep_bn_local":
                if args.keep_bn_local:
                    config[key] = True
            else:
                config[key] = v
    return config


def main() -> None:
    args = parse_args()
    config = load_config(args)
    try:
        run_virtual_dfl(config)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
