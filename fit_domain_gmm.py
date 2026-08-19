"""
用联邦训练好的 CNN backbone 提取特征, 拟合 GMM-DaDiL 结构的域 GMM
====================================================================

Pipeline:
    1. 加载 BearingFeatureExtractor (dfl_cnn_backbone.pth)
    2. 从 Data_iid_nomalicious/train_set 读取 8 个客户端的原始振动信号
    3. per-sample z-score 标准化 (与训练时一致)
    4. 用 backbone 提取 256 维特征 (eval 模式, BN 用 running stats)
    5. 保留客户端边界, 对每个客户端分别处理特征 + 标签
    6. 在每个客户端内, 对已观测类别拟合最多 n 个对角协方差高斯分量
       (少样本时自适应减少分量, 缺失类别不造假数据)
    7. 为每个客户端组装 GMM-DaDiL 结构的域 GMM:

       每个高斯分量 (atom) 携带:
         - 均值 mu_k              (D,)
         - 方差 var_k (对角)      (D,)
         - 类别分配向量           (C,)   one-hot, 如 [1,0,0,0]
         - 类内权重 alpha_k       标量,  sum(alpha_k for k in class c) = 1
         - 类先验 pi_c            标量,  sum(pi_c) = 1
       有效混合权重 = pi_c * alpha_k

       参考:
         - GMM-DaDiL (Montesuma & Mboula, "Lighter, Better, Faster...")
         - DeFed-GMM-DaDiL (Clain et al. 2026, arXiv:2605.04324v1)

    8. 保存 <client>/domain_gmm.npz + <client>/extracted_features.npz
    9. 按客户端输出 BIC 诊断 (n=1..20) 帮助验证 n 的选择

用法:
    python fit_domain_gmm.py
    python fit_domain_gmm.py --n-components 4 --device cuda
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn

# 确保能 import model.py 中的 load_feature_extractor / BearingFeatureExtractor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import load_feature_extractor


# ============================ Defaults ============================
DEFAULT_BACKBONE_PATH = Path(__file__).parent / "backbone_results" / "dfl_cnn_backbone.pth"
DEFAULT_TRAIN_ROOT    = r"E:\FL\Data\Data_iid_nomalicious\train_set"
DEFAULT_OUTPUT_DIR    = Path(__file__).parent / "gmm_results"

DEFAULT_N_COMPONENTS  = 3       # 每类高斯分量数
DEFAULT_NUM_CLASSES   = 4
DEFAULT_MAX_ITER      = 200
DEFAULT_TOL           = 1e-4
DEFAULT_VAR_FLOOR     = 1e-3
DEFAULT_REG_COV       = 1e-3
DEFAULT_SEED          = 42
DEFAULT_DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_BATCH_SIZE    = 1024       # 特征提取时的 batch 大小

LABEL_NAMES = {0: "Normal (N)", 1: "Inner Fault (IF)",
               2: "Roller Fault (RF)", 3: "Outer Fault (OF)"}


# ============================ 数据加载 ============================
def per_sample_zscore(X: np.ndarray) -> np.ndarray:
    """对每个样本独立做 z-score: (x - mean) / std, 沿 signal 轴.
    与训练时的预处理完全一致, BatchNorm 才能用 running stats 正确归一化。
    """
    if X.ndim == 2:
        mean = X.mean(axis=1, keepdims=True)
        std = X.std(axis=1, keepdims=True) + 1e-6
    elif X.ndim == 3:
        mean = X.mean(axis=2, keepdims=True)
        std = X.std(axis=2, keepdims=True) + 1e-6
    else:
        raise ValueError(f"X.ndim 必须是 2 或 3, 收到 {X.ndim}")
    return (X - mean) / std


def load_all_clients(
    train_root: str,
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[str, slice]]]:
    """读取所有客户端的数据, 汇集提取特征并保留各客户端切片边界.

    Returns:
        X: (N_total, 1, signal_length) float32  (已 z-score, 已加通道维)
        y: (N_total,) int64
        client_slices: [(客户端名, 该客户端在 X/y 中的切片), ...]
    """
    root = Path(train_root)
    if not root.is_dir():
        raise ValueError(f"训练集根目录不存在: {train_root}")

    client_dirs = sorted([d for d in root.iterdir()
                          if d.is_dir() and (d / "data.npy").exists()])
    if not client_dirs:
        raise ValueError(f"未在 {train_root} 下发现含 data.npy 的客户端目录")

    Xs, ys = [], []
    client_slices = []
    offset = 0
    for d in client_dirs:
        X = np.load(d / "data.npy").astype(np.float32)
        y = np.load(d / "labels.npy").astype(np.int64)
        if len(X) != len(y):
            raise ValueError(
                f"{d.name} 的 data/labels 样本数不一致: {len(X)} != {len(y)}"
            )
        # per-sample z-score (与训练预处理一致)
        X = per_sample_zscore(X).astype(np.float32)
        if X.ndim == 2:
            X = X[:, np.newaxis, :]          # (N, 1, L)
        Xs.append(X)
        ys.append(y)
        client_slices.append((d.name, slice(offset, offset + len(X))))
        offset += len(X)
        print(f"  {d.name}: {len(X)} 样本, 分布={np.bincount(y, minlength=4).tolist()}")

    X_all = np.concatenate(Xs, axis=0)
    y_all = np.concatenate(ys, axis=0)
    print(f"  汇总: {len(X_all)} 样本, 分布={np.bincount(y_all, minlength=4).tolist()}")
    return X_all, y_all, client_slices


# ============================ 特征提取 ============================
@torch.no_grad()
def extract_features(extractor: nn.Module, X: np.ndarray,
                     device: str, batch_size: int) -> np.ndarray:
    """用 backbone 提取特征.

    Args:
        extractor: BearingFeatureExtractor (已 eval)
        X: (N, 1, L) float32
        device: 'cuda' 或 'cpu'
        batch_size: 每次前向的样本数
    Returns:
        features: (N, D) float32
    """
    N = X.shape[0]
    feats = np.empty((N, extractor.feature_dim), dtype=np.float32)
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        batch = torch.from_numpy(X[start:end]).to(device).float()
        out = extractor(batch)                 # (B, D)
        feats[start:end] = out.cpu().numpy()
    return feats


# ============================ 对角协方差 GMM ============================
class DiagonalCovGMM:
    """对角协方差 GMM, 用 EM 拟合. 所有计算在 torch 张量上 (CPU 或 CUDA).

    存储:
        weights   (K,)
        means     (K, D)
        variances (K, D)        对角协方差, 始终 >= var_floor
    """

    def __init__(self, n_components: int, dim: int,
                 var_floor: float = DEFAULT_VAR_FLOOR,
                 reg_cov: float = DEFAULT_REG_COV,
                 max_iter: int = DEFAULT_MAX_ITER,
                 tol: float = DEFAULT_TOL,
                 seed: int = DEFAULT_SEED,
                 device: str = "cpu"):
        self.K = n_components
        self.D = dim
        self.var_floor = var_floor
        self.reg_cov = reg_cov
        self.max_iter = max_iter
        self.tol = tol
        self.seed = seed
        self.device = device

        self.weights   = np.full(n_components, 1.0 / n_components, dtype=np.float64)
        self.means     = np.zeros((n_components, dim), dtype=np.float64)
        self.variances = np.full((n_components, dim), 1.0, dtype=np.float64)
        self.converged_ = False
        self.n_iter_    = 0
        self.log_likelihood_history_: List[float] = []

    def _kmeans_init(self, X: np.ndarray, rng: np.random.Generator):
        N = X.shape[0]
        K = self.K
        idx = rng.choice(N, size=K, replace=False)
        centers = X[idx].astype(np.float64)

        for _ in range(5):
            d2 = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            labels = d2.argmin(axis=1)
            for k in range(K):
                if (labels == k).sum() > 0:
                    centers[k] = X[labels == k].mean(axis=0)

        for k in range(K):
            mask = (labels == k)
            cnt = int(mask.sum())
            if cnt > 1:
                self.means[k]     = X[mask].mean(axis=0)
                self.variances[k] = X[mask].var(axis=0) + self.reg_cov
                self.weights[k]   = cnt / N
            elif cnt == 1:
                self.means[k]     = X[mask][0]
                self.variances[k] = X.var(axis=0) + self.reg_cov
                self.weights[k]   = 1.0 / N
            else:
                ridx = rng.integers(0, N)
                self.means[k]     = X[ridx]
                self.variances[k] = X.var(axis=0) + self.reg_cov
                self.weights[k]   = 1.0 / N
        self.variances = np.maximum(self.variances, self.var_floor)
        self.weights = self.weights / self.weights.sum()

    @staticmethod
    def _log_gaussian(X: torch.Tensor, means: torch.Tensor,
                      variances: torch.Tensor) -> torch.Tensor:
        """(N, K) log-density, 对角协方差."""
        D = X.shape[1]
        diff = X.unsqueeze(1) - means.unsqueeze(0)          # (N, K, D)
        mahala = (diff ** 2 / variances.unsqueeze(0)).sum(dim=2)   # (N, K)
        log_det = torch.log(variances).sum(dim=1)                   # (K,)
        return -0.5 * (D * np.log(2.0 * np.pi) + log_det.unsqueeze(0) + mahala)

    def _e_step(self, X, weights_t, means_t, variances_t):
        log_pdf = self._log_gaussian(X, means_t, variances_t) + torch.log(weights_t).unsqueeze(0)
        max_log, _ = log_pdf.max(dim=1, keepdim=True)
        log_prob_norm = max_log.squeeze(1) + torch.log(torch.exp(log_pdf - max_log).sum(dim=1))
        log_resp = log_pdf - log_prob_norm.unsqueeze(1)
        return log_resp, log_prob_norm

    def _m_step(self, X, log_resp):
        resp = torch.exp(log_resp)
        Nk = resp.sum(dim=0)
        Nk_safe = torch.clamp(Nk, min=1e-10)
        means_t = (resp.t() @ X) / Nk_safe.unsqueeze(1)
        diff = X.unsqueeze(1) - means_t.unsqueeze(0)
        variances_t = (resp.unsqueeze(2) * diff ** 2).sum(dim=0) / Nk_safe.unsqueeze(1)
        variances_t = variances_t + self.reg_cov
        variances_t = torch.clamp(variances_t, min=self.var_floor)
        weights_t = Nk / Nk.sum()
        self.weights   = weights_t.cpu().numpy().astype(np.float64)
        self.means     = means_t.cpu().numpy().astype(np.float64)
        self.variances = variances_t.cpu().numpy().astype(np.float64)

    def fit(self, X: np.ndarray) -> "DiagonalCovGMM":
        rng = np.random.default_rng(self.seed)
        self._kmeans_init(X, rng)
        X_t = torch.from_numpy(X).to(self.device).double()
        weights_t   = torch.from_numpy(self.weights).to(self.device)
        means_t     = torch.from_numpy(self.means).to(self.device)
        variances_t = torch.from_numpy(self.variances).to(self.device)

        prev_ll = -np.inf
        self.log_likelihood_history_ = []
        for it in range(1, self.max_iter + 1):
            log_resp, log_prob_norm = self._e_step(X_t, weights_t, means_t, variances_t)
            self._m_step(X_t, log_resp)
            weights_t   = torch.from_numpy(self.weights).to(self.device)
            means_t     = torch.from_numpy(self.means).to(self.device)
            variances_t = torch.from_numpy(self.variances).to(self.device)
            ll = log_prob_norm.sum().item()
            self.log_likelihood_history_.append(ll)
            self.n_iter_ = it
            if it > 1 and abs(ll - prev_ll) < self.tol * abs(prev_ll):
                self.converged_ = True
                break
            prev_ll = ll
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        X_t = torch.from_numpy(X).to(self.device).double()
        weights_t   = torch.from_numpy(self.weights).to(self.device)
        means_t     = torch.from_numpy(self.means).to(self.device)
        variances_t = torch.from_numpy(self.variances).to(self.device)
        _, log_prob_norm = self._e_step(X_t, weights_t, means_t, variances_t)
        return log_prob_norm.cpu().numpy()

    def score(self, X: np.ndarray) -> float:
        return float(self.score_samples(X).mean())

    def bic(self, X: np.ndarray) -> float:
        """Bayesian Information Criterion: BIC = -2*LL + k_params * log(N).
        越小越好. 对角协方差 GMM 每分量参数数 = D (均值) + D (方差) + 1 (权重).
        """
        N = X.shape[0]
        ll = self.score_samples(X).sum()
        n_params = self.K * (2 * self.D + 1)
        return -2 * ll + n_params * np.log(N)


# ============================ 域 GMM 组装 ============================
def fit_domain_gmm(features: np.ndarray, labels: np.ndarray,
                   n_components: int, num_classes: int,
                   max_iter: int, tol: float,
                   var_floor: float, reg_cov: float,
                   seed: int, device: str) -> dict:
    """对客户端实际观测到的类别拟合 GMM, 组装域 GMM.

    每类最多使用 ``n_components`` 个分量；当该类样本更少时，使用
    ``min(n_components, Nc)`` 个分量，不复制样本。缺失类别不创建虚假分量，
    其类别先验为 0。不同客户端因此可以有不同的 GMM 分量总数。

    返回 dict 含:
        means                (K_total, D)       各分量均值
        variances            (K_total, D)       各分量对角方差
        component_onehot     (K_total, C)       类别分配向量 (one-hot)
        within_class_weights (K_total,)         alpha_k: 类内权重
        class_priors         (C,)               pi_c: 类先验
        effective_weights    (K_total,)         pi_c * alpha_k: 有效混合权重
        per_class_ll         (C,)               每类平均 log-likelihood
        per_class_bic        (C,)               每类 BIC
        n_components_per_class (C,)             每类实际分量数
        requested_n_components_per_class int    请求的每类最大分量数
        num_classes          int
        feature_dim          int
    """
    if len(features) != len(labels) or len(labels) == 0:
        raise ValueError("features/labels 必须样本数一致且非空")
    invalid_labels = np.setdiff1d(np.unique(labels), np.arange(num_classes))
    if len(invalid_labels):
        raise ValueError(f"发现超出 [0, {num_classes - 1}] 的标签: {invalid_labels.tolist()}")

    D = features.shape[1]
    means_parts = []
    variances_parts = []
    onehot_parts = []
    within_weight_parts = []
    effective_weight_parts = []
    class_priors  = np.zeros(num_classes, dtype=np.float64)
    per_class_ll  = np.full(num_classes, np.nan, dtype=np.float64)
    per_class_bic = np.full(num_classes, np.nan, dtype=np.float64)
    component_counts = np.zeros(num_classes, dtype=np.int64)

    N_total = len(labels)

    for c in range(num_classes):
        mask = (labels == c)
        Xc = features[mask].astype(np.float64)
        Nc = len(Xc)
        class_priors[c] = Nc / N_total
        if Nc == 0:
            print(f"      [class {c}] N=0, 跳过 (由其他客户端提供该类别知识)")
            continue

        class_components = min(n_components, Nc)
        component_counts[c] = class_components
        if class_components < n_components:
            print(f"      [class {c}] 仅 {Nc} 个样本, 分量数自适应为 {class_components}")

        gmm = DiagonalCovGMM(
            n_components=class_components, dim=D,
            var_floor=var_floor, reg_cov=reg_cov,
            max_iter=max_iter, tol=tol,
            seed=seed + c, device=device,
        )
        gmm.fit(Xc)
        per_class_ll[c]  = gmm.score(Xc)
        per_class_bic[c] = gmm.bic(Xc)

        onehot = np.zeros((class_components, num_classes), dtype=np.float64)
        onehot[:, c] = 1.0
        means_parts.append(gmm.means)
        variances_parts.append(gmm.variances)
        onehot_parts.append(onehot)
        within_weight_parts.append(gmm.weights)
        effective_weight_parts.append(class_priors[c] * gmm.weights)

        label_name = LABEL_NAMES.get(c, f"Class {c}")
        print(f"      [class {c} ({label_name:<15})] "
              f"N={Nc}, prior={class_priors[c]:.4f}, "
              f"alpha={gmm.weights.round(4).tolist()}, "
              f"LL={per_class_ll[c]:.2f}, BIC={per_class_bic[c]:.1f}, "
              f"iters={gmm.n_iter_}, converged={gmm.converged_}")

    all_means = np.concatenate(means_parts, axis=0)
    all_variances = np.concatenate(variances_parts, axis=0)
    component_onehot = np.concatenate(onehot_parts, axis=0)
    within_class_weights = np.concatenate(within_weight_parts)
    effective_weights = np.concatenate(effective_weight_parts)

    # sanity: 有效权重之和 = sum(pi_c * sum(alpha_k)) = sum(pi_c * 1) = 1
    assert abs(effective_weights.sum() - 1.0) < 1e-10, \
        f"effective_weights sum = {effective_weights.sum()}, should be 1.0"

    return {
        "means":                all_means,
        "variances":            all_variances,
        "component_onehot":     component_onehot,
        "within_class_weights": within_class_weights,
        "class_priors":         class_priors,
        "effective_weights":    effective_weights,
        "per_class_ll":         per_class_ll,
        "per_class_bic":        per_class_bic,
        "n_components_per_class": component_counts,
        "requested_n_components_per_class": n_components,
        "num_classes":          num_classes,
        "feature_dim":          D,
    }


# ============================ 模型选择诊断 ============================
def knee_of_bic(bic_results: dict) -> int:
    """BIC 单调下降时, 取 '膝盖' (二阶差分最大) 处的 n 作为启发式选择."""
    ns = sorted(bic_results)
    if len(ns) < 3:
        return ns[0]
    bics = [bic_results[n] for n in ns]
    # 二阶差分: 收益递减最明显的地方 = 拐点
    d2 = [bics[i - 1] - 2 * bics[i] + bics[i + 1] for i in range(1, len(bics) - 1)]
    return ns[1 + int(np.argmax(d2))]


def validation_ll_sweep(features: np.ndarray, labels: np.ndarray,
                        n_candidates: List[int], num_classes: int,
                        val_frac: float, max_iter: int, tol: float,
                        var_floor: float, reg_cov: float,
                        seed: int, device: str) -> dict:
    """held-out 验证集 LL 扫描 (BIC 单调下降时的替代准则).

    每类按 val_frac 划分 train/val (对所有 n 使用同一划分保证公平),
    EM 只在训练集上拟合, 在验证集上评估 LL. 返回 {n: 平均 val LL}, 越大越好.
    """
    rng = np.random.default_rng(seed)
    splits = []
    for c in range(num_classes):
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)
        n_val = max(1, int(len(idx) * val_frac))
        splits.append((features[idx[n_val:]].astype(np.float64),
                       features[idx[:n_val]].astype(np.float64)))

    results = {}
    for n in n_candidates:
        total_ll, total_n = 0.0, 0
        for c, (Xtr, Xva) in enumerate(splits):
            if len(Xtr) < n:
                continue
            gmm = DiagonalCovGMM(
                n_components=n, dim=features.shape[1],
                var_floor=var_floor, reg_cov=reg_cov,
                max_iter=max_iter, tol=tol,
                seed=seed + c, device=device,
            )
            gmm.fit(Xtr)
            total_ll += gmm.score_samples(Xva).sum()
            total_n += len(Xva)
        results[n] = total_ll / max(1, total_n)
        print(f"  n={n}: val LL = {results[n]:.2f}")
    return results


def bic_sweep(features: np.ndarray, labels: np.ndarray,
              n_candidates: List[int], num_classes: int,
              max_iter: int, tol: float,
              var_floor: float, reg_cov: float,
              seed: int, device: str) -> dict:
    """对 n=1..N 计算 total BIC (4 类之和), 帮助选择 n.

    Returns: {n: total_bic}
    """
    results = {}
    for n in n_candidates:
        total_bic = 0.0
        for c in range(num_classes):
            Xc = features[labels == c].astype(np.float64)
            if len(Xc) < n:
                continue
            gmm = DiagonalCovGMM(
                n_components=n, dim=features.shape[1],
                var_floor=var_floor, reg_cov=reg_cov,
                max_iter=max_iter, tol=tol,
                seed=seed + c, device=device,
            )
            gmm.fit(Xc)
            total_bic += gmm.bic(Xc)
        results[n] = total_bic
        print(f"  n={n}: total BIC = {total_bic:.1f}")
    return results


# ============================ Main ============================
def main():
    parser = argparse.ArgumentParser(
        description="用 backbone 提取特征, 为每个训练客户端分别拟合域 GMM"
    )
    parser.add_argument("--backbone-path", type=str, default=str(DEFAULT_BACKBONE_PATH),
                        help="BearingFeatureExtractor 的 .pth 路径")
    parser.add_argument("--train-root", type=str, default=DEFAULT_TRAIN_ROOT,
                        help="train_set 根目录 (含 client0..client7)")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR),
                        help="输出根目录; 每个客户端会创建一个子目录")
    parser.add_argument("--n-components", type=int, default=DEFAULT_N_COMPONENTS,
                        help=f"每类高斯分量数 (默认 {DEFAULT_N_COMPONENTS})")
    parser.add_argument("--num-classes", type=int, default=DEFAULT_NUM_CLASSES)
    parser.add_argument("--max-iter", type=int, default=DEFAULT_MAX_ITER)
    parser.add_argument("--tol", type=float, default=DEFAULT_TOL)
    parser.add_argument("--var-floor", type=float, default=DEFAULT_VAR_FLOOR)
    parser.add_argument("--reg-cov", type=float, default=DEFAULT_REG_COV)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                        help="特征提取时的 batch 大小")
    parser.add_argument("--bic-sweep", action="store_true",
                        help="额外对 n=1..20 做 BIC + 验证集 LL 扫描, 帮助选择 n")
    parser.add_argument("--val-frac", type=float, default=0.2,
                        help="验证集 LL 扫描时每类划出的验证比例 (默认 0.2)")
    parser.add_argument(
        "--save-features",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否按客户端保存提取的特征 (可用 --no-save-features 关闭)",
    )
    args = parser.parse_args()
    if args.n_components < 1:
        parser.error("--n-components 必须 >= 1")
    if args.num_classes < 1:
        parser.error("--num-classes 必须 >= 1")
    if not 0.0 < args.val_frac < 1.0:
        parser.error("--val-frac 必须在 (0, 1) 内")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("特征提取 + 按客户端拟合域 GMM (GMM-DaDiL 结构)")
    print("=" * 70)
    print(f"  backbone      : {args.backbone_path}")
    print(f"  train_root    : {args.train_root}")
    print(f"  output_dir    : {args.output_dir}")
    print(f"  n_components  : {args.n_components} (每类)")
    print(f"  num_classes   : {args.num_classes}")
    print(f"  atoms/client  : {args.num_classes * args.n_components}")
    print(f"  device        : {args.device}")
    print()

    # 1. 加载 backbone
    print("[1/4] 加载特征提取器 ...")
    extractor = load_feature_extractor(args.backbone_path, device=args.device)
    print(f"  architecture  : {extractor.__class__.__name__}")
    print(f"  signal_length : {extractor.signal_length}")
    print(f"  num_channels  : {extractor.num_channels}")
    print(f"  feature_dim   : {extractor.feature_dim}")
    print(f"  (eval mode: BatchNorm 使用 running stats)")

    # 2. 加载数据并记录客户端边界
    print("\n[2/4] 加载训练数据并记录客户端边界 ...")
    X_raw, y_all, client_slices = load_all_clients(args.train_root)
    print(f"  总样本: {len(X_raw)}, shape={X_raw.shape}")
    print(f"  客户端数: {len(client_slices)}")

    # 3. 统一前向提取特征; eval 模式不会跨客户端更新/混合 BN 统计
    print("\n[3/4] 提取特征 ...")
    t0 = time.time()
    features = extract_features(extractor, X_raw, args.device, args.batch_size)
    del X_raw
    elapsed = time.time() - t0
    print(f"  features shape: {features.shape}, dtype={features.dtype}")
    print(f"  value range: [{features.min():.4f}, {features.max():.4f}]")
    print(f"  mean={features.mean():.4f}, std={features.std():.4f}")
    print(f"  耗时: {elapsed:.1f}s")

    # 4. 每个客户端独立拟合并保存域 GMM
    print("\n[4/4] 按客户端独立拟合域 GMM ...")
    client_outputs = {}
    for client_index, (client_name, client_slice) in enumerate(client_slices, start=1):
        client_features = features[client_slice]
        client_labels = y_all[client_slice]
        class_counts = np.bincount(
            client_labels, minlength=args.num_classes
        )[:args.num_classes]
        client_output_dir = output_dir / client_name
        client_output_dir.mkdir(parents=True, exist_ok=True)

        print()
        print("=" * 70)
        print(f"[{client_index}/{len(client_slices)}] {client_name}: "
              f"{len(client_features)} 样本")
        print(f"  类别分布: {class_counts.tolist()}")
        print("=" * 70)

        positive_counts = class_counts[class_counts > 0]
        min_observed_count = int(positive_counts.min())
        if np.any(class_counts == 0):
            missing_classes = np.where(class_counts == 0)[0].tolist()
            print(
                f"  [信息] 本地缺少类别 {missing_classes}; 不创建虚假分量, "
                "该知识将由联邦字典聚合从其他客户端迁移"
            )
        if min_observed_count < args.n_components:
            print(
                f"  [信息] 最少的已观测类别只有 {min_observed_count} 个样本; "
                "该类分量数将自适应缩减, 不复制样本"
            )

        feat_path = None
        if args.save_features:
            feat_path = client_output_dir / "extracted_features.npz"
            np.savez_compressed(
                feat_path,
                features=client_features,
                labels=client_labels,
                client_name=np.asarray(client_name),
            )
            print(f"  特征已保存: {feat_path}")

        if args.bic_sweep:
            max_candidate = min(20, min_observed_count)
            n_candidates = list(range(1, max_candidate + 1))
            print(f"\n  BIC 扫描 (n=1..{max_candidate}) ...")
            bic_results = bic_sweep(
                client_features, client_labels,
                n_candidates=n_candidates,
                num_classes=args.num_classes,
                max_iter=args.max_iter, tol=args.tol,
                var_floor=args.var_floor, reg_cov=args.reg_cov,
                seed=args.seed, device=args.device,
            )
            best_n = min(bic_results, key=bic_results.get)
            print(f"  BIC 最优 n = {best_n} (当前使用 n={args.n_components})")
            if best_n == max(bic_results):
                print("  [警告] BIC 单调下降, 请结合下面的验证集 LL 判断")
            knee_n = knee_of_bic(bic_results)
            print(f"  BIC 拐点 (启发式) n = {knee_n}")

            print("\n  验证集 LL 扫描 ...")
            val_results = validation_ll_sweep(
                client_features, client_labels,
                n_candidates=n_candidates,
                num_classes=args.num_classes,
                val_frac=args.val_frac,
                max_iter=args.max_iter, tol=args.tol,
                var_floor=args.var_floor, reg_cov=args.reg_cov,
                seed=args.seed, device=args.device,
            )
            best_val_n = max(val_results, key=val_results.get)
            print(f"  验证集 LL 最优 n = {best_val_n}  <-- 推荐以此为准")

        print(f"\n  拟合域 GMM (每类 {args.n_components} 分量) ...")
        t0 = time.time()
        result = fit_domain_gmm(
            features=client_features, labels=client_labels,
            n_components=args.n_components, num_classes=args.num_classes,
            max_iter=args.max_iter, tol=args.tol,
            var_floor=args.var_floor, reg_cov=args.reg_cov,
            seed=args.seed, device=args.device,
        )
        print(f"  拟合耗时: {time.time() - t0:.1f}s")

        actual_total_atoms = len(result["means"])
        print(f"\n  GMM 汇总: {actual_total_atoms} atoms; "
              f"每类实际分量数={result['n_components_per_class'].tolist()}")
        print(f"  class_priors            : {result['class_priors'].round(4).tolist()}")
        print(f"  effective_weights sum   : {result['effective_weights'].sum():.10f}")
        print(f"  per_class_ll            : {result['per_class_ll'].round(2).tolist()}")
        print(f"  per_class_bic           : {result['per_class_bic'].round(1).tolist()}")
        print(f"  variance range          : "
              f"[{result['variances'].min():.4e}, {result['variances'].max():.4e}]")

        gmm_path = client_output_dir / "domain_gmm.npz"
        np.savez_compressed(
            gmm_path,
            means=result["means"],
            variances=result["variances"],
            component_onehot=result["component_onehot"],
            within_class_weights=result["within_class_weights"],
            class_priors=result["class_priors"],
            effective_weights=result["effective_weights"],
            per_class_ll=result["per_class_ll"],
            per_class_bic=result["per_class_bic"],
            n_components_per_class=result["n_components_per_class"],
            requested_n_components_per_class=result[
                "requested_n_components_per_class"
            ],
            num_classes=result["num_classes"],
            feature_dim=result["feature_dim"],
            client_name=np.asarray(client_name),
            num_samples=np.asarray(len(client_features), dtype=np.int64),
        )
        size_kb = os.path.getsize(gmm_path) / 1024
        print(f"  域 GMM 已保存: {gmm_path} ({size_kb:.1f} KB)")

        client_outputs[client_name] = {
            "num_samples": len(client_features),
            "class_counts": class_counts.tolist(),
            "n_components_per_class": result[
                "n_components_per_class"
            ].tolist(),
            "total_atoms": actual_total_atoms,
            "domain_gmm_path": str(gmm_path),
            "extracted_features_path": str(feat_path) if feat_path else None,
        }

    # 保存配置
    config_path = output_dir / "gmm_config.json"
    import json
    config = {
        "backbone_path": args.backbone_path,
        "train_root": args.train_root,
        "requested_n_components_per_class": args.n_components,
        "num_classes": args.num_classes,
        "max_total_atoms_per_client": args.num_classes * args.n_components,
        "feature_dim": result["feature_dim"],
        "max_iter": args.max_iter,
        "tol": args.tol,
        "var_floor": args.var_floor,
        "reg_cov": args.reg_cov,
        "seed": args.seed,
        "device": args.device,
        "clients": client_outputs,
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    print(f"  配置已保存: {config_path}")

    first_client = client_slices[0][0]
    first_gmm_path = client_outputs[first_client]["domain_gmm_path"]
    print("\n全部客户端拟合完成:")
    for client_name, info in client_outputs.items():
        print(f"  {client_name}: {info['domain_gmm_path']}")

    print("\n加载示例:")
    print("  import numpy as np")
    print(f'  d = np.load(r"{first_gmm_path}")')
    print("  d['client_name']          # 当前 GMM 所属客户端")
    print("  d['means']                # (K_total, feature_dim)")
    print("  d['effective_weights']    # (K_total,) pi_c * alpha_k")


if __name__ == "__main__":
    main()
