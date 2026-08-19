"""
DeFed-GMM-DaDiL: Decentralized Federated GMM Dataset Dictionary Learning
=========================================================================
Implements Algorithm 1 of Clain et al. 2026 (arXiv:2605.04324v1) for the
IID bearing fault diagnosis setting. Dictionary atoms use the paper's uniform
component masses; fitted client-domain GMMs retain their learned mixture weights.

Pipeline:
    1. Load one fitted domain GMM per client as fixed targets Q_client.
    2. Each of N clients initializes K=3 atoms locally:
       - Each atom is a GMM with C=4 components (one per class).
       - Each component carries: pi (internal weight), mu (mean),
         var (diagonal variance), V (class assignment vector).
       - Client also has private alpha in Delta^K (barycentric coordinates
         over the local dictionary); alpha is NEVER exchanged.
    3. Local update (per client, per inner step):
       a. B = free_support_barycenter(alpha, P_local)  # differentiable
       b. L = SMW2^2(B_i, Q_i)                         # local target
       c. Backprop: gradients on P_local (mu, var, V) and alpha.
       d. Gradient step; project var>0 and V/alpha onto the simplex.
    4. Aggregation (per client, per round):
       a. Pick 2 random peers (without replacement).
       b. For each k in 1..K:
            new_atom_k = free_support_barycenter((1/3,1/3,1/3),
                        (own_atom_k, peer1_atom_k, peer2_atom_k))
       c. P_local <- {new_atom_k}_{k=1..K}.  alpha unchanged.
    5. Loop until P_ell converges across clients.
    6. Compute and save the Wasserstein barycenter global atom dictionary
       + each client's private alpha.

References:
    - DeFed-GMM-DaDiL: Clain, Montesuma, Ngole Mboula, arXiv:2605.04324v1.
    - GMM-DaDiL: Montesuma & Mboula, ECML PKDD 2024.
    - W2^2 between diagonal Gaussians: Bures metric, closed form.

Usage:
    python defed_gmm_dadil.py
    python defed_gmm_dadil.py --rounds 50 --inner-steps 5 --device cpu
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linprog


# ============================================================================
# Defaults
# ============================================================================
DEFAULT_CONFIG = {
    # Data / fitted GMM
    "domain_gmm_dir": str(Path(__file__).parent / "gmm_results"),
    "save_dir": str(Path(__file__).parent / "defed_gmm_dadil_results"),

    # Dictionary structure
    "K": 3,               # atoms per client dictionary
    "C": 4,               # components per atom (one per class)
    "num_classes": 4,
    "feature_dim": 256,

    # Training
    "num_clients": 8,
    "rounds": 40,            # DFL communication rounds
    "inner_steps": 5,        # local gradient steps per round
    "lr_mu": 0.5,            # learning rate for means
    "lr_var": 0.05,          # learning rate for variances (smaller, sensitive)
    "lr_V": 0.05,            # learning rate for class vectors
    "lr_pi": 0.0,            # deprecated: atom component masses stay uniform
    "lr_alpha": 0.02,        # learning rate for private coordinates
    "grad_clip": 50.0,       # max grad norm per parameter (prevents blow-ups)
    "entropy_reg": 0.5,      # weight on alpha entropy reg (prevents collapse to one-hot)

    # Barycenter / loss
    "barycenter_iters": 8,   # fixed-point iterations for free-support barycenter
    "beta_class": 1.0,       # multiplier for data-scaled label mismatch cost
    "var_floor": 1e-3,       # lower bound on variances
    "var_init": 1.0,         # initial diagonal variance
    "eps": 1e-9,

    # Aggregation
    "neighbors_per_client": 2,
    "self_weight": 1.0 / 3.0,  # weight of own atom in 3-way barycenter

    # Diagnostics
    "eval_every": 1,
    "converge_tol": 1e-3,    # pairwise dictionary divergence threshold for "converged"

    # Misc
    "seed": 42,
    "device": "cpu",         # CPU is fine; OT dominates runtime not matmul
    "dtype": "float64",
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "float64": torch.float64}[name]


# ============================================================================
# Atom data structures
# ============================================================================
@dataclass
class AtomGMM:
    """One atom: a GMM with C components, each with (pi, mu, var, V).

    Shapes:
        pi:  (C,)
        mu:  (C, D)
        var: (C, D)
        V:   (C, n_class)
    All torch tensors, requires_grad set externally.
    """
    pi: torch.Tensor
    mu: torch.Tensor
    var: torch.Tensor
    V: torch.Tensor

    @classmethod
    def from_array(cls, pi, mu, var, V, device, dtype, requires_grad=False) -> "AtomGMM":
        t = lambda x: torch.as_tensor(x, dtype=dtype, device=device)
        return cls(
            pi=t(pi).clone().requires_grad_(requires_grad),
            mu=t(mu).clone().requires_grad_(requires_grad),
            var=t(var).clone().requires_grad_(requires_grad),
            V=t(V).clone().requires_grad_(requires_grad),
        )

    def detach(self) -> "AtomGMM":
        return AtomGMM(self.pi.detach(), self.mu.detach(),
                       self.var.detach(), self.V.detach())

    def clone(self) -> "AtomGMM":
        return AtomGMM(self.pi.clone(), self.mu.clone(),
                       self.var.clone(), self.V.clone())

    def to_dict(self) -> Dict[str, torch.Tensor]:
        return {"pi": self.pi.detach().cpu(), "mu": self.mu.detach().cpu(),
                "var": self.var.detach().cpu(), "V": self.V.detach().cpu()}

    def __repr__(self) -> str:
        return (f"AtomGMM(C={self.pi.shape[0]}, D={self.mu.shape[1]}, "
                f"n_class={self.V.shape[1]}, requires_grad={self.mu.requires_grad})")


@dataclass
class AtomDictionary:
    """Local atom dictionary: K atoms + private alpha in Delta^K."""
    atoms: List[AtomGMM]            # length K
    alpha: torch.Tensor             # (K,)

    def __len__(self) -> int:
        return len(self.atoms)

    @property
    def K(self) -> int:
        return len(self.atoms)

    def detach(self) -> "AtomDictionary":
        return AtomDictionary([a.detach() for a in self.atoms], self.alpha.detach())

    def clone(self) -> "AtomDictionary":
        return AtomDictionary([a.clone() for a in self.atoms], self.alpha.clone())

    def parameters(self) -> Dict[str, List[torch.Tensor]]:
        """Return all learnable tensors in a structured dict (for optimizer)."""
        return {
            "mu": [a.mu for a in self.atoms],
            "var": [a.var for a in self.atoms],
            "V": [a.V for a in self.atoms],
            "pi": [a.pi for a in self.atoms],
            "alpha": [self.alpha],
        }


# ============================================================================
# Initialization
# ============================================================================
def init_atom_dictionary(K: int, C: int, D: int, n_class: int,
                         device: torch.device, dtype: torch.dtype,
                         var_init: float = 1.0, seed: int = 0) -> AtomDictionary:
    """Initialize one client's atom dictionary.

    Per paper Algorithm 1:
        M ~ N(0, I)
        S <- 1   (std=1 => var=1)
        V <- repeated one-hot class assignments
        alpha <- 1/K     (uniform weights)
    pi <- 1/C and remains fixed, as in the reference GMM-DaDiL algorithm.
    """
    g = torch.Generator(device=device).manual_seed(seed)
    atoms: List[AtomGMM] = []
    for _ in range(K):
        mu = torch.randn(C, D, generator=g, device=device, dtype=dtype)
        var = torch.full((C, D), float(var_init), device=device, dtype=dtype)
        component_labels = torch.arange(C, device=device) % n_class
        V = F.one_hot(component_labels, num_classes=n_class).to(dtype=dtype)
        pi = torch.full((C,), 1.0 / C, device=device, dtype=dtype)
        atoms.append(AtomGMM(
            pi=pi.clone(),
            mu=mu.clone().requires_grad_(True),
            var=var.clone().requires_grad_(True),
            V=V.clone().requires_grad_(True),
        ))
    alpha = torch.full((K,), 1.0 / K, device=device, dtype=dtype).requires_grad_(True)
    return AtomDictionary(atoms=atoms, alpha=alpha)


def init_all_clients(config: Dict,
                     client_names: Sequence[str]) -> Dict[str, AtomDictionary]:
    """Each client gets a different random init (different seed)."""
    device = torch.device(config["device"])
    dtype = get_dtype(config["dtype"])
    clients = {}
    for i, client_name in enumerate(client_names):
        # Different seed per client so initial dictionaries differ.
        # The paper uses the same init then diverges via local updates;
        # we follow the DFL convention (different inits) to make aggregation
        # non-trivial from round 1.
        seed = config["seed"] + 1000 * (i + 1)
        clients[client_name] = init_atom_dictionary(
            K=config["K"], C=config["C"], D=config["feature_dim"],
            n_class=config["num_classes"], device=device, dtype=dtype,
            var_init=config["var_init"], seed=seed,
        )
    return clients


# ============================================================================
# Wasserstein utilities (diagonal covariance)
# ============================================================================
def _std_from_var(var: torch.Tensor, var_floor: float = 1e-12) -> torch.Tensor:
    return torch.sqrt(torch.clamp(var, min=var_floor))


def w2_sq_gaussian(mu1: torch.Tensor, var1: torch.Tensor,
                   mu2: torch.Tensor, var2: torch.Tensor,
                   var_floor: float = 1e-12) -> torch.Tensor:
    """W2^2 between diagonal-covariance Gaussians. Broadcasts over leading dims.

    W2^2 = ||m1 - m2||^2 + sum_d (sqrt(var1_d) - sqrt(var2_d))^2
    """
    sq_mu = ((mu1 - mu2) ** 2).sum(-1)
    s1 = _std_from_var(var1, var_floor)
    s2 = _std_from_var(var2, var_floor)
    sq_std = ((s1 - s2) ** 2).sum(-1)
    return sq_mu + sq_std


def cost_matrix(B: AtomGMM, Q: AtomGMM, beta: float,
                var_floor: float = 1e-12) -> torch.Tensor:
    """Pairwise SMW2^2 cost between B's components and Q's components.

    C[i,j] = W2^2(B_i, Q_j) + beta * ||V_B_i - V_Q_j||^2

    Returns: (K_B, K_Q) tensor, differentiable in B.mu, B.var, B.V.
    (Cost does NOT depend on B.pi or Q.pi.)
    """
    # mu: (K, D)
    mu_diff = B.mu.unsqueeze(1) - Q.mu.unsqueeze(0)        # (K_B, K_Q, D)
    sq_mu = (mu_diff ** 2).sum(-1)                          # (K_B, K_Q)

    sB = _std_from_var(B.var, var_floor)                   # (K_B, D)
    sQ = _std_from_var(Q.var, var_floor)                   # (K_Q, D)
    std_diff = sB.unsqueeze(1) - sQ.unsqueeze(0)           # (K_B, K_Q, D)
    sq_std = (std_diff ** 2).sum(-1)                        # (K_B, K_Q)

    V_diff = B.V.unsqueeze(1) - Q.V.unsqueeze(0)           # (K_B, K_Q, n_class)
    sq_V = (V_diff ** 2).sum(-1)                            # (K_B, K_Q)

    feature_cost = sq_mu + sq_std
    # Match the reference GMM-DaDiL implementation: label mismatch must be on
    # the same scale as the feature-space transport cost.  A fixed beta=1 is
    # negligible for 256-d features and lets class-bearing components collapse.
    label_scale = float(beta) * feature_cost.detach().amax()
    return feature_cost + label_scale * sq_V


def _normalize_marginal(p: np.ndarray, tol: float = 1e-10,
                        smooth_eps: float = 1e-8) -> np.ndarray:
    """Make p a valid probability vector that the LP solver will accept.

    Steps:
        1. Clip to non-negative.
        2. Threshold tiny values (< tol) to 0.
        3. Add a small `smooth_eps` to every entry.  This is the key step:
           without it, a one-hot p like [0,0,1,0] combined with a near-one-hot
           q like [1e-8, 1e-8, 1e-8, 1.0] gives a genuinely infeasible LP
           (q's true sum > 1.0 but fp rounds it to 1.0, so the marginals
           disagree at the 1e-8 level).
        4. Renormalize to sum to exactly 1.0 (residual dumped on argmax).
    """
    p = np.clip(p, 0.0, None)
    p = np.where(p < tol, 0.0, p)
    p = p + smooth_eps
    s = p.sum()
    if s <= 0:
        return np.full_like(p, 1.0 / p.shape[0])
    p = p / s
    # Force sum to exactly 1.0 (dump fp residual on the argmax).
    residual = 1.0 - p.sum()
    if abs(residual) > 0:
        i = int(np.argmax(p))
        p[i] += residual
    return p


def solve_ot_lp(cost: np.ndarray, p: np.ndarray, q: np.ndarray
                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve the OT LP:  min <C, omega>  s.t.  omega 1 = p,  omega^T 1 = q,  omega>=0.

    Returns:
        omega:   (K_B, K_Q) optimal transport plan
        dual_p:  (K_B,) dual variables for the marginal-p constraint (= dL/dp by envelope)
        dual_q:  (K_Q,) dual variables for the marginal-q constraint
    """
    K_B, K_Q = cost.shape
    # Make both marginals valid probability vectors with sum EXACTLY 1.
    p = _normalize_marginal(p)
    q = _normalize_marginal(q)
    c = cost.flatten()                               # (K_B*K_Q,)
    # Equality constraints: A_eq @ omega = b_eq
    #   row i (i in 0..K_B-1):           sum_j omega[i,j] = p[i]
    #   row K_B + j (j in 0..K_Q-1):     sum_i omega[i,j] = q[j]
    A_eq = np.zeros((K_B + K_Q, K_B * K_Q), dtype=cost.dtype)
    for i in range(K_B):
        A_eq[i, i * K_Q:(i + 1) * K_Q] = 1.0
    for j in range(K_Q):
        for i in range(K_B):
            A_eq[K_B + j, i * K_Q + j] = 1.0
    b_eq = np.concatenate([p, q])

    bounds = [(0, None)] * (K_B * K_Q)
    # presolve=False: HiGHS's presolve incorrectly flags near-degenerate
    # marginals (one-hot-like p combined with q containing tiny values) as
    # infeasible.  Disabling presolve lets the simplex run on the raw problem,
    # which handles these cases correctly.  Cost is negligible for our small
    # (K_B x K_Q) OT problems.
    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs",
                  options={"presolve": False})
    if not res.success:
        # Numerical fallback: tiny epsilon on marginals
        b_eq = b_eq + 1e-12
        res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs",
                      options={"presolve": False})
        if not res.success:
            raise RuntimeError(f"OT LP failed: {res.message}")

    omega = res.x.reshape(K_B, K_Q)
    # Duals for equality constraints live in res.eqlin.marginals (scipy >= 1.7)
    duals = res.eqlin.marginals if res.eqlin is not None else np.zeros(K_B + K_Q)
    dual_p = duals[:K_B]
    dual_q = duals[K_B:]
    return omega, dual_p, dual_q


# ============================================================================
# Free-support SMW2 barycenter (GMM-DaDiL Algorithm 1)
# ============================================================================
def free_support_barycenter(gmms: Sequence[AtomGMM],
                            weights: torch.Tensor,
                            n_iters: int,
                            beta: float,
                            var_floor: float,
                            eps: float,
                            init_idx: int = 0,
                            differentiable: bool = True) -> AtomGMM:
    """Free-support SMW2 barycenter of C_input GMMs.

    Algorithm (GMM-DaDiL Alg.1):
        Init B with K_B components (here K_B = C, copied from gmms[init_idx]).
        Repeat:
            For each c, solve OT(B, P_c) -> omega_c.
            Update B's (mu, var, V) via barycentric projection:
                x_B[i] <- sum_c w_c * sum_j (omega_c[i,j] / p_B[i]) * x_Pc[j]
            B.pi unchanged (set by init).
        Return B.

    Args:
        gmms: list of C_input AtomGMMs (each with C components).
        weights: (C_input,) barycentric weights, sums to 1.
        n_iters: fixed-point iterations.
        differentiable: if True, the barycentric projection is unrolled and
            differentiable through gmms' mu/var/V (OT plans are detached).
            If False, all tensors are detached (used for aggregation).

    Returns:
        AtomGMM barycenter with C components.
    """
    C_input = len(gmms)
    if C_input < 1:
        raise ValueError("gmms must contain at least one GMM")
    if weights.ndim != 1 or weights.numel() != C_input:
        raise ValueError(
            f"barycenter weight count ({weights.numel()}) must match "
            f"the number of GMMs ({C_input})"
        )
    if not torch.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
        raise ValueError("barycenter weights must be finite, non-negative, and sum to > 0")
    if not 0 <= init_idx < C_input:
        raise ValueError(f"init_idx out of range: {init_idx}")
    C = gmms[0].mu.shape[0]
    device = gmms[0].mu.device
    dtype = gmms[0].mu.dtype
    for index, gmm in enumerate(gmms):
        if gmm.pi.shape != (gmm.mu.shape[0],) or gmm.var.shape != gmm.mu.shape:
            raise ValueError(f"invalid GMM tensor shapes at index {index}")
        if gmm.mu.shape[0] != C or gmm.mu.shape[1:] != gmms[0].mu.shape[1:] \
           or gmm.V.shape != gmms[0].V.shape:
            raise ValueError("all barycenter input GMMs must have matching component shapes")

    # Init B from gmms[init_idx]. Detach so the init isn't part of the graph
    # (gradients flow through the barycentric projection updates, not the init).
    B_mu = gmms[init_idx].mu.detach().clone()
    B_var = gmms[init_idx].var.detach().clone()
    B_V = gmms[init_idx].V.detach().clone()
    B_pi = gmms[init_idx].pi.detach().clone()       # (C,)
    p_B = B_pi                                       # marginal, fixed

    weights = weights.to(device=device, dtype=dtype)
    weights = weights / weights.sum()
    if not differentiable:
        weights = weights.detach()

    # The barycenter component marginal is the weighted input marginal.  Keep
    # its differentiable value for the returned GMM, while OT plans and the
    # fixed-point denominator use a detached, normalized marginal.
    B_pi_final = torch.zeros_like(B_pi)
    for c in range(C_input):
        source_pi = gmms[c].pi if differentiable else gmms[c].pi.detach()
        B_pi_final = B_pi_final + weights[c] * source_pi
    B_pi_final = torch.clamp(B_pi_final, min=eps)
    B_pi_final = B_pi_final / B_pi_final.sum()
    p_B = B_pi_final.detach()

    for _ in range(n_iters):
        # 1. Compute OT plans between B and each input GMM (detached).
        ot_plans: List[torch.Tensor] = []
        for c in range(C_input):
            with torch.no_grad():
                cost = cost_matrix(AtomGMM(B_pi, B_mu, B_var, B_V),
                                   gmms[c].detach(), beta, var_floor)
                omega, _, _ = solve_ot_lp(
                    cost.detach().cpu().numpy().astype(np.float64),
                    p_B.detach().cpu().numpy().astype(np.float64),
                    gmms[c].pi.detach().cpu().numpy().astype(np.float64),
                )
            ot_plans.append(torch.as_tensor(omega, dtype=dtype, device=device))

        # 2. Barycentric projection update (differentiable given plans as constants).
        #    coef_c[i, j] = omega_c[i, j] / p_B[i]
        #    new_x[i] = sum_c w_c * sum_j coef_c[i, j] * x_Pc[j]
        # Vectorized:  new_x = sum_c w_c * (coef_c @ x_Pc)   where coef_c is (C, C)
        new_mu = torch.zeros_like(B_mu)
        new_std = torch.zeros_like(B_var)
        new_V = torch.zeros_like(B_V)
        for c in range(C_input):
            omega_c = ot_plans[c]                                   # (C, C)
            coef = omega_c / (p_B.unsqueeze(-1) + eps)             # (C, C)
            # coef @ x_Pc : (C, C) @ (C, D) -> (C, D)
            if differentiable:
                pc_mu = gmms[c].mu
                pc_std = _std_from_var(gmms[c].var, var_floor)
                pc_V = gmms[c].V
            else:
                pc_mu = gmms[c].mu.detach()
                pc_std = _std_from_var(gmms[c].var.detach(), var_floor)
                pc_V = gmms[c].V.detach()
            new_mu = new_mu + weights[c] * (coef @ pc_mu)
            new_std = new_std + weights[c] * (coef @ pc_std)
            new_V = new_V + weights[c] * (coef @ pc_V)

        B_mu = new_mu
        # W2 geometry is Euclidean in diagonal standard deviations, not in
        # variances.  Project stds, then square to return to variance form.
        B_var = torch.clamp(new_std, min=math.sqrt(var_floor)) ** 2
        B_V = new_V
        # p_B unchanged

    return AtomGMM(pi=B_pi_final, mu=B_mu, var=B_var, V=B_V)


# ============================================================================
# SMW2^2 loss with envelope-theorem gradient for pi
# ============================================================================
def smw2_sq_loss(B: AtomGMM, Q: AtomGMM, beta: float,
                 var_floor: float, eps: float) -> torch.Tensor:
    """Supervised Mixture-Wasserstein^2 loss between B and Q.

    L = min_omega  sum_ij  omega_ij * ( W2^2(B_i, Q_j) + beta * ||V_B_i - V_Q_j||^2 )
      = sum_ij  omega*_ij * C_ij(B)

    where omega* solves the OT LP with marginals B.pi and Q.pi.

    Differentiability:
        - dB.mu, dB.var, dB.V: via C_ij (autograd).
        - dB.pi: via the envelope theorem.  dL/dB.pi_i = dual_p_i.
          We implement this with a straight-through term:
            loss += dual_p.detach() @ (B.pi - B.pi.detach())
          which is 0 in forward but contributes gradient dual_p in backward.

    Q is treated as a constant (target).
    """
    # Cost matrix (differentiable in B.mu, B.var, B.V)
    C = cost_matrix(B, Q, beta=beta, var_floor=var_floor)     # (K_B, K_Q)

    # Solve OT (non-differentiable)
    with torch.no_grad():
        C_np = C.detach().cpu().numpy().astype(np.float64)
        p_np = B.pi.detach().cpu().numpy().astype(np.float64)
        q_np = Q.pi.detach().cpu().numpy().astype(np.float64)
        omega, dual_p, _ = solve_ot_lp(C_np, p_np, q_np)
    omega_t = torch.as_tensor(omega, dtype=C.dtype, device=C.device)        # (K_B, K_Q)
    dual_p_t = torch.as_tensor(dual_p, dtype=B.pi.dtype, device=B.pi.device)  # (K_B,)

    # Differentiable loss: sum_ij omega_ij * C_ij(B)
    loss = (omega_t * C).sum()

    # Envelope-theorem gradient for B.pi (straight-through).
    # Forward value: dual_p @ (B.pi - B.pi.detach()) = 0.
    # Backward gradient w.r.t. B.pi: dual_p.
    pi_envelope = (dual_p_t * (B.pi - B.pi.detach())).sum()
    loss = loss + pi_envelope
    return loss


# ============================================================================
# Local update: gradient descent on SMW2^2(B_i(alpha_i, P_i), Q_i)
# ============================================================================
def project_simplex(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Project onto the probability simplex (sum to 1, non-negative).

    Uses the standard sort-based algorithm.  Final renormalization guarantees
    sum = 1 exactly (up to fp rounding), which is critical for OT feasibility.
    """
    n = x.shape[0]
    u, _ = torch.sort(x, descending=True)
    cssv = torch.cumsum(u, dim=0) - 1
    ind = torch.arange(1, n + 1, dtype=x.dtype, device=x.device)
    cond = u > cssv / ind
    rho = cond.nonzero().max()
    theta = cssv[rho] / (rho + 1)
    w = torch.clamp(x - theta, min=eps)
    return w / w.sum()


def project_var(var: torch.Tensor, var_floor: float) -> torch.Tensor:
    return torch.clamp(var, min=var_floor)


def local_update(client_dict: AtomDictionary,
                 Q_local: AtomGMM,
                 config: Dict,
                 device: torch.device,
                 dtype: torch.dtype) -> Dict[str, float]:
    """Run `inner_steps` gradient steps on (P, alpha) to minimize SMW2^2(B, Q).

    The total loss is:
        L = SMW2^2(B(alpha, P), Q) - entropy_reg * H(alpha)
    where the entropy term keeps alpha away from one-hot collapse. Each client
    uses its own Q_i; knowledge transfer happens during dictionary aggregation.

    Returns a dict of per-step loss values (for diagnostics).
    """
    log = {"losses": []}
    entropy_reg = float(config.get("entropy_reg", 0.0))
    grad_clip = float(config.get("grad_clip", 0.0))

    for step in range(config["inner_steps"]):
        # Zero grads
        for a in client_dict.atoms:
            for p in (a.mu, a.var, a.V):
                if p.grad is not None:
                    p.grad.zero_()
        if client_dict.alpha.grad is not None:
            client_dict.alpha.grad.zero_()

        # Forward: B = barycenter(alpha, P)
        B = free_support_barycenter(
            gmms=client_dict.atoms,
            weights=client_dict.alpha,
            n_iters=config["barycenter_iters"],
            beta=config["beta_class"],
            var_floor=config["var_floor"],
            eps=config["eps"],
            init_idx=0,
            differentiable=True,
        )

        # Loss: SMW2^2 + optional entropy reg on alpha
        loss = smw2_sq_loss(B, Q_local,
                            beta=config["beta_class"],
                            var_floor=config["var_floor"],
                            eps=config["eps"])
        if entropy_reg > 0:
            # H(alpha) = -sum_k alpha_k log(alpha_k).  Minimizing -H encourages uniform.
            a = torch.clamp(client_dict.alpha, min=1e-12)
            entropy = -(a * torch.log(a)).sum()
            loss = loss - entropy_reg * entropy

        # Backward
        loss.backward()
        log["losses"].append(float(loss.detach()))

        # Gradient clipping (per parameter tensor, by global norm)
        if grad_clip > 0:
            for a in client_dict.atoms:
                for p in (a.mu, a.var, a.V):
                    if p.grad is not None:
                        torch.nn.utils.clip_grad_norm_(p, grad_clip)
            if client_dict.alpha.grad is not None:
                torch.nn.utils.clip_grad_norm_(client_dict.alpha, grad_clip)

        # Manual SGD step with per-parameter LR + projection.
        # IMPORTANT: use .data so the leaf tensors remain leaves with
        # requires_grad=True. Reassigning (a.var = ...) inside no_grad
        # would create a new non-leaf tensor and break the next backward.
        with torch.no_grad():
            for a in client_dict.atoms:
                # mu: unconstrained
                a.mu.data -= config["lr_mu"] * a.mu.grad
                # var: project to >= var_floor
                a.var.data -= config["lr_var"] * a.var.grad
                a.var.data.clamp_(min=config["var_floor"])
                # V: project each row onto the simplex
                a.V.data -= config["lr_V"] * a.V.grad
                for i in range(a.V.shape[0]):
                    a.V.data[i] = project_simplex(a.V.data[i])
            # alpha: project onto simplex
            client_dict.alpha.data -= config["lr_alpha"] * client_dict.alpha.grad
            client_dict.alpha.data = project_simplex(client_dict.alpha.data)

    return log


# ============================================================================
# Aggregation: per-k 3-way free-support barycenter
# ============================================================================
def aggregate_dictionaries(own: AtomDictionary,
                           peer1: AtomDictionary,
                           peer2: AtomDictionary,
                           config: Dict,
                           device: torch.device,
                           dtype: torch.dtype) -> AtomDictionary:
    """Per-k aggregation: for each k in 1..K, compute the free-support barycenter
    of (own.atom_k, peer1.atom_k, peer2.atom_k) with weights (1/3, 1/3, 1/3).

    The result is a new K-atom dictionary.  alpha is kept from `own` (private).
    Aggregation is non-differentiable: we don't need gradients here.
    """
    K = len(own.atoms)
    assert len(peer1.atoms) == K and len(peer2.atoms) == K
    w = torch.tensor([config["self_weight"],
                      (1.0 - config["self_weight"]) / 2.0,
                      (1.0 - config["self_weight"]) / 2.0],
                     device=device, dtype=dtype)
    # Renormalize for safety (should already sum to 1)
    w = w / w.sum()

    new_atoms: List[AtomGMM] = []
    for k in range(K):
        B = free_support_barycenter(
            gmms=[own.atoms[k], peer1.atoms[k], peer2.atoms[k]],
            weights=w,
            n_iters=config["barycenter_iters"],
            beta=config["beta_class"],
            var_floor=config["var_floor"],
            eps=config["eps"],
            init_idx=0,
            differentiable=False,
        )
        # Detach and wrap as a new atom (no grad in aggregated dictionary).
        new_atoms.append(AtomGMM(
            pi=B.pi.detach().clone(),
            mu=B.mu.detach().clone().requires_grad_(True),
            var=B.var.detach().clone().requires_grad_(True),
            V=B.V.detach().clone().requires_grad_(True),
        ))

    # alpha is private: keep own client's alpha (but re-create the tensor so
    # it's a fresh leaf for the next round's autograd).
    new_alpha = own.alpha.detach().clone().requires_grad_(True)
    return AtomDictionary(atoms=new_atoms, alpha=new_alpha)


def build_global_dictionary(client_dicts: Dict[str, AtomDictionary],
                            config: Dict,
                            device: torch.device,
                            dtype: torch.dtype) -> AtomDictionary:
    """Compute one explicit global dictionary from all final client dictionaries.

    For every atom index ``k``, take an equal-weight free-support Wasserstein
    barycenter of all clients' ``atom_k``. Client alpha vectors remain private;
    the stored global alpha is only their mean for backward compatibility.
    """
    names = list(client_dicts)
    if not names:
        raise ValueError("client_dicts 不能为空")

    K = client_dicts[names[0]].K
    weights = torch.full(
        (len(names),), 1.0 / len(names), device=device, dtype=dtype
    )
    global_atoms: List[AtomGMM] = []
    for k in range(K):
        barycenter = free_support_barycenter(
            gmms=[client_dicts[name].atoms[k] for name in names],
            weights=weights,
            n_iters=config["barycenter_iters"],
            beta=config["beta_class"],
            var_floor=config["var_floor"],
            eps=config["eps"],
            init_idx=0,
            differentiable=False,
        )
        global_atoms.append(barycenter.detach())

    mean_alpha = torch.stack([
        client_dicts[name].alpha.detach() for name in names
    ]).mean(dim=0)
    mean_alpha = mean_alpha / mean_alpha.sum()
    return AtomDictionary(atoms=global_atoms, alpha=mean_alpha)


# ============================================================================
# Diagnostics
# ============================================================================
def dictionary_divergence(d1: AtomDictionary, d2: AtomDictionary,
                          beta: float, var_floor: float,
                          eps: float) -> float:
    """Mean SMW2^2 distance between corresponding atoms of two dictionaries.

    Used to check pairwise convergence across clients.
    (Atom-index alignment is assumed; with shared init / per-k aggregation
    the k-th atom remains roughly comparable across clients.)
    """
    assert len(d1.atoms) == len(d2.atoms)
    K = len(d1.atoms)
    total = 0.0
    for k in range(K):
        with torch.no_grad():
            C = cost_matrix(d1.atoms[k].detach(), d2.atoms[k].detach(),
                            beta=beta, var_floor=var_floor)
            omega, _, _ = solve_ot_lp(
                C.cpu().numpy().astype(np.float64),
                d1.atoms[k].pi.detach().cpu().numpy().astype(np.float64),
                d2.atoms[k].pi.detach().cpu().numpy().astype(np.float64),
            )
            total += float((torch.as_tensor(omega, dtype=C.dtype) * C).sum())
    return total / max(1, K)


def max_pairwise_divergence(client_dicts: Dict[str, AtomDictionary],
                            beta: float, var_floor: float, eps: float) -> float:
    """Max SMW2^2 over all client pairs (measure of dictionary consensus)."""
    names = list(client_dicts)
    worst = 0.0
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            d = dictionary_divergence(client_dicts[names[i]],
                                      client_dicts[names[j]],
                                      beta=beta, var_floor=var_floor, eps=eps)
            worst = max(worst, d)
    return worst


def eval_dictionary_loss(client_dict: AtomDictionary, Q_local: AtomGMM,
                         config: Dict) -> float:
    """Compute SMW2^2(B_i(alpha_i, P_i), Q_i) without backprop."""
    with torch.no_grad():
        B = free_support_barycenter(
            gmms=client_dict.atoms,
            weights=client_dict.alpha,
            n_iters=config["barycenter_iters"],
            beta=config["beta_class"],
            var_floor=config["var_floor"],
            eps=config["eps"],
            init_idx=0,
            differentiable=False,
        )
        loss = smw2_sq_loss(B, Q_local,
                            beta=config["beta_class"],
                            var_floor=config["var_floor"],
                            eps=config["eps"])
    return float(loss.detach())


# ============================================================================
# Domain GMM loading
# ============================================================================
def load_domain_gmm(path: str, device: torch.device,
                    dtype: torch.dtype) -> AtomGMM:
    """Load one fitted client-domain GMM as a frozen target.

    The number of components may differ across clients because rare classes
    use fewer components and missing classes contribute no artificial component.
    Each observed component's class vector is its `component_onehot` row.
    Internal weights are the `effective_weights` (pi_c * alpha_k) renormalized.
    """
    with np.load(path, allow_pickle=False) as d:
        means = d["means"]                  # (K_client, D)
        variances = d["variances"]          # (K_client, D)
        onehot = d["component_onehot"]      # (K_client, n_class)
        eff_w = d["effective_weights"]      # (K_client,)
        if not (len(means) == len(variances) == len(onehot) == len(eff_w)):
            raise ValueError(f"域 GMM 字段长度不一致: {path}")
        if len(eff_w) == 0 or not np.isfinite(eff_w).all() or eff_w.sum() <= 0:
            raise ValueError(f"域 GMM 权重无效: {path}")
    # Renormalize eff_w to sum to 1 (it already does, but for safety)
    pi = eff_w / eff_w.sum()

    return AtomGMM.from_array(
        pi=pi, mu=means, var=variances, V=onehot,
        device=device, dtype=dtype, requires_grad=False,
    )


def load_client_domain_gmms(domain_gmm_dir: str,
                            device: torch.device,
                            dtype: torch.dtype
                            ) -> Tuple[Dict[str, AtomGMM], Dict[str, Dict]]:
    """Discover and load ``<root>/<client>/domain_gmm.npz`` files."""
    root = Path(domain_gmm_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"domain GMM 根目录不存在: {root}")
    paths = sorted(root.glob("*/domain_gmm.npz"), key=lambda p: p.parent.name)
    if not paths:
        raise FileNotFoundError(
            f"未在 {root} 下发现 <client>/domain_gmm.npz; "
            "请先运行 fit_domain_gmm.py"
        )

    domains: Dict[str, AtomGMM] = {}
    domain_info: Dict[str, Dict] = {}
    for path in paths:
        client_name = path.parent.name
        with np.load(path, allow_pickle=False) as d:
            if "client_name" in d:
                stored_name = str(d["client_name"].item())
                if stored_name != client_name:
                    raise ValueError(
                        f"客户端名不一致: 目录={client_name}, 文件={stored_name}"
                    )
            num_samples = int(d["num_samples"].item()) \
                if "num_samples" in d else -1

        if client_name in domains:
            raise ValueError(f"重复的客户端域 GMM: {client_name}")
        domain = load_domain_gmm(str(path), device, dtype)
        domains[client_name] = domain
        domain_info[client_name] = {
            "path": str(path),
            "num_samples": num_samples,
            "num_components": int(domain.pi.shape[0]),
        }
    return domains, domain_info


# ============================================================================
# DFL training loop
# ============================================================================
def run_defed(config: Dict) -> Dict:
    set_seed(config["seed"])
    if config["K"] < 1:
        raise ValueError("K must be at least 1")
    if config["C"] < config["num_classes"]:
        raise ValueError(
            "C (components per atom) must be >= num_classes so virtual "
            "sampling can represent every class"
        )
    save_dir = Path(config["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(config["device"])
    dtype = get_dtype(config["dtype"])

    Q_domains, domain_info = load_client_domain_gmms(
        config["domain_gmm_dir"], device, dtype
    )
    client_names = list(Q_domains)
    config["domain_gmm_paths"] = {
        name: domain_info[name]["path"] for name in client_names
    }
    configured_num_clients = config["num_clients"]
    config["num_clients"] = len(client_names)
    if configured_num_clients != len(client_names):
        print(f"[信息] num_clients={configured_num_clients} 已按发现的域 GMM 数量 "
              f"修正为 {len(client_names)}")
    if config["neighbors_per_client"] >= len(client_names):
        raise ValueError(
            "neighbors_per_client 必须小于发现的客户端数量: "
            f"{config['neighbors_per_client']} >= {len(client_names)}"
        )
    for name, domain in Q_domains.items():
        if domain.mu.shape[1] != config["feature_dim"]:
            raise ValueError(
                f"{name} feature_dim={domain.mu.shape[1]}, "
                f"配置为 {config['feature_dim']}"
            )
        if domain.V.shape[1] != config["num_classes"]:
            raise ValueError(
                f"{name} num_classes={domain.V.shape[1]}, "
                f"配置为 {config['num_classes']}"
            )

    print("=" * 70)
    print("DeFed-GMM-DaDiL  (Decentralized Federated GMM Dataset Dictionary Learning)")
    print("=" * 70)
    print(f"  domain_gmms : {config['domain_gmm_dir']}")
    print(f"  save_dir    : {save_dir}")
    print(f"  K (atoms)   : {config['K']}")
    print(f"  C (comp/atom): {config['C']}")
    print(f"  num_classes : {config['num_classes']}")
    print(f"  feature_dim : {config['feature_dim']}")
    print(f"  num_clients : {config['num_clients']}")
    print(f"  rounds      : {config['rounds']}")
    print(f"  inner_steps : {config['inner_steps']}")
    print(f"  beta_class  : {config['beta_class']}")
    print(f"  device/dtype: {device} / {dtype}")

    # 1. Each client owns one frozen local target Q_i.
    print("\n[1] Load per-client domain GMMs (frozen local targets Q_i)")
    for name in client_names:
        info = domain_info[name]
        print(f"  {name}: {Q_domains[name]}, samples={info['num_samples']}, "
              f"components={info['num_components']}")

    # 2. Initialize each client's atom dictionary
    print(f"\n[2] Initialize {config['num_clients']} client dictionaries")
    client_dicts = init_all_clients(config, client_names)
    for name, d in client_dicts.items():
        print(f"  {name}: {d.atoms[0]}  alpha={d.alpha.detach().cpu().numpy().round(3)}")

    # 3. DFL training loop
    print(f"\n[3] DFL training ({config['rounds']} rounds)")
    print("-" * 70)
    rng = random.Random(config["seed"])
    history = []
    initial_loss = float("nan")
    for r in range(config["rounds"]):
        round_num = r + 1

        # (a) Local update per client
        local_logs = {}
        for name in client_dicts:
            log = local_update(
                client_dicts[name], Q_domains[name], config, device, dtype
            )
            local_logs[name] = log

        # (b) Pick 2 random peers per client (without replacement)
        # (Deterministic given the rng; we sample one topology per round.)
        names = list(client_dicts)
        topology: Dict[str, List[str]] = {}
        for name in names:
            peers = rng.sample([n for n in names if n != name],
                               config["neighbors_per_client"])
            topology[name] = peers

        # (c) Per-k aggregation
        # IMPORTANT: read each peer's post-local-update state into a snapshot,
        # so all clients see the same round snapshot during aggregation.
        snapshot = {name: d.clone() for name, d in client_dicts.items()}
        new_dicts: Dict[str, AtomDictionary] = {}
        for name in names:
            peers = topology[name]
            new_dicts[name] = aggregate_dictionaries(
                own=snapshot[name],
                peer1=snapshot[peers[0]],
                peer2=snapshot[peers[1]],
                config=config, device=device, dtype=dtype,
            )
        client_dicts = new_dicts

        # (d) Diagnostics
        mean_loss = float(np.mean([local_logs[n]["losses"][-1] for n in names]))
        max_div = max_pairwise_divergence(
            client_dicts, beta=config["beta_class"],
            var_floor=config["var_floor"], eps=config["eps"])
        if round_num == 1:
            initial_loss = mean_loss
        round_record = {
            "round": round_num,
            "mean_loss": mean_loss,
            "max_pairwise_div": max_div,
            "topology": topology,
        }
        history.append(round_record)
        if round_num % config["eval_every"] == 0 or round_num == 1 or \
           round_num == config["rounds"]:
            print(f"  R{round_num:03d}/{config['rounds']}: "
                  f"loss={mean_loss:.4f}  max_pairwise_div={max_div:.4f}  "
                  f"(init_loss={initial_loss:.4f})")
            print(f"    topology: " + "; ".join(
                f"{n}->{','.join(ps)}" for n, ps in topology.items()))

        # Early stop on consensus
        if max_div < config["converge_tol"] and round_num >= 5:
            print(f"\n  >> Converged (max_pairwise_div < {config['converge_tol']}) "
                  f"at round {round_num}")
            break

    # 4. Final evaluation
    print("\n" + "=" * 70)
    print("Final evaluation")
    print("=" * 70)
    final_losses = {name: eval_dictionary_loss(
                        client_dicts[name], Q_domains[name], config
                    )
                    for name in client_dicts}
    final_divs = {}
    names = list(client_dicts)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            d = dictionary_divergence(client_dicts[names[i]],
                                      client_dicts[names[j]],
                                      beta=config["beta_class"],
                                      var_floor=config["var_floor"],
                                      eps=config["eps"])
            final_divs[f"{names[i]}<->{names[j]}"] = d
    for name, l in final_losses.items():
        print(f"  {name}: final_loss={l:.4f}  alpha={client_dicts[name].alpha.detach().cpu().numpy().round(3)}")
    print(f"\n  Mean final loss: {np.mean(list(final_losses.values())):.4f}")
    print(f"  Max pairwise divergence: {max(final_divs.values()):.4f}")
    print(f"  Mean pairwise divergence: {np.mean(list(final_divs.values())):.4f}")

    # 5. Build one actual global atom dictionary from all final clients.
    print("\n[5] Build global atom dictionary (all-client Wasserstein barycenter)")
    global_dict = build_global_dictionary(client_dicts, config, device, dtype)
    global_personalized_losses = {
        name: eval_dictionary_loss(
            AtomDictionary(global_dict.atoms, client_dicts[name].alpha),
            Q_domains[name], config,
        )
        for name in names
    }
    for name, loss in global_personalized_losses.items():
        print(f"  {name}: global atoms + private alpha loss={loss:.4f}")
    print(f"  Mean global personalized loss: "
          f"{np.mean(list(global_personalized_losses.values())):.4f}")

    save_dict = {
        "format_version": np.asarray(2, dtype=np.int64),
        "alpha": global_dict.alpha.detach().cpu().numpy(),
        "global_alpha": global_dict.alpha.detach().cpu().numpy(),
        "global_aggregation": np.asarray("uniform_wasserstein_barycenter"),
        "client_names": np.asarray(names),
        "K": config["K"], "C": config["C"],
        "num_classes": config["num_classes"], "feature_dim": config["feature_dim"],
    }
    for k in range(config["K"]):
        a = global_dict.atoms[k]
        save_dict[f"atom{k}_pi"] = a.pi.detach().cpu().numpy()
        save_dict[f"atom{k}_mu"] = a.mu.detach().cpu().numpy()
        save_dict[f"atom{k}_var"] = a.var.detach().cpu().numpy()
        save_dict[f"atom{k}_V"] = a.V.detach().cpu().numpy()

    # Save all clients' dictionaries too
    for name in names:
        d = client_dicts[name]
        save_dict[f"{name}_alpha"] = d.alpha.detach().cpu().numpy()
        for k in range(config["K"]):
            a = d.atoms[k]
            save_dict[f"{name}_atom{k}_pi"] = a.pi.detach().cpu().numpy()
            save_dict[f"{name}_atom{k}_mu"] = a.mu.detach().cpu().numpy()
            save_dict[f"{name}_atom{k}_var"] = a.var.detach().cpu().numpy()
            save_dict[f"{name}_atom{k}_V"] = a.V.detach().cpu().numpy()

    out_npz = save_dir / "defed_dictionary.npz"
    np.savez_compressed(out_npz, **save_dict)
    print(f"\n  Saved global dictionary: {out_npz}  "
          f"({os.path.getsize(out_npz)/1024:.1f} KB)")

    # Save config + history
    with open(save_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2, default=str)
    with open(save_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2, default=str)
    print(f"  Saved config + history to: {save_dir}")

    return {
        "client_dicts": client_dicts,
        "global_dict": global_dict,
        "history": history,
        "final_losses": final_losses,
        "global_personalized_losses": global_personalized_losses,
        "final_divs": final_divs,
    }


# ============================================================================
# CLI
# ============================================================================
def load_config(args) -> Dict:
    config = DEFAULT_CONFIG.copy()
    if args.config and os.path.exists(args.config):
        with open(args.config, "r", encoding="utf-8") as f:
            config.update(json.load(f))
    # CLI overrides
    for key in ("domain_gmm_dir", "save_dir", "K", "C", "num_classes",
                "feature_dim", "num_clients", "rounds", "inner_steps",
                "lr_mu", "lr_var", "lr_V", "lr_pi", "lr_alpha",
                "grad_clip", "entropy_reg",
                "barycenter_iters", "beta_class", "var_floor", "var_init",
                "neighbors_per_client", "self_weight", "eval_every",
                "converge_tol", "seed", "device", "dtype"):
        v = getattr(args, key, None)
        if v is not None:
            config[key] = v
    return config


def main():
    parser = argparse.ArgumentParser(
        description="DeFed-GMM-DaDiL: decentralized federated GMM dictionary learning"
    )
    parser.add_argument("--config", type=str, default=None, help="JSON config path")
    parser.add_argument(
        "--domain-gmm-dir", type=str, default=None,
        help="包含 <client>/domain_gmm.npz 的根目录",
    )
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--K", type=int, default=None, help="atoms per dictionary")
    parser.add_argument("--C", type=int, default=None, help="components per atom")
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--feature-dim", type=int, default=None)
    parser.add_argument("--num-clients", type=int, default=None)
    parser.add_argument("--rounds", type=int, default=None, help="DFL rounds")
    parser.add_argument("--inner-steps", type=int, default=None,
                        help="local gradient steps per round")
    parser.add_argument("--lr-mu", type=float, default=None)
    parser.add_argument("--lr-var", type=float, default=None)
    parser.add_argument("--lr-V", type=float, default=None)
    parser.add_argument("--lr-pi", type=float, default=None)
    parser.add_argument("--lr-alpha", type=float, default=None)
    parser.add_argument("--grad-clip", type=float, default=None,
                        help="max grad norm per parameter (0 = no clipping)")
    parser.add_argument("--entropy-reg", type=float, default=None,
                        help="weight on alpha entropy regularizer (0 = no reg)")
    parser.add_argument("--barycenter-iters", type=int, default=None)
    parser.add_argument("--beta-class", type=float, default=None)
    parser.add_argument("--var-floor", type=float, default=None)
    parser.add_argument("--var-init", type=float, default=None)
    parser.add_argument("--neighbors-per-client", type=int, default=None)
    parser.add_argument("--self-weight", type=float, default=None)
    parser.add_argument("--eval-every", type=int, default=None)
    parser.add_argument("--converge-tol", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None,
                        choices=["cpu", "cuda"])
    parser.add_argument("--dtype", type=str, default=None,
                        choices=["float32", "float64"])
    args = parser.parse_args()
    config = load_config(args)
    try:
        start = time.time()
        run_defed(config)
        print(f"\nTotal elapsed: {time.time() - start:.1f} s")
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
