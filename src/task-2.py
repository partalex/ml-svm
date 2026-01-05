import os
from typing import Callable

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np

from src.shared import ArrayF, ArrayI, load_multiclass_csv, INPUT_FILE, OUT_DIR

PATH_TASK_2 = "task-2"


# ============================================================
# SMO algorithm (dual SVM)
# ============================================================

def train_svm_dual_cvxpy(
        features: ArrayF,
        labels: ArrayI,
        regularization_param: float,
        kernel_func: Callable[[ArrayF, ArrayF], float | np.ndarray],
        solver: str = "OSQP",
) -> np.ndarray:
    """
    Trains a dual SVM using CVXPY.
    Args:
        features (np.ndarray): Feature matrix of shape (n_samples, n_features).
        labels (np.ndarray): Label vector of shape (n_samples,).
        regularization_param (float): Regularization parameter (C).
        kernel_func: Kernel function to compute the kernel matrix.
        solver (str): CVXPY solver to use.
    Returns:
        np.ndarray: Dual coefficients (alpha) of shape (n_samples,).
    """
    n = len(labels)
    kernel_matrix = np.array(
        [[kernel_func(features[i], features[j]) for j in range(n)] for i in range(n)],
        dtype=np.float64)
    p_matrix = np.outer(labels, labels) * kernel_matrix
    # ensure symmetry and add jitter
    p_matrix = (p_matrix + p_matrix.T) / 2.0
    p_matrix += 1e-8 * np.eye(n)

    alpha = cp.Variable(n)
    p_const = cp.Constant(p_matrix)
    # wrap with psd_wrap to avoid CVXPY PSD certification failures
    objective = 0.5 * cp.quad_form(alpha, cp.psd_wrap(p_const)) - cp.sum(alpha)
    constraints = [alpha >= 0, alpha <= regularization_param, labels @ alpha == 0]
    problem = cp.Problem(cp.Minimize(objective), constraints)
    problem.solve(solver=solver, warm_start=True)
    return np.maximum(0.0, (alpha.value if alpha.value is not None else np.zeros(n)))


def plot_hyperparameter_search(
        c_range: list[float],
        gamma_range: list[float],
        grid_losses: np.ndarray,
) -> None:
    """
    Plots a heatmap for hyperparameter search results.
    Args:
        c_range (list of float): List of C values (penalty parameters).
        gamma_range (list of float): List of gamma values.
        grid_losses (2D array): Average hinge losses for each (C, gamma) pair
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(grid_losses, interpolation='nearest', cmap=plt.cm.hot)
    plt.xlabel('Sigma')
    plt.ylabel('C - Penalty Parameter')
    plt.colorbar(label='Avg Hinge Loss')
    plt.xticks(np.arange(len(gamma_range)), labels=[str(g) for g in gamma_range], rotation=45)
    plt.yticks(np.arange(len(c_range)), labels=[str(c) for c in c_range])
    plt.title('Hyperparameter Selection (C and Gamma)')
    plt.savefig(os.path.join(OUT_DIR, PATH_TASK_2, "hyperparameter_search.png"), bbox_inches='tight')
    plt.close()
    plt.show()


def plot_nonlinear_svm(
        features: np.ndarray,
        labels: np.ndarray,
        alpha: np.ndarray, bias: float,
        kernel_func,
        sv_indices: np.ndarray,
        slack: np.ndarray,
        label_decimals: int = 6,
) -> None:
    """
    Plots the decision boundary of a non-linear SVM along with support vectors.
    Args:
        features (np.ndarray): Feature matrix of shape (n_samples, n_features).
        labels (np.ndarray): Label vector of shape (n_samples,).
        alpha (np.ndarray): Dual coefficients of shape (n_samples,).
        bias (float): Bias term.
        kernel_func: Kernel function used in SVM.
        sv_indices (np.ndarray): Indices of support vectors.
        slack (np.ndarray): Slack variables of shape (n_samples,).
    """
    plt.figure(figsize=(8, 6))

    # 1. Plot all points
    for label, color in [(-1, "red"), (1, "blue")]:
        mask = labels == label
        plt.scatter(features[mask, 0], features[mask, 1], c=color, label=f"Class {label}")

    # 2. Creating a mesh grid for contour plotting
    x_min, x_max = features[:, 0].min() - 0.5, features[:, 0].max() + 0.5
    y_min, y_max = features[:, 1].min() - 0.5, features[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

    # Calculating predictions for each point in the grid
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    decision_vals = np.zeros(grid_points.shape[0])

    # Dual prediction: f(x) = sum(alpha_i * y_i * K(x_i, x)) + b
    for i in range(len(alpha)):
        if alpha[i] > 1e-5:
            decision_vals += alpha[i] * labels[i] * kernel_func(features[i], grid_points)
    decision_vals += bias
    decision_vals = decision_vals.reshape(xx.shape)

    # 3. Plot decision boundary (Z=0) and margins (Z=1, Z=-1)
    plt.contour(xx, yy, decision_vals, levels=[-1, 0, 1], colors=['gray', 'black', 'gray'],
                linestyles=['--', '-', '--'], alpha=0.5)

    # 4. Marking support vectors (alpha > 0)
    plt.scatter(features[sv_indices, 0], features[sv_indices, 1],
                facecolors="none", edgecolors="black", s=150, label="Support Vectors")

    # Annotate first few support vectors with higher precision / scientific format
    for i in sv_indices[:5]:
        val = float(slack[i])
        if 0 < abs(val) < 10 ** (-label_decimals):
            text = f"{val:.2e}"  # scientific for very small values
        else:
            text = f"{val:.{label_decimals}f}"  # fixed precision
        plt.text(features[i, 0], features[i, 1] + 0.05, text, fontsize=9, fontweight='bold')

    plt.title(f"Dual SVM (RBF Kernel) - Support Vectors: {len(sv_indices)}")
    plt.legend()
    plt.savefig(os.path.join(OUT_DIR, PATH_TASK_2, f"nonlinear_svm_sv{len(sv_indices)}.png"), bbox_inches='tight')
    plt.show()
    plt.close()


def rbf_kernel(sigma: float) -> Callable[[np.ndarray, np.ndarray], float | np.ndarray]:
    """
    Radial Basis Function (RBF) kernel.
    Args:
        sigma (float): Parameter for the RBF kernel.
    Returns:
        Callable: Function that computes the RBF kernel between two input vectors.
    """

    def kernel(x1: np.ndarray, x2: np.ndarray) -> float | np.ndarray:
        """
        Computes the RBF kernel between x1 and x2.
        Args:
            x1 (np.ndarray): First input vector of shape (d,).
            x2 (np.ndarray): Second input vector(s) of shape (d or (m, d)).
        Returns:
            float or np.ndarray: RBF kernel value(s).
        Raises:
            ValueError: If dimensions of x1 and x2 do not match.
        """
        x1 = np.asarray(x1, dtype=np.float64).ravel()
        x2 = np.asarray(x2, dtype=np.float64)

        # If x2 is a 1-D array exactly matching x1 -> single point
        if x2.ndim == 1 and x2.size == x1.size:
            diff = x1 - x2
            return float(np.exp(-sigma * np.dot(diff, diff)))

        # Now expect x2 to be 2-D: (m, d)
        if x2.ndim == 2:
            if x2.shape[1] != x1.size:
                raise ValueError("dimension mismatch between x1 and rows of x2")
            diff = x2 - x1  # shape (m, d)
            sq = np.einsum('ij,ij->i', diff, diff)
            return np.exp(-sigma * sq)  # shape (m,)

        raise ValueError("x2 must be either a 1-D or 2-D array")

    return kernel


class BestParams:
    c_val: float = 0.0
    sigma: float = 0.0
    alpha: np.ndarray = np.empty(0)
    bias: float = 0.0
    slack: np.ndarray = np.empty(0)


best_params = BestParams()

if __name__ == "__main__":

    features, labels = load_multiclass_csv(INPUT_FILE)
    c_values = [0.01, 0.1, 1.0, 10.0, 100.0]

    sigmas = [0.1, 0.5, 1.0, 2.0, 5.0]
    loss_grid = np.zeros((len(c_values), len(sigmas)))

    best_dual_loss = float('inf')

    for i, c_val in enumerate(c_values):
        for j, sigma in enumerate(sigmas):
            current_kernel = rbf_kernel(sigma=sigma)

            # 1. Training - the function returns only alpha
            alpha = train_svm_dual_cvxpy(features, labels, c_val, kernel_func=current_kernel)

            # 2. Calculating bias (intercept)
            # Bias is calculated using Support Vectors that are on the margin (0 < alpha < C)
            margin_sv = np.where((alpha > 1e-5) & (alpha < c_val - 1e-5))[0]

            if len(margin_sv) > 0:
                # b = y_k - sum(alpha_i * y_i * K(x_i, x_k))
                idx = margin_sv[0]
                # Vectorized calculation of prediction for a single point
                ks = np.array([current_kernel(features[n], features[idx]) for n in range(len(labels))])
                pred_k = np.sum(alpha * labels * ks)
                bias_dual = float(labels[idx] - pred_k)
            else:
                bias_dual = 0.0

            # 3. Calculating Hinge Loss (Vectorized)
            # First, we compute the Kernel matrix for the entire dataset to avoid a triple loop
            # If the dataset is huge, this can be optimized, but for svmData.csv it's fine
            dual_slack = np.zeros(len(labels))

            # 3. Calculating Hinge Loss (Vectorized) f(X) = K(X, X_sv) @ (alpha_sv * y_sv) + b
            # For simplicity, we use a loop that is faster than your previous one:
            for k in range(len(labels)):
                ks = np.array([current_kernel(features[n], features[k]) for n in range(len(labels))])
                pred = np.sum(alpha * labels * ks) + bias_dual
                dual_slack[k] = max(0.0, 1.0 - labels[k] * pred)

            avg_loss = float(np.mean(dual_slack))
            loss_grid[i, j] = avg_loss

            if avg_loss < best_dual_loss:
                best_dual_loss = avg_loss
                best_params.c_val = c_val
                best_params.sigma = sigma
                best_params.alpha = alpha
                best_params.bias = bias_dual
                best_params.slack = dual_slack

            print(f"C={c_val:<5} gamma={sigma:<4} | Loss: {avg_loss:.4f}")

    print(f"\nBest dual parameters: C={best_params.c_val}, gamma={best_params.sigma}")

    print(f"\nBest dual parameters: C={best_params.c_val}, gamma={best_params.sigma}")

    # 1. Heatmap for C values and Sigmas selection
    plot_hyperparameter_search(c_values, sigmas, loss_grid)

    # 2. Non-linear boundary and support vectors
    sv_indices = np.where(best_params.alpha > 1e-5)[0]
    plot_nonlinear_svm(
        features,
        labels,
        best_params.alpha,
        best_params.bias,
        rbf_kernel(sigma=best_params.sigma),
        sv_indices,
        best_params.slack,
    )
