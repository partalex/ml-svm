import os

import numpy as np
import matplotlib.pyplot as plt

from src.shared import INPUT_FILE, load_multiclass_csv, ArrayF, ArrayI, OUT_DIR

# ============================================================
# Hinge loss
# ============================================================

PATH_TASK_1 = "task-1"


def hinge_loss(
        features: ArrayF,
        labels: ArrayI,
        weight: ArrayF,
        bias: float,
) -> ArrayF:
    """
    Vectorized hinge loss for all samples.
    Args:
        features (ArrayF): Feature matrix of shape (n_samples, n_features).
        labels (ArrayI): Label vector of shape (n_samples,) with integer labels (-1 or +1).
        weight (ArrayF): Weight vector of shape (n_features,).
        bias (float): Bias term (can be python float or numpy float).
    Returns:
        loss (ArrayF): Hinge loss vector of shape (n_samples,).
    """
    return np.maximum(0.0, 1.0 - labels * (features @ weight + float(bias)))


# ============================================================
# PRIMAL SVM – SGD (hinge loss)
# ============================================================

def train_linear_svm_primal_sgd(
        features: ArrayF,
        labels: ArrayI,
        reg_param: float,
        learning_rate: float = 1e-2,
        epochs: int = 2000
) -> tuple[ArrayF, float, ArrayF]:
    """
    Train a linear SVM using the primal formulation and SGD.
    Args:
        features (ArrayF): Feature matrix of shape (n_samples, n_features).
        labels (ArrayI): Label vector of shape (n_samples,) with integer labels (-1 or +1).
        reg_param (float): Regularization parameter.
        learning_rate (float): Learning rate.
        epochs (int): Number of epochs.
    Returns:
        weight (ArrayF): Weight vector of shape (n_features,).
        bias (float): Bias term (python float).
        xi (ArrayF): Slack variables (hinge loss) of shape (n_samples,).
    """
    n_samples, n_features = features.shape

    weight: ArrayF = np.zeros(n_features, dtype=np.float64)
    bias: float = 0.0

    for _ in range(epochs):
        for i in range(n_samples):
            margin = labels[i] * (features[i] @ weight + bias)
            if margin < 1:
                weight -= learning_rate * (weight - reg_param * labels[i] * features[i])
                bias += learning_rate * reg_param * labels[i]
            else:
                weight -= learning_rate * weight

    xi = hinge_loss(features, labels, weight, bias)
    return weight, float(bias), xi


# ============================================================
# Plotting
# ============================================================

def plot_linear_svm(
        features: ArrayF,
        labels: ArrayI,
        weight: ArrayF,
        bias: float,
        xi: ArrayF
) -> None:
    """
    Plot the linear SVM decision boundary and support vectors.
    Args:
        features (ArrayF): Feature matrix of shape (n_samples, n_features).
        labels (ArrayI): Label vector of shape (n_samples,).
        weight (ArrayF): Weight vector of shape (n_features,).
        bias (float): Bias term.
        xi (ArrayF): Slack variables of shape (n_samples,).
    """
    plt.figure(figsize=(8, 6))

    for label, color in [(-1, "red"), (1, "blue")]:
        mask = labels == label
        plt.scatter(features[mask, 0], features[mask, 1], c=color, label=f"Class {label}")

    xx = np.linspace(features[:, 0].min() - 1, features[:, 0].max() + 1, 200)
    yy = -(weight[0] * xx + bias) / weight[1]
    plt.plot(xx, yy, "k--", label="Decision boundary")

    sv = np.where(xi > 1e-3)[0]
    plt.scatter(features[sv, 0], features[sv, 1],
                facecolors="none", edgecolors="black", s=120, label="Support vectors")

    for i in sv[:5]:
        plt.text(features[i, 0], features[i, 1], f"{xi[i]:.2f}", fontsize=9)

    plt.legend()
    plt.title("Linear SVM – Primal (SGD)")
    plt.savefig(os.path.join(OUT_DIR, PATH_TASK_1, "linear_svm_primal_sgd.png"))
    plt.show()
    plt.close()


# ============================================================
# C Selection Plotting
# ============================================================

def plot_c_selection(
        c_values: list[float],
        losses: list[float],
) -> None:
    """
    Plot average hinge loss vs C values.
    Args:
        c_values (list[float]): List of C values.
        losses (list[float]): Corresponding average hinge losses.
    """
    plt.figure()
    plt.plot(c_values, losses, marker="o")
    plt.xscale("log")
    plt.xlabel("C")
    plt.ylabel("Avg hinge loss")
    plt.title("C selection")
    plt.savefig(os.path.join(OUT_DIR, PATH_TASK_1, "c_selection.png"))
    plt.show()
    plt.close()


if __name__ == "__main__":
    features, labels = load_multiclass_csv(INPUT_FILE)
    c_values = [5.0, 6.0, 7.5, 8.0, 9.5, 10.0]
    print(f"Testing C values: {c_values}")

    avg_losses_linear = []
    trained_models_linear = {}

    for c_val in c_values:
        w, b, slack = train_linear_svm_primal_sgd(features, labels, c_val)
        avg_losses_linear.append(float(np.mean(slack)))
        trained_models_linear[c_val] = (w, b, slack)

    best_c_lin = c_values[np.argmin(avg_losses_linear)]
    w_lin, b_lin, slack_lin = trained_models_linear[best_c_lin]

    plot_linear_svm(features, labels, w_lin, b_lin, slack_lin)
    plot_c_selection(c_values, avg_losses_linear)
