import csv
import numpy as np
import numpy.typing as npt

INPUT_FILE: str = "../res/svmData.csv"
OUT_DIR: str = "../out/"

ArrayF = npt.NDArray[np.floating]
ArrayI = npt.NDArray[np.integer]


def load_multiclass_csv(path: str) -> tuple[ArrayF, ArrayI]:
    """
    Load a classification dataset from a CSV file.
    Args:
        path (str): Path to the CSV file.
    Returns:
        features (ArrayF): Feature matrix of shape (n_samples, n_features).
        labels (ArrayI): Label vector of shape (n_samples,).
    """
    data: list[list[float]] = []

    with open(path, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            data.append([float(x) for x in row])

    data_np: np.ndarray = np.asarray(data, dtype=np.float64)

    features: np.ndarray = data_np[:, :-1].astype(np.float64)
    labels: np.ndarray = data_np[:, -1].astype(np.int64)

    return features, labels
