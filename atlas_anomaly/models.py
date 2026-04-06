from __future__ import annotations

import pickle
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import RobustScaler

from atlas_anomaly.constants import MODEL_FEATURES


def prepare_matrix(frame: pd.DataFrame, feature_columns: Optional[List[str]] = None) -> np.ndarray:
    columns = feature_columns if feature_columns is not None else MODEL_FEATURES
    matrix = frame.loc[:, columns].to_numpy(dtype=float)
    if np.isnan(matrix).any():
        raise ValueError("Input matrix contains NaN values. Check the feature-building stage.")
    return matrix


def train_isolation_forest(x_train: np.ndarray, random_state: int = 42) -> IsolationForest:
    model = IsolationForest(
        n_estimators=300,
        max_samples="auto",
        contamination="auto",
        random_state=random_state,
    )
    model.fit(x_train)
    return model


def score_isolation_forest(model: IsolationForest, x: np.ndarray) -> np.ndarray:
    return -model.score_samples(x)


def train_knn(x_train: np.ndarray, n_neighbors: int = 10) -> NearestNeighbors:
    k = min(n_neighbors, len(x_train))
    model = NearestNeighbors(n_neighbors=max(k, 1))
    model.fit(x_train)
    return model


def score_knn(model: NearestNeighbors, x: np.ndarray) -> np.ndarray:
    distances, _ = model.kneighbors(x)
    return distances.mean(axis=1)


def train_autoencoder(x_train: np.ndarray, random_state: int = 42) -> MLPRegressor:
    early_stopping = len(x_train) >= 100
    model = MLPRegressor(
        hidden_layer_sizes=(32, 12, 32),
        activation="relu",
        solver="adam",
        max_iter=400,
        random_state=random_state,
        early_stopping=early_stopping,
        validation_fraction=0.15 if early_stopping else 0.0,
    )
    model.fit(x_train, x_train)
    return model


def score_autoencoder(model: MLPRegressor, x: np.ndarray) -> np.ndarray:
    reconstruction = model.predict(x)
    return ((x - reconstruction) ** 2).mean(axis=1)


def validation_threshold(scores: np.ndarray, false_positive_rate: float) -> float:
    quantile = 1.0 - false_positive_rate
    return float(np.quantile(scores, quantile))


def save_bundle(path: Union[str, Path], bundle: dict) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        pickle.dump(bundle, handle)


def load_bundle(path: Union[str, Path]) -> dict:
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def bundle_scores(bundle: dict, frame: pd.DataFrame) -> np.ndarray:
    feature_columns = bundle["feature_columns"]
    x = prepare_matrix(frame, feature_columns)
    x_scaled = bundle["scaler"].transform(x)
    score_fn = bundle["score_function"]
    return score_fn(bundle["model"], x_scaled)


def build_training_bundle(
    model_name: str,
    model,
    scaler: RobustScaler,
    feature_columns: list[str],
    threshold: float,
    score_function,
    train_rows: int,
    val_rows: int,
    test_rows: int,
) -> dict:
    return {
        "model_name": model_name,
        "model": model,
        "scaler": scaler,
        "feature_columns": feature_columns,
        "threshold": threshold,
        "score_function": score_function,
        "train_rows": train_rows,
        "val_rows": val_rows,
        "test_rows": test_rows,
    }


def fit_all_baselines(
    train_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
    feature_columns: Optional[List[str]] = None,
    false_positive_rate: float = 0.01,
    random_state: int = 42,
) -> dict[str, dict]:
    columns = feature_columns if feature_columns is not None else MODEL_FEATURES
    scaler = RobustScaler()
    x_train = scaler.fit_transform(prepare_matrix(train_frame, columns))
    x_val = scaler.transform(prepare_matrix(val_frame, columns))

    bundles = {}

    iforest = train_isolation_forest(x_train, random_state=random_state)
    iforest_scores = score_isolation_forest(iforest, x_val)
    bundles["iforest"] = build_training_bundle(
        model_name="iforest",
        model=iforest,
        scaler=scaler,
        feature_columns=columns,
        threshold=validation_threshold(iforest_scores, false_positive_rate),
        score_function=score_isolation_forest,
        train_rows=len(train_frame),
        val_rows=len(val_frame),
        test_rows=0,
    )

    knn = train_knn(x_train, n_neighbors=10)
    knn_scores = score_knn(knn, x_val)
    bundles["knn"] = build_training_bundle(
        model_name="knn",
        model=knn,
        scaler=scaler,
        feature_columns=columns,
        threshold=validation_threshold(knn_scores, false_positive_rate),
        score_function=score_knn,
        train_rows=len(train_frame),
        val_rows=len(val_frame),
        test_rows=0,
    )

    autoencoder = train_autoencoder(x_train, random_state=random_state)
    autoencoder_scores = score_autoencoder(autoencoder, x_val)
    bundles["autoencoder"] = build_training_bundle(
        model_name="autoencoder",
        model=autoencoder,
        scaler=scaler,
        feature_columns=columns,
        threshold=validation_threshold(autoencoder_scores, false_positive_rate),
        score_function=score_autoencoder,
        train_rows=len(train_frame),
        val_rows=len(val_frame),
        test_rows=0,
    )

    return bundles


def compare_scores(normal_scores: np.ndarray, anomaly_scores: np.ndarray, threshold: float) -> dict:
    labels = np.concatenate(
        [np.zeros(len(normal_scores), dtype=int), np.ones(len(anomaly_scores), dtype=int)]
    )
    scores = np.concatenate([normal_scores, anomaly_scores])
    return {
        "auc": float(roc_auc_score(labels, scores)),
        "normal_mean_score": float(np.mean(normal_scores)),
        "anomaly_mean_score": float(np.mean(anomaly_scores)),
        "normal_flag_rate": float(np.mean(normal_scores >= threshold)),
        "anomaly_flag_rate": float(np.mean(anomaly_scores >= threshold)),
    }
