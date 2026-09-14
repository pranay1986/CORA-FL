from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import DatasetBundle


@dataclass
class SoftmaxRegression:
    n_features: int
    n_classes: int
    l2: float

    @property
    def parameter_count(self) -> int:
        return (self.n_features + 1) * self.n_classes

    def initial_parameters(self) -> np.ndarray:
        return np.zeros(self.parameter_count, dtype=np.float64)

    def _matrix(self, theta: np.ndarray) -> np.ndarray:
        return theta.reshape(self.n_features + 1, self.n_classes)

    def loss_grad(self, theta: np.ndarray, x: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
        weights = self._matrix(theta)
        logits = x @ weights[:-1] + weights[-1]
        logits -= logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        probabilities = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        sample_count = max(len(y), 1)
        loss = -np.log(np.clip(probabilities[np.arange(len(y)), y], 1e-15, 1.0)).mean()
        loss += 0.5 * self.l2 * float(np.sum(weights[:-1] ** 2))
        residual = probabilities
        residual[np.arange(len(y)), y] -= 1.0
        residual /= sample_count
        gradient = np.empty_like(weights)
        gradient[:-1] = x.T @ residual + self.l2 * weights[:-1]
        gradient[-1] = residual.sum(axis=0)
        return float(loss), gradient.ravel()

    def predict(self, theta: np.ndarray, x: np.ndarray) -> np.ndarray:
        weights = self._matrix(theta)
        return np.argmax(x @ weights[:-1] + weights[-1], axis=1)

    def metrics(self, theta: np.ndarray, x: np.ndarray, y: np.ndarray) -> dict[str, float]:
        loss, gradient = self.loss_grad(theta, x, y)
        prediction = self.predict(theta, x)
        return {"loss": loss, "accuracy": float(np.mean(prediction == y)), "mae": float("nan"), "rmse": float("nan"), "gradient_norm": float(np.linalg.norm(gradient))}


@dataclass
class LinearRegression:
    n_features: int
    n_outputs: int
    l2: float

    @property
    def parameter_count(self) -> int:
        return (self.n_features + 1) * self.n_outputs

    def initial_parameters(self) -> np.ndarray:
        return np.zeros(self.parameter_count, dtype=np.float64)

    def _matrix(self, theta: np.ndarray) -> np.ndarray:
        return theta.reshape(self.n_features + 1, self.n_outputs)

    def loss_grad(self, theta: np.ndarray, x: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
        weights = self._matrix(theta)
        prediction = x @ weights[:-1] + weights[-1]
        residual = prediction - y
        sample_count = max(len(y), 1)
        loss = 0.5 * float(np.mean(residual**2)) + 0.5 * self.l2 * float(np.sum(weights[:-1] ** 2))
        scale = 1.0 / (sample_count * self.n_outputs)
        gradient = np.empty_like(weights)
        gradient[:-1] = scale * (x.T @ residual) + self.l2 * weights[:-1]
        gradient[-1] = scale * residual.sum(axis=0)
        return loss, gradient.ravel()

    def predict(self, theta: np.ndarray, x: np.ndarray) -> np.ndarray:
        weights = self._matrix(theta)
        return x @ weights[:-1] + weights[-1]

    def metrics(self, theta: np.ndarray, x: np.ndarray, y: np.ndarray) -> dict[str, float]:
        loss, gradient = self.loss_grad(theta, x, y)
        error = self.predict(theta, x) - y
        return {"loss": loss, "accuracy": float("nan"), "mae": float(np.mean(np.abs(error))), "rmse": float(np.sqrt(np.mean(error**2))), "gradient_norm": float(np.linalg.norm(gradient))}


def make_model(dataset: DatasetBundle, l2: float) -> SoftmaxRegression | LinearRegression:
    if dataset.task == "classification":
        return SoftmaxRegression(dataset.n_features, dataset.n_outputs, l2)
    if dataset.task == "regression":
        return LinearRegression(dataset.n_features, dataset.n_outputs, l2)
    raise ValueError(f"Unsupported task: {dataset.task}")
