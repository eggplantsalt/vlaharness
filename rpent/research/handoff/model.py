"""Trainable probabilistic outcome models with calibration and uncertainty.

scikit-learn is an optional research dependency and is imported only inside
fit/load operations. Importing RPent's normal runtime never imports it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol, Sequence, runtime_checkable

import numpy as np

from rpent.research.handoff.types import OutcomeEstimate


class ModelCompatibilityError(ValueError):
    """Raised when a model and feature vector/schema are incompatible."""


class ModelTrainingError(ValueError):
    """Raised when a requested estimator cannot be fitted honestly."""


@runtime_checkable
class FeatureVectorLike(Protocol):
    spec_id: str
    names: tuple[str, ...]
    values: tuple[float, ...]


@runtime_checkable
class OutcomeModel(Protocol):
    feature_spec_id: str
    feature_names: tuple[str, ...]

    def predict_one(self, features: FeatureVectorLike) -> OutcomeEstimate:
        """Estimate learned-controller success at one handoff state."""


def _as_training_arrays(
    x: Sequence[Sequence[float]] | np.ndarray,
    y: Sequence[bool | int | float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(x, dtype=np.float64)
    raw_labels = np.asarray(y)
    if raw_labels.ndim != 1 or raw_labels.shape[0] != matrix.shape[0]:
        raise ModelTrainingError("labels must be one-dimensional and align with features")
    try:
        numeric_labels = np.asarray(raw_labels, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ModelTrainingError("outcome labels must be numeric binary values") from exc
    if not np.isfinite(numeric_labels).all() or np.any(
        (numeric_labels != 0.0) & (numeric_labels != 1.0)
    ):
        raise ModelTrainingError("outcome labels must be exactly binary")
    labels = numeric_labels.astype(np.int64)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        raise ModelTrainingError("training features must be a non-empty 2-D matrix")
    if not np.isfinite(matrix).all():
        raise ModelTrainingError("training features contain NaN or Inf")
    unique = set(int(value) for value in np.unique(labels))
    if not unique.issubset({0, 1}):
        raise ModelTrainingError("outcome labels must be binary")
    if len(unique) < 2:
        raise ModelTrainingError(
            "probabilistic estimator requires both success and failure examples"
        )
    return matrix, labels


def _validate_probability_array(values: np.ndarray, *, context: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64).reshape(-1)
    if result.size == 0 or not np.isfinite(result).all():
        raise ValueError(f"{context} returned empty, NaN, or Inf probabilities")
    if np.any((result < 0.0) | (result > 1.0)):
        raise ValueError(f"{context} returned probabilities outside [0, 1]")
    return result


@dataclass
class ProbabilityCalibrator:
    """Platt/sigmoid or isotonic calibration over raw probabilities."""

    method: Literal["none", "platt", "isotonic"] = "none"
    clip_epsilon: float = 1e-6
    _estimator: object | None = field(default=None, init=False, repr=False)

    def fit(
        self,
        probabilities: Sequence[float] | np.ndarray,
        labels: Sequence[bool | int] | np.ndarray,
    ) -> "ProbabilityCalibrator":
        probs = _validate_probability_array(
            np.asarray(probabilities), context="calibration input"
        )
        raw_y = np.asarray(labels).reshape(-1)
        if probs.shape[0] != raw_y.shape[0] or probs.size == 0:
            raise ModelTrainingError("calibration probabilities and labels do not align")
        try:
            numeric_y = np.asarray(raw_y, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ModelTrainingError("calibration labels must be binary") from exc
        if not np.isfinite(numeric_y).all() or np.any(
            (numeric_y != 0.0) & (numeric_y != 1.0)
        ):
            raise ModelTrainingError("calibration labels must be exactly binary")
        y = numeric_y.astype(np.int64)
        if len(np.unique(y)) < 2:
            raise ModelTrainingError(
                "calibration split needs both success and failure examples"
            )
        if self.method == "none":
            self._estimator = None
            return self
        if self.method == "platt":
            try:
                from sklearn.linear_model import LogisticRegression
            except ImportError as exc:  # pragma: no cover - dependency path
                raise RuntimeError(
                    "Platt calibration requires the optional handoff dependency "
                    "scikit-learn"
                ) from exc
            clipped = np.clip(probs, self.clip_epsilon, 1.0 - self.clip_epsilon)
            logits = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
            estimator = LogisticRegression(random_state=0, max_iter=1000)
            estimator.fit(logits, y)
            self._estimator = estimator
            return self
        if self.method == "isotonic":
            try:
                from sklearn.isotonic import IsotonicRegression
            except ImportError as exc:  # pragma: no cover - dependency path
                raise RuntimeError(
                    "isotonic calibration requires the optional handoff dependency "
                    "scikit-learn"
                ) from exc
            estimator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            estimator.fit(probs, y)
            self._estimator = estimator
            return self
        raise ValueError(f"unknown calibration method: {self.method!r}")

    @property
    def fitted(self) -> bool:
        return self.method == "none" or self._estimator is not None

    def transform(self, probabilities: Sequence[float] | np.ndarray) -> np.ndarray:
        probs = _validate_probability_array(
            np.asarray(probabilities), context="calibration input"
        )
        if self.method == "none":
            return probs
        if self._estimator is None:
            raise RuntimeError("calibrator has not been fitted")
        if self.method == "platt":
            clipped = np.clip(probs, self.clip_epsilon, 1.0 - self.clip_epsilon)
            logits = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
            transformed = self._estimator.predict_proba(logits)[:, 1]
        else:
            transformed = self._estimator.predict(probs)
        return _validate_probability_array(
            np.asarray(transformed), context=f"{self.method} calibrator"
        )


@dataclass
class ConstantOutcomeModel:
    """Constant/prior model for sanity checks and fake-boundary tests only."""

    probability: float
    feature_spec_id: str
    feature_names: tuple[str, ...]

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.probability) <= 1.0:
            raise ValueError("constant probability must be in [0, 1]")
        if not self.feature_spec_id:
            raise ValueError("feature_spec_id must be non-empty")

    def _check(self, features: FeatureVectorLike) -> None:
        if features.spec_id != self.feature_spec_id:
            raise ModelCompatibilityError(
                f"feature spec mismatch: model={self.feature_spec_id!r}, "
                f"vector={features.spec_id!r}"
            )
        if tuple(features.names) != self.feature_names:
            raise ModelCompatibilityError("feature name/order mismatch")

    def predict_one(self, features: FeatureVectorLike) -> OutcomeEstimate:
        self._check(features)
        probability = float(self.probability)
        return OutcomeEstimate(
            mean_success_probability=probability,
            epistemic_std=0.0,
            conservative_success_probability=probability,
            lower_quantile_probability=probability,
            upper_quantile_probability=probability,
            ensemble_size=1,
            calibrated=False,
        )


@dataclass
class SklearnProbabilityModel:
    """Concrete logistic or nonlinear HistGradientBoosting outcome model."""

    estimator_kind: Literal["logistic", "hist_gradient_boosting"]
    feature_spec_id: str
    feature_names: tuple[str, ...]
    random_state: int = 0
    max_iter: int = 500
    _estimator: object | None = field(default=None, init=False, repr=False)

    def fit(
        self,
        x: Sequence[Sequence[float]] | np.ndarray,
        y: Sequence[bool | int | float] | np.ndarray,
        *,
        sample_weight: Sequence[float] | np.ndarray | None = None,
    ) -> "SklearnProbabilityModel":
        matrix, labels = _as_training_arrays(x, y)
        if matrix.shape[1] != len(self.feature_names):
            raise ModelTrainingError(
                "training matrix width does not match the declared feature order"
            )
        weights = None
        if sample_weight is not None:
            weights = np.asarray(sample_weight, dtype=np.float64).reshape(-1)
            if weights.shape[0] != matrix.shape[0] or not np.isfinite(weights).all():
                raise ModelTrainingError("invalid sample weights")
        try:
            if self.estimator_kind == "logistic":
                from sklearn.linear_model import LogisticRegression
                from sklearn.pipeline import Pipeline
                from sklearn.preprocessing import StandardScaler

                estimator = Pipeline(
                    [
                        ("scale", StandardScaler()),
                        (
                            "classifier",
                            LogisticRegression(
                                max_iter=self.max_iter,
                                random_state=self.random_state,
                            ),
                        ),
                    ]
                )
                fit_kwargs = (
                    {"classifier__sample_weight": weights}
                    if weights is not None
                    else {}
                )
                estimator.fit(matrix, labels, **fit_kwargs)
            elif self.estimator_kind == "hist_gradient_boosting":
                from sklearn.ensemble import HistGradientBoostingClassifier

                estimator = HistGradientBoostingClassifier(
                    max_iter=self.max_iter,
                    random_state=self.random_state,
                )
                estimator.fit(matrix, labels, sample_weight=weights)
            else:
                raise ValueError(f"unknown estimator kind: {self.estimator_kind!r}")
        except ImportError as exc:  # pragma: no cover - dependency path
            raise RuntimeError(
                "trainable handoff outcome models require the optional "
                "scikit-learn dependency"
            ) from exc
        self._estimator = estimator
        return self

    @property
    def fitted(self) -> bool:
        return self._estimator is not None

    def predict_probabilities(self, matrix: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
        if self._estimator is None:
            raise RuntimeError("outcome model has not been fitted")
        values = np.asarray(matrix, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != len(self.feature_names):
            raise ModelCompatibilityError("prediction matrix has incompatible shape")
        if not np.isfinite(values).all():
            raise ValueError("prediction features contain NaN or Inf")
        probabilities = self._estimator.predict_proba(values)[:, 1]
        return _validate_probability_array(
            probabilities, context=self.estimator_kind
        )

    def _matrix_for(self, features: FeatureVectorLike) -> np.ndarray:
        if features.spec_id != self.feature_spec_id:
            raise ModelCompatibilityError(
                f"feature spec mismatch: model={self.feature_spec_id!r}, "
                f"vector={features.spec_id!r}"
            )
        if tuple(features.names) != self.feature_names:
            raise ModelCompatibilityError("feature name/order mismatch")
        matrix = np.asarray(features.values, dtype=np.float64).reshape(1, -1)
        if matrix.shape[1] != len(self.feature_names):
            raise ModelCompatibilityError("feature vector width mismatch")
        return matrix

    def predict_one(self, features: FeatureVectorLike) -> OutcomeEstimate:
        probability = float(self.predict_probabilities(self._matrix_for(features))[0])
        return OutcomeEstimate(
            mean_success_probability=probability,
            epistemic_std=0.0,
            conservative_success_probability=probability,
            lower_quantile_probability=probability,
            upper_quantile_probability=probability,
            ensemble_size=1,
            calibrated=False,
        )


@dataclass
class CalibratedOutcomeModel:
    """Held-out Platt/isotonic calibration wrapper for any probability model."""

    base_model: OutcomeModel
    calibrator: ProbabilityCalibrator

    @property
    def feature_spec_id(self) -> str:
        return self.base_model.feature_spec_id

    @property
    def feature_names(self) -> tuple[str, ...]:
        return self.base_model.feature_names

    def fit_calibrator(
        self,
        features: Sequence[FeatureVectorLike],
        labels: Sequence[bool | int] | np.ndarray,
    ) -> "CalibratedOutcomeModel":
        if not features:
            raise ModelTrainingError("calibration feature set is empty")
        raw = [
            self.base_model.predict_one(vector).mean_success_probability
            for vector in features
        ]
        self.calibrator.fit(raw, labels)
        return self

    def predict_one(self, features: FeatureVectorLike) -> OutcomeEstimate:
        base = self.base_model.predict_one(features)
        calibrated = float(
            self.calibrator.transform([base.mean_success_probability])[0]
        )
        # A monotone point calibrator does not create epistemic uncertainty.
        # Preserve a base model's uncertainty scale conservatively when present.
        conservative = float(
            np.clip(calibrated - base.epistemic_std, 0.0, 1.0)
        )
        return OutcomeEstimate(
            mean_success_probability=calibrated,
            epistemic_std=base.epistemic_std,
            conservative_success_probability=conservative,
            lower_quantile_probability=(
                conservative
                if base.lower_quantile_probability is not None
                else None
            ),
            upper_quantile_probability=(
                float(np.clip(calibrated + base.epistemic_std, 0.0, 1.0))
                if base.upper_quantile_probability is not None
                else None
            ),
            ensemble_size=base.ensemble_size,
            calibrated=self.calibrator.method != "none",
        )


@dataclass
class BootstrapOutcomeModel:
    """Group-bootstrap ensemble providing executable epistemic uncertainty."""

    estimator_kind: Literal["logistic", "hist_gradient_boosting"]
    feature_spec_id: str
    feature_names: tuple[str, ...]
    ensemble_size: int = 20
    random_state: int = 0
    uncertainty_beta: float = 1.0
    lower_quantile: float = 0.1
    upper_quantile: float = 0.9
    calibration_method: Literal["none", "platt", "isotonic"] = "none"
    max_iter: int = 500
    members: list[SklearnProbabilityModel] = field(default_factory=list, init=False)
    calibrator: ProbabilityCalibrator = field(init=False)

    def __post_init__(self) -> None:
        if self.ensemble_size < 2:
            raise ValueError("bootstrap ensemble_size must be at least 2")
        if self.uncertainty_beta < 0.0:
            raise ValueError("uncertainty_beta must be non-negative")
        if not 0.0 <= self.lower_quantile <= self.upper_quantile <= 1.0:
            raise ValueError("bootstrap quantiles must be ordered in [0, 1]")
        self.calibrator = ProbabilityCalibrator(method=self.calibration_method)

    def fit(
        self,
        x: Sequence[Sequence[float]] | np.ndarray,
        y: Sequence[bool | int | float] | np.ndarray,
        *,
        groups: Sequence[str] | np.ndarray | None = None,
        calibration_x: Sequence[Sequence[float]] | np.ndarray | None = None,
        calibration_y: Sequence[bool | int] | np.ndarray | None = None,
    ) -> "BootstrapOutcomeModel":
        matrix, labels = _as_training_arrays(x, y)
        if matrix.shape[1] != len(self.feature_names):
            raise ModelTrainingError("training matrix width and feature names disagree")
        group_values = (
            np.asarray(groups, dtype=object).reshape(-1)
            if groups is not None
            else np.asarray([str(index) for index in range(matrix.shape[0])], dtype=object)
        )
        if group_values.shape[0] != matrix.shape[0]:
            raise ModelTrainingError("bootstrap groups do not align with examples")
        unique_groups = np.asarray(sorted(set(str(item) for item in group_values)), dtype=object)
        if unique_groups.size < 2:
            raise ModelTrainingError("group bootstrap requires at least two groups")

        rng = np.random.default_rng(self.random_state)
        members: list[SklearnProbabilityModel] = []
        for member_index in range(self.ensemble_size):
            fitted = False
            for _attempt in range(100):
                sampled_groups = rng.choice(
                    unique_groups, size=unique_groups.size, replace=True
                )
                sampled_indices = np.concatenate(
                    [
                        np.flatnonzero(group_values.astype(str) == str(group))
                        for group in sampled_groups
                    ]
                )
                sampled_labels = labels[sampled_indices]
                if len(np.unique(sampled_labels)) < 2:
                    continue
                member = SklearnProbabilityModel(
                    estimator_kind=self.estimator_kind,
                    feature_spec_id=self.feature_spec_id,
                    feature_names=self.feature_names,
                    random_state=self.random_state + member_index + 1,
                    max_iter=self.max_iter,
                ).fit(matrix[sampled_indices], sampled_labels)
                members.append(member)
                fitted = True
                break
            if not fitted:
                raise ModelTrainingError(
                    "could not draw a two-class group bootstrap sample; "
                    "collect more independent success and failure groups"
                )
        self.members = members

        if self.calibration_method != "none":
            if calibration_x is None or calibration_y is None:
                raise ModelTrainingError(
                    "calibration method requested but no held-out calibration split provided"
                )
            cal_matrix = np.asarray(calibration_x, dtype=np.float64)
            raw_mean = self._member_matrix(cal_matrix).mean(axis=0)
            self.calibrator.fit(raw_mean, calibration_y)
        return self

    @property
    def fitted(self) -> bool:
        return len(self.members) == self.ensemble_size

    def _member_matrix(self, matrix: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("bootstrap outcome model has not been fitted")
        values = np.asarray(matrix, dtype=np.float64)
        return np.vstack([member.predict_probabilities(values) for member in self.members])

    def predict_member_probabilities(self, features: FeatureVectorLike) -> np.ndarray:
        if features.spec_id != self.feature_spec_id:
            raise ModelCompatibilityError(
                f"feature spec mismatch: model={self.feature_spec_id!r}, "
                f"vector={features.spec_id!r}"
            )
        if tuple(features.names) != self.feature_names:
            raise ModelCompatibilityError("feature name/order mismatch")
        matrix = np.asarray(features.values, dtype=np.float64).reshape(1, -1)
        raw = self._member_matrix(matrix)[:, 0]
        if self.calibration_method == "none":
            return raw
        return self.calibrator.transform(raw)

    def predict_one(self, features: FeatureVectorLike) -> OutcomeEstimate:
        if features.spec_id != self.feature_spec_id:
            raise ModelCompatibilityError(
                f"feature spec mismatch: model={self.feature_spec_id!r}, "
                f"vector={features.spec_id!r}"
            )
        if tuple(features.names) != self.feature_names:
            raise ModelCompatibilityError("feature name/order mismatch")
        matrix = np.asarray(features.values, dtype=np.float64).reshape(1, -1)
        raw = self._member_matrix(matrix)[:, 0]
        raw_mean = float(np.mean(raw))
        raw_lower = float(np.quantile(raw, self.lower_quantile))
        raw_upper = float(np.quantile(raw, self.upper_quantile))
        if self.calibration_method == "none":
            probabilities = raw
            mean = raw_mean
            lower = raw_lower
            upper = raw_upper
        else:
            # The calibrator is fitted to the ensemble mean, so the reported
            # point prediction must be calibration(mean), not the mean of
            # independently calibrated members.  Monotonic endpoint/member
            # transforms propagate the uncertainty scale without changing the
            # estimand used during held-out calibration.
            probabilities = self.calibrator.transform(raw)
            mean = float(self.calibrator.transform([raw_mean])[0])
            lower = float(self.calibrator.transform([raw_lower])[0])
            upper = float(self.calibrator.transform([raw_upper])[0])
        std = float(np.std(probabilities, ddof=1))
        conservative = float(np.clip(min(lower, mean - self.uncertainty_beta * std), 0.0, 1.0))
        return OutcomeEstimate(
            mean_success_probability=mean,
            epistemic_std=std,
            conservative_success_probability=conservative,
            lower_quantile_probability=lower,
            upper_quantile_probability=upper,
            ensemble_size=len(self.members),
            calibrated=self.calibration_method != "none",
        )
