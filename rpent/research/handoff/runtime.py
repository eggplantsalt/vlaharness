"""JSON-configured policy construction for online and controlled runners."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from rpent.research.handoff.artifacts import load_model_artifact
from rpent.research.handoff.candidates import (
    CandidateFeaturePredictor,
    CandidateFeaturePredictorConfig,
)
from rpent.research.handoff.features import FeatureBuilder, FeaturePreset, make_feature_spec
from rpent.research.handoff.model import OutcomeModel
from rpent.research.handoff.policies import (
    CompetenceProjectionPolicy,
    CompetenceThresholdPolicy,
    DirectHandoffPolicy,
    FixedCanonicalPolicy,
    FixedDistancePolicy,
    HandoffPolicy,
    OutcomeCalibratedSwitchingPolicy,
    PositiveNearestSuccessPolicy,
    PositiveReference,
    PositiveSupportRegionPolicy,
    PostHocOraclePolicy,
    RiskAwareSwitchingConfig,
)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _references(value: Any) -> tuple[PositiveReference, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("positive_references must be a JSON array")
    return tuple(PositiveReference.model_validate(item) for item in value)


def _policy_references(policy_config: Mapping[str, Any]) -> tuple[PositiveReference, ...]:
    inline = policy_config.get("positive_references")
    artifact_path = policy_config.get("positive_references_file")
    if inline is not None and artifact_path is not None:
        raise ValueError(
            "configure either positive_references or positive_references_file, not both"
        )
    if artifact_path is not None:
        if not isinstance(artifact_path, str) or not artifact_path:
            raise ValueError("positive_references_file must be a non-empty path")
        from rpent.research.handoff.baseline_data import (
            load_positive_reference_artifact,
        )

        return load_positive_reference_artifact(artifact_path).references
    return _references(inline if inline is not None else [])


def _model_components(
    config: Mapping[str, Any],
    supplied_model: OutcomeModel | None,
    *,
    expected_model_artifact_id: str | None,
) -> tuple[OutcomeModel | None, FeatureBuilder | None, CandidateFeaturePredictor]:
    feature_config = _mapping(config.get("feature_spec", {}), "feature_spec")
    skill_vocabulary = feature_config.get("skill_vocabulary")
    if not isinstance(skill_vocabulary, Sequence) or isinstance(
        skill_vocabulary, (str, bytes)
    ):
        raise ValueError("feature_spec.skill_vocabulary must be a JSON array")
    spec = make_feature_spec(
        FeaturePreset(feature_config.get("preset", FeaturePreset.DEPLOYMENT_FULL)),
        skill_vocabulary=[str(value) for value in skill_vocabulary],
    )
    builder = FeatureBuilder(spec)
    predictor = CandidateFeaturePredictor(
        CandidateFeaturePredictorConfig.model_validate(
            config.get("candidate_feature_predictor", {})
        )
    )
    model = supplied_model
    model_artifact = config.get("model_artifact")
    if model_artifact and model is not None and expected_model_artifact_id is not None:
        raise ValueError(
            "cannot verify expected_model_artifact_id when a supplied in-memory "
            "model bypasses the configured artifact"
        )
    if model is None and model_artifact:
        if expected_model_artifact_id is None:
            raise ValueError(
                "configured model_artifact requires a runtime-bound "
                "expected_model_artifact_id"
            )
        model, _manifest = load_model_artifact(
            str(model_artifact),
            trusted=bool(config.get("trusted_model_artifact", False)),
            expected_feature_spec=spec,
            expected_model_artifact_id=expected_model_artifact_id,
        )
    elif expected_model_artifact_id is not None and not model_artifact:
        raise ValueError(
            "expected_model_artifact_id was configured without a model_artifact"
        )
    if model is not None:
        if model.feature_spec_id != spec.spec_id or tuple(model.feature_names) != spec.names:
            raise ValueError("configured feature specification is incompatible with model")
    return model, builder, predictor


def _build_one(
    policy_config: Mapping[str, Any],
    *,
    model: OutcomeModel | None,
    feature_builder: FeatureBuilder | None,
    predictor: CandidateFeaturePredictor,
) -> HandoffPolicy:
    name = str(policy_config.get("name") or "")
    if not name:
        raise ValueError("policy.name is required")
    if name == "direct_frozen_pi0":
        return DirectHandoffPolicy()
    if name == "fixed_canonical_precontact":
        relative = policy_config.get("target_relative_position_m")
        if not isinstance(relative, (list, tuple)) or len(relative) != 3:
            raise ValueError("fixed canonical policy needs target_relative_position_m")
        return FixedCanonicalPolicy(
            tuple(float(value) for value in relative),
            tolerance_m=float(policy_config.get("tolerance_m", 0.015)),
        )
    if name == "fixed_distance":
        return FixedDistancePolicy(
            float(policy_config["distance_m"]),
            tolerance_m=float(policy_config.get("tolerance_m", 0.015)),
        )
    if name == "positive_nearest_success":
        return PositiveNearestSuccessPolicy(
            _policy_references(policy_config),
            handoff_radius=float(policy_config.get("handoff_radius", 0.03)),
        )
    if name == "positive_support_region":
        bandwidth = policy_config.get("bandwidth_m", (0.04, 0.04, 0.04))
        if not isinstance(bandwidth, (list, tuple)) or len(bandwidth) != 3:
            raise ValueError("support-region bandwidth_m must have three components")
        return PositiveSupportRegionPolicy(
            _policy_references(policy_config),
            bandwidth_m=tuple(float(value) for value in bandwidth),
        )
    if name == "posthoc_oracle_upper_bound":
        costs = _mapping(
            policy_config.get("actual_cost_by_candidate", {}),
            "actual_cost_by_candidate",
        )
        return PostHocOraclePolicy(
            {str(key): float(value) for key, value in costs.items()},
            allow_privileged=bool(policy_config.get("allow_privileged", False)),
        )

    if model is None or feature_builder is None:
        raise ValueError(f"policy {name!r} requires a compatible trained outcome model")
    common = {
        "model": model,
        "feature_builder": feature_builder,
        "predictor": predictor,
    }
    if name == "competence_probability_threshold":
        return CompetenceThresholdPolicy(
            **common,
            threshold=float(policy_config.get("threshold", 0.7)),
            conservative=bool(policy_config.get("conservative", False)),
        )
    if name == "competence_projection":
        return CompetenceProjectionPolicy(
            **common,
            threshold=float(policy_config.get("threshold", 0.7)),
        )
    if name == "outcome_calibrated_switching":
        cost_config = RiskAwareSwitchingConfig.model_validate(
            policy_config.get("costs", {})
        )
        return OutcomeCalibratedSwitchingPolicy(**common, config=cost_config)
    raise ValueError(f"unknown handoff policy: {name!r}")


def build_runtime_policy(
    config: Mapping[str, Any],
    *,
    model: OutcomeModel | None = None,
    expected_model_artifact_id: str | None = None,
) -> tuple[HandoffPolicy, HandoffPolicy | None]:
    """Build primary and optional fallback policies from resolved config."""
    resolved_model, feature_builder, predictor = _model_components(
        config,
        model,
        expected_model_artifact_id=expected_model_artifact_id,
    )
    policy = _build_one(
        _mapping(config.get("policy"), "policy"),
        model=resolved_model,
        feature_builder=feature_builder,
        predictor=predictor,
    )
    fallback_payload = config.get("fallback_policy")
    fallback = None
    if fallback_payload is not None:
        fallback = _build_one(
            _mapping(fallback_payload, "fallback_policy"),
            model=resolved_model,
            feature_builder=feature_builder,
            predictor=predictor,
        )
    return policy, fallback
