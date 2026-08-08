"""Checksummed model artifacts with schema/feature compatibility checks."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from rpent.research.handoff.features import FeatureSpec
from rpent.research.handoff.model import ModelCompatibilityError, OutcomeModel
from rpent.research.handoff.types import HandoffRecord, STATE_SCHEMA_VERSION

MODEL_ARTIFACT_SCHEMA_VERSION = "rpent.handoff-model-artifact/v1"


class SourceIdentity(HandoffRecord):
    git_revision: str | None = None
    dirty: bool | None = None
    package_version: str | None = None
    external_runtime_identity: str | None = None


class ModelArtifactManifest(HandoffRecord):
    schema_version: Literal[MODEL_ARTIFACT_SCHEMA_VERSION] = (
        MODEL_ARTIFACT_SCHEMA_VERSION
    )
    artifact_id: str
    model_kind: str
    estimator_file: str = "estimator.joblib"
    estimator_sha256: str
    feature_spec: FeatureSpec
    state_schema_version: Literal[STATE_SCHEMA_VERSION] = STATE_SCHEMA_VERSION
    training_target_label: str
    calibration_method: str
    dataset_fingerprint: str
    split_assignment_fingerprint: str
    training_record_ids: tuple[str, ...]
    calibration_record_ids: tuple[str, ...]
    held_out_record_ids: tuple[str, ...]
    training_configuration: dict[str, Any]
    source_identity: SourceIdentity
    trusted_loader_required: bool = True
    format_notes: str = (
        "The estimator is joblib-serialized executable Python data. Load only "
        "artifacts produced by a trusted experiment pipeline. JSONL remains the "
        "authoritative dataset ground truth."
    )

    @field_validator(
        "artifact_id",
        "model_kind",
        "estimator_file",
        "estimator_sha256",
        "training_target_label",
        "dataset_fingerprint",
        "split_assignment_fingerprint",
    )
    @classmethod
    def validate_text(cls, value: str, info) -> str:
        if not value:
            raise ValueError(f"{info.field_name} must be non-empty")
        return value

    @field_validator("training_configuration")
    @classmethod
    def validate_config(cls, value: dict[str, Any]) -> dict[str, Any]:
        try:
            json.dumps(value, sort_keys=True, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("training configuration must be finite JSON") from exc
        return value

    @model_validator(mode="after")
    def validate_split_membership(self):
        partitions = (
            self.training_record_ids,
            self.calibration_record_ids,
            self.held_out_record_ids,
        )
        for values in partitions:
            if not values or values != tuple(sorted(values)):
                raise ValueError(
                    "artifact split record IDs must be non-empty and sorted"
                )
            if len(values) != len(set(values)):
                raise ValueError("artifact split record IDs must be unique")
        if any(set(left).intersection(right) for index, left in enumerate(partitions) for right in partitions[index + 1 :]):
            raise ValueError("artifact split record IDs must be disjoint")
        return self


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_id(manifest: ModelArtifactManifest) -> str:
    """Bind the ID to every authoritative manifest field and estimator bytes.

    ``format_notes`` and the loader-safety flag are intentionally included too:
    using the complete canonical manifest (apart from the self-referential ID)
    prevents a newly added authoritative field from being omitted from identity
    by accident.
    """
    payload = json.dumps(
        manifest.model_dump(
            mode="json",
            exclude={"artifact_id"},
            exclude_none=False,
        ),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return "model-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def save_model_artifact(
    directory: str | Path,
    *,
    model: OutcomeModel,
    feature_spec: FeatureSpec,
    training_target_label: str,
    dataset_fingerprint: str,
    training_configuration: dict[str, Any],
    source_identity: SourceIdentity,
    calibration_method: str,
    split_assignment_fingerprint: str,
    training_record_ids: tuple[str, ...],
    calibration_record_ids: tuple[str, ...],
    held_out_record_ids: tuple[str, ...],
    overwrite: bool = False,
) -> ModelArtifactManifest:
    """Save estimator and canonical manifest atomically within one directory."""
    if model.feature_spec_id != feature_spec.spec_id:
        raise ModelCompatibilityError("model and artifact feature spec IDs differ")
    if tuple(model.feature_names) != feature_spec.names:
        raise ModelCompatibilityError("model and artifact feature order differs")
    output = Path(directory)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    estimator_path = output / "estimator.joblib"
    if not overwrite and (manifest_path.exists() or estimator_path.exists()):
        raise FileExistsError(
            f"model artifact already exists at {output}; pass overwrite=True explicitly"
        )
    try:
        import joblib
    except ImportError as exc:  # pragma: no cover - optional dependency path
        raise RuntimeError(
            "saving handoff model artifacts requires the optional joblib dependency"
        ) from exc

    estimator_tmp: Path | None = None
    manifest_tmp: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix="estimator-", suffix=".joblib.tmp", dir=output, delete=False
        ) as handle:
            estimator_tmp = Path(handle.name)
        joblib.dump(model, estimator_tmp)
        estimator_digest = _sha256(estimator_tmp)
        manifest_without_id = ModelArtifactManifest(
            artifact_id="pending",
            model_kind=type(model).__name__,
            estimator_sha256=estimator_digest,
            feature_spec=feature_spec,
            training_target_label=training_target_label,
            calibration_method=calibration_method,
            dataset_fingerprint=dataset_fingerprint,
            split_assignment_fingerprint=split_assignment_fingerprint,
            training_record_ids=tuple(sorted(training_record_ids)),
            calibration_record_ids=tuple(sorted(calibration_record_ids)),
            held_out_record_ids=tuple(sorted(held_out_record_ids)),
            training_configuration=training_configuration,
            source_identity=source_identity,
        )
        manifest = manifest_without_id.model_copy(
            update={"artifact_id": _artifact_id(manifest_without_id)}
        )
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix="manifest-",
            suffix=".json.tmp",
            dir=output,
            delete=False,
        ) as handle:
            manifest_tmp = Path(handle.name)
            handle.write(manifest.canonical_json())
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(estimator_tmp, estimator_path)
        estimator_tmp = None
        os.replace(manifest_tmp, manifest_path)
        manifest_tmp = None
        return manifest
    finally:
        for temporary in (estimator_tmp, manifest_tmp):
            if temporary is not None and temporary.exists():
                temporary.unlink()


def load_model_artifact(
    directory: str | Path,
    *,
    trusted: bool,
    expected_feature_spec: FeatureSpec | None = None,
    expected_dataset_fingerprint: str | None = None,
    expected_state_schema_version: str = STATE_SCHEMA_VERSION,
    expected_model_artifact_id: str | None = None,
) -> tuple[OutcomeModel, ModelArtifactManifest]:
    """Load a trusted artifact after verifying checksum and compatibility."""
    if not trusted:
        raise PermissionError(
            "joblib model artifacts can execute code; pass trusted=True only for "
            "artifacts produced by your trusted training pipeline"
        )
    root = Path(directory)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"model manifest not found: {manifest_path}")
    manifest = ModelArtifactManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    actual_artifact_id = _artifact_id(manifest)
    if manifest.artifact_id != actual_artifact_id:
        raise ValueError(
            "model artifact identity mismatch; manifest fields or the bound "
            "estimator checksum may have been modified"
        )
    if (
        expected_model_artifact_id is not None
        and manifest.artifact_id != expected_model_artifact_id
    ):
        raise ModelCompatibilityError(
            "model artifact identity differs from the runtime-bound identity"
        )
    if manifest.state_schema_version != expected_state_schema_version:
        raise ModelCompatibilityError(
            "model artifact state schema is incompatible with this runtime"
        )
    if expected_feature_spec is not None:
        if manifest.feature_spec.fingerprint != expected_feature_spec.fingerprint:
            raise ModelCompatibilityError("model artifact feature specification differs")
    if (
        expected_dataset_fingerprint is not None
        and manifest.dataset_fingerprint != expected_dataset_fingerprint
    ):
        raise ModelCompatibilityError("model artifact dataset fingerprint differs")
    estimator_path = root / manifest.estimator_file
    if not estimator_path.is_file():
        raise FileNotFoundError(f"model estimator not found: {estimator_path}")
    actual_digest = _sha256(estimator_path)
    if actual_digest != manifest.estimator_sha256:
        raise ValueError(
            "model estimator checksum mismatch; artifact may be corrupt or modified"
        )
    try:
        import joblib
    except ImportError as exc:  # pragma: no cover - optional dependency path
        raise RuntimeError(
            "loading handoff model artifacts requires the optional joblib dependency"
        ) from exc
    model = joblib.load(estimator_path)
    if not isinstance(model, OutcomeModel):
        raise TypeError("loaded object does not implement the OutcomeModel protocol")
    if model.feature_spec_id != manifest.feature_spec.spec_id:
        raise ModelCompatibilityError("loaded estimator feature spec ID mismatch")
    if tuple(model.feature_names) != manifest.feature_spec.names:
        raise ModelCompatibilityError("loaded estimator feature order mismatch")
    return model, manifest
