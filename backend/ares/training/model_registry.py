"""
ARES Per-Client Model Registry Manager.

Manages `backend/ares/training/model_registry.json`, preserving legacy global
adapter/target fields while registering per-organization DistilBERT/hybrid intent
& threat classifiers, per-client compression adapters, fidelity thresholds, and metrics.
"""

from __future__ import annotations

import hmac
import json
import logging
import os
import re
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

DEFAULT_REGISTRY_PATH = Path(__file__).resolve().parent / "model_registry.json"
MODELS_ROOT_DIR = Path(__file__).resolve().parents[2] / "models"

_SAFE_ORG_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_WINDOWS_RESERVED_NAMES = frozenset(
    {
        "con",
        "prn",
        "aux",
        "nul",
        *(f"com{i}" for i in range(1, 10)),
        *(f"lpt{i}" for i in range(1, 10)),
    }
)


def validate_org_id(org_id: str) -> str:
    """Validate `org_id` to prevent directory traversal and Windows path collisions outside `backend/models/`."""
    if not isinstance(org_id, str):
        raise ValueError(
            f"Invalid organization ID '{org_id}'. Must match ^[A-Za-z0-9][A-Za-z0-9._-]{{0,127}}$ with no path separators."
        )
    if (
        not org_id
        or not _SAFE_ORG_ID_RE.match(org_id)
        or ".." in org_id
        or org_id.endswith(".")
        or org_id.split(".", 1)[0].lower() in _WINDOWS_RESERVED_NAMES
    ):
        raise ValueError(
            f"Invalid organization ID '{org_id}'. Must match ^[A-Za-z0-9][A-Za-z0-9._-]{{0,127}}$ with no path separators."
        )
    return org_id

DEFAULT_CLIENT_PROFILES: Dict[str, Dict[str, Any]] = {
    "org_acme_fintech": {
        "display_name": "Acme Fintech RAG",
        "domain": "fintech",
        "application_name": "AcmeFintechRAG",
        "canary_token": "CANARY_ACME_FINTECH_77",
        "fidelity_thresholds": {
            "coding_technical": 0.90,
            "data_extraction_json": 0.92,
            "analytical_qa": 0.88,
            "creative_writing": 0.78,
            "system_directive": 0.88,
        },
    },
    "org_dev_ops_agent": {
        "display_name": "DevOps Tool Agent",
        "domain": "software",
        "application_name": "DevOpsToolAgent",
        "canary_token": "CANARY_DEVOPS_AGENT_88",
        "fidelity_thresholds": {
            "coding_technical": 0.92,
            "data_extraction_json": 0.90,
            "analytical_qa": 0.85,
            "creative_writing": 0.75,
            "system_directive": 0.86,
        },
    },
    "org_medshield_health": {
        "display_name": "MedShield HIPAA Triage",
        "domain": "healthcare",
        "application_name": "MedShieldTriage",
        "canary_token": "CANARY_MEDSHIELD_HIPAA_99",
        "fidelity_thresholds": {
            "coding_technical": 0.90,
            "data_extraction_json": 0.94,
            "analytical_qa": 0.92,
            "creative_writing": 0.80,
            "system_directive": 0.90,
        },
    },
}

_INTENT_ALIASES = {
    "data_extraction": "data_extraction_json",
    "analytical_reasoning": "analytical_qa",
    "general_qa": "analytical_qa",
}

_REGISTRY_LOCK = threading.RLock()


class ModelRegistry:
    """Thread-safe reader/writer for `model_registry.json` with optional SQLAlchemy durability."""

    def __init__(
        self,
        registry_path: Optional[Path] = None,
        session_factory: Optional[Callable[[], Any]] = None,
        use_db: Optional[bool] = None,
    ) -> None:
        self._custom_registry_path = registry_path is not None
        self.registry_path = Path(registry_path) if registry_path else DEFAULT_REGISTRY_PATH
        self._session_factory = session_factory
        if use_db is not None:
            self._use_db = bool(use_db)
        elif session_factory is not None:
            self._use_db = True
        elif self._custom_registry_path:
            self._use_db = False
        else:
            try:
                from ares.config import settings as default_settings

                self._use_db = getattr(default_settings, "environment", "development") != "test"
            except Exception:
                self._use_db = False

    def _get_session_factory(self) -> Optional[Callable[[], Any]]:
        if not self._use_db:
            return None
        if self._session_factory is not None:
            return self._session_factory
        try:
            from app.database import SessionLocal

            return SessionLocal
        except Exception:
            return None

    @staticmethod
    def _orm_to_client_dict(row: Any) -> Dict[str, Any]:
        updated_dt = getattr(row, "updated_at", None)
        if updated_dt is not None:
            updated_iso = (
                updated_dt if updated_dt.tzinfo else updated_dt.replace(tzinfo=timezone.utc)
            ).isoformat()
        else:
            updated_iso = datetime.now(timezone.utc).isoformat()
        return {
            "org_id": str(row.org_id),
            "display_name": str(row.display_name),
            "domain": str(row.domain or "general"),
            "application_name": str(row.application_name or row.org_id),
            "canary_token": row.canary_token,
            "intent_model": row.intent_model,
            "compression_adapter": row.compression_adapter,
            "version": str(row.version or "untrained"),
            "fidelity_thresholds": dict(row.fidelity_thresholds or {}),
            "metrics": dict(row.metrics or {}),
            "target_endpoint_url": row.target_endpoint_url,
            "target_auth_header": row.target_auth_header,
            "target_request_format": str(row.target_request_format or "openai_chat"),
            "target_model": row.target_model,
            "api_key_hash": row.api_key_hash,
            "updated_at": updated_iso,
        }

    def _db_load_clients(self) -> Dict[str, Dict[str, Any]]:
        factory = self._get_session_factory()
        if factory is None:
            return {}
        try:
            from sqlalchemy import select
            from app.models import ClientTenantRecord

            with factory() as db:
                rows = db.scalars(select(ClientTenantRecord)).all()
                return {str(r.org_id): self._orm_to_client_dict(r) for r in rows}
        except Exception as exc:
            logger.debug("ClientTenantRecord DB read skipped: %s", exc)
            return {}

    def _db_upsert_client(self, entry: Dict[str, Any]) -> bool:
        factory = self._get_session_factory()
        if factory is None:
            return False
        try:
            from app.models import ClientTenantRecord

            org_id = str(entry["org_id"])
            updated_raw = entry.get("updated_at")
            try:
                updated_dt = (
                    datetime.fromisoformat(str(updated_raw).replace("Z", "+00:00"))
                    if updated_raw
                    else datetime.now(timezone.utc)
                )
            except Exception:
                updated_dt = datetime.now(timezone.utc)
            if updated_dt.tzinfo is None:
                updated_dt = updated_dt.replace(tzinfo=timezone.utc)

            with factory() as db:
                row = db.get(ClientTenantRecord, org_id)
                if row is None:
                    row = ClientTenantRecord(
                        org_id=org_id,
                        display_name=str(entry.get("display_name") or org_id),
                        domain=str(entry.get("domain") or "general"),
                        application_name=str(entry.get("application_name") or org_id),
                        canary_token=entry.get("canary_token"),
                        intent_model=entry.get("intent_model"),
                        compression_adapter=entry.get("compression_adapter"),
                        version=str(entry.get("version") or "untrained"),
                        fidelity_thresholds=dict(entry.get("fidelity_thresholds") or {}),
                        metrics=dict(entry.get("metrics") or {}),
                        target_endpoint_url=entry.get("target_endpoint_url"),
                        target_auth_header=entry.get("target_auth_header"),
                        target_request_format=str(entry.get("target_request_format") or "openai_chat"),
                        target_model=entry.get("target_model"),
                        api_key_hash=entry.get("api_key_hash"),
                        updated_at=updated_dt,
                    )
                    db.add(row)
                else:
                    row.display_name = str(entry.get("display_name") or row.display_name)
                    row.domain = str(entry.get("domain") or row.domain)
                    row.application_name = str(entry.get("application_name") or row.application_name)
                    row.canary_token = entry.get("canary_token")
                    row.intent_model = entry.get("intent_model")
                    row.compression_adapter = entry.get("compression_adapter")
                    row.version = str(entry.get("version") or row.version)
                    row.fidelity_thresholds = dict(entry.get("fidelity_thresholds") or {})
                    row.metrics = dict(entry.get("metrics") or {})
                    row.target_endpoint_url = entry.get("target_endpoint_url")
                    row.target_auth_header = entry.get("target_auth_header")
                    row.target_request_format = str(entry.get("target_request_format") or "openai_chat")
                    row.target_model = entry.get("target_model")
                    row.api_key_hash = entry.get("api_key_hash")
                    row.updated_at = updated_dt
                db.commit()
                return True
        except Exception as exc:
            logger.warning("ClientTenantRecord DB upsert skipped: %s", exc)
            return False

    def load(self) -> Dict[str, Any]:
        """Load registry JSON merged with durable DB client records when enabled."""
        with _REGISTRY_LOCK:
            if self.registry_path.exists():
                try:
                    data = json.loads(self.registry_path.read_text(encoding="utf-8"))
                except Exception:
                    data = {}
            else:
                data = {}

            if "targets" not in data:
                data["targets"] = {
                    "coding_technical": {"min_fidelity": 0.85, "target_savings_pct": 0.50},
                    "creative_writing": {"min_fidelity": 0.87, "target_savings_pct": 0.45},
                    "analytical_qa": {"min_fidelity": 0.86, "target_savings_pct": 0.45},
                    "data_extraction_json": {"min_fidelity": 0.80, "target_savings_pct": 0.40},
                    "system_directive": {"min_fidelity": 0.80, "target_savings_pct": 0.40},
                }
            if "clients" not in data or not isinstance(data["clients"], dict):
                data["clients"] = {}

            if self._use_db:
                db_clients = self._db_load_clients()
                for oid, db_entry in db_clients.items():
                    data["clients"][oid] = db_entry
            return data

    def save(self, data: Dict[str, Any]) -> None:
        """Persist registry JSON atomically via temp file + os.replace."""
        with _REGISTRY_LOCK:
            self.registry_path.parent.mkdir(parents=True, exist_ok=True)
            data["last_updated"] = datetime.now(timezone.utc).isoformat()
            payload = json.dumps(data, indent=2)
            fd, tmp_name = tempfile.mkstemp(
                dir=str(self.registry_path.parent),
                prefix=f".{self.registry_path.name}.",
                suffix=".tmp",
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as tmp_file:
                    tmp_file.write(payload)
                os.replace(tmp_name, self.registry_path)
            except Exception:
                if os.path.exists(tmp_name):
                    os.unlink(tmp_name)
                raise

    def register_client(
        self,
        org_id: str,
        *,
        display_name: Optional[str] = None,
        domain: Optional[str] = None,
        application_name: Optional[str] = None,
        canary_token: Optional[str] = None,
        fidelity_thresholds: Optional[Dict[str, float]] = None,
        intent_model_path: Optional[str] = None,
        compression_model_path: Optional[str] = None,
        version: Optional[str] = None,
        metrics: Optional[Dict[str, Any]] = None,
        target_endpoint_url: Optional[str] = None,
        target_auth_header: Optional[str] = None,
        target_request_format: Optional[str] = None,
        target_model: Optional[str] = None,
        api_key_hash: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create or update a client organization's configuration and model references."""
        org_id = validate_org_id(org_id)
        with _REGISTRY_LOCK:
            data = self.load()
            clients = data["clients"]
            defaults = DEFAULT_CLIENT_PROFILES.get(org_id, {})
            existing = clients.get(org_id, {})

            inferred_domain = domain or existing.get("domain") or defaults.get("domain")
            if not inferred_domain:
                lower = org_id.lower()
                if "fin" in lower:
                    inferred_domain = "fintech"
                elif "med" in lower or "health" in lower:
                    inferred_domain = "healthcare"
                elif "dev" in lower or "ops" in lower or "code" in lower:
                    inferred_domain = "software"
                else:
                    inferred_domain = "general"

            merged_thresholds = dict(
                defaults.get(
                    "fidelity_thresholds",
                    {
                        "coding_technical": 0.85,
                        "data_extraction_json": 0.85,
                        "analytical_qa": 0.85,
                        "creative_writing": 0.82,
                        "system_directive": 0.85,
                    },
                )
            )
            merged_thresholds.update(existing.get("fidelity_thresholds", {}))
            if fidelity_thresholds:
                merged_thresholds.update(fidelity_thresholds)

            resolved_endpoint = (
                (target_endpoint_url.strip() or None)
                if target_endpoint_url is not None
                else existing.get("target_endpoint_url")
            )
            resolved_auth_hdr = (
                (target_auth_header.strip() or None)
                if target_auth_header is not None
                else existing.get("target_auth_header")
            )
            resolved_req_fmt = (
                (target_request_format.strip() or "openai_chat")
                if target_request_format is not None
                else (existing.get("target_request_format") or "openai_chat")
            )
            resolved_target_model = (
                (target_model.strip() or None)
                if target_model is not None
                else existing.get("target_model")
            )
            resolved_api_key_hash = (
                api_key_hash if api_key_hash is not None else existing.get("api_key_hash")
            )

            entry = {
                "org_id": org_id,
                "display_name": display_name or existing.get("display_name") or defaults.get("display_name") or org_id,
                "domain": inferred_domain,
                "application_name": (
                    application_name
                    or existing.get("application_name")
                    or defaults.get("application_name")
                    or org_id
                ),
                "canary_token": (
                    canary_token if canary_token is not None else existing.get("canary_token") or defaults.get("canary_token")
                ),
                "intent_model": intent_model_path if intent_model_path is not None else existing.get("intent_model"),
                "compression_adapter": (
                    compression_model_path if compression_model_path is not None else existing.get("compression_adapter")
                ),
                "version": version or existing.get("version") or "untrained",
                "fidelity_thresholds": merged_thresholds,
                "metrics": {**existing.get("metrics", {}), **(metrics or {})},
                "target_endpoint_url": resolved_endpoint,
                "target_auth_header": resolved_auth_hdr,
                "target_request_format": resolved_req_fmt,
                "target_model": resolved_target_model,
                "api_key_hash": resolved_api_key_hash,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            clients[org_id] = entry
            self.save(data)
            if self._use_db:
                self._db_upsert_client(entry)
            return entry

    def set_client_api_key_hash(self, org_id: str, api_key_hash: str) -> Dict[str, Any]:
        """Persist a SHA-256 tenant-scoped API key hash for `org_id`."""
        return self.register_client(org_id, api_key_hash=api_key_hash)

    def find_org_by_api_key_hash(self, api_key_hash: str) -> Optional[str]:
        """Resolve an organization ID matching the given SHA-256 API key digest (constant-time)."""
        if not api_key_hash:
            return None
        data = self.load()
        matched: Optional[str] = None
        for oid, cfg in (data.get("clients") or {}).items():
            stored = cfg.get("api_key_hash")
            if stored and hmac.compare_digest(str(stored), str(api_key_hash)):
                matched = oid
        return matched

    def register_client_models(
        self,
        org_id: str,
        intent_model_path: str,
        compression_model_path: str,
        version: str = "v1.0",
        domain: Optional[str] = None,
        fidelity_thresholds: Optional[Dict[str, float]] = None,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Register trained per-client intent & compression artifacts."""
        return self.register_client(
            org_id,
            domain=domain,
            fidelity_thresholds=fidelity_thresholds,
            intent_model_path=intent_model_path,
            compression_model_path=compression_model_path,
            version=version,
            metrics=metrics,
        )

    def get_client_config(self, org_id: Optional[str]) -> Optional[Dict[str, Any]]:
        """Retrieve a client's registered config, or default profile if predefined."""
        if not org_id:
            return None
        data = self.load()
        client = data.get("clients", {}).get(org_id)
        if client is not None:
            return client
        if org_id in DEFAULT_CLIENT_PROFILES:
            return {"org_id": org_id, **DEFAULT_CLIENT_PROFILES[org_id]}
        return None

    def list_clients(self) -> Dict[str, Dict[str, Any]]:
        """Return all registered clients combined with default baseline client profiles."""
        data = self.load()
        merged: Dict[str, Dict[str, Any]] = {}
        for org_id, profile in DEFAULT_CLIENT_PROFILES.items():
            merged[org_id] = {"org_id": org_id, "version": "untrained", **profile}
        for org_id, cfg in data.get("clients", {}).items():
            merged[org_id] = cfg
        return merged

    def get_fidelity_threshold(
        self,
        org_id: Optional[str],
        intent_value: str,
        fallback: float = 0.85,
    ) -> float:
        """Resolve per-client target semantic fidelity for an intent class."""
        cfg = self.get_client_config(org_id)
        if not cfg:
            return fallback
        thresholds = cfg.get("fidelity_thresholds") or {}
        if intent_value in thresholds:
            return float(thresholds[intent_value])
        canonical = _INTENT_ALIASES.get(intent_value)
        if canonical and canonical in thresholds:
            return float(thresholds[canonical])
        return fallback
