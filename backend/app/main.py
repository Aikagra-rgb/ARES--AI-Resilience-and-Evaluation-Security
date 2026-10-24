import hashlib
import secrets
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Dict, Tuple

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from sqlalchemy import Integer, func, select
from sqlalchemy.orm import Session, selectinload
from sqlalchemy.exc import OperationalError, DisconnectionError

from app.config import get_settings
from app.database import Base, engine, get_db
from app.dependencies import CurrentUser, get_current_user
from app.arena import CHALLENGES, CHALLENGE_BY_ID, PATHS, ROOMS, score_run
from app.models import ArenaSubmission, AuditLog, Finding, Policy, Project, TestRun, TestRunStatus
from app.orchestrator import TestOrchestrator
from app.schemas import (
    CancelTestResponse,
    CreateTestRequest,
    CreateTestResponse,
    DashboardSummary,
    PolicyCreate,
    PolicyResponse,
    ProjectCreate,
    ProjectResponse,
    RecentTestsResponse,
    TestRunResponse,
    ArenaSubmissionCreate,
)
from app.security import PayloadCipher, payload_expiry
from ares.api.dependencies import verify_api_key
from ares.api.routes import (
    corpus as ares_corpus,
    dashboard as ares_dashboard,
    gateway as ares_gateway,
    harden as ares_harden,
    search as ares_search,
    stats as ares_stats,
    tests as ares_tests,
    models as ares_models,
    moderation as ares_moderation,
    optimizer as ares_optimizer,
)
from ares.observability.metrics import (
    get_metrics,
    get_metrics_output,
    MetricsMiddleware,
)
from ares.gateway.enforcer import RuntimeEnforcer
from ares.security.rate_limiter import MemoryRateLimiter, RateLimitMiddleware
from ares.security.signing import HmacVerifier, HmacVerificationMiddleware
from ares.observability.logging import set_correlation_id, set_organization_id

settings = get_settings()
orchestrator = TestOrchestrator(settings)

# Cache single RuntimeEnforcer in FastAPI.state / lifespan context
# so models and benign vectors are not reloaded on every request.
_runtime_enforcer: RuntimeEnforcer | None = None


def get_runtime_enforcer() -> RuntimeEnforcer:
    """Get or create the cached RuntimeEnforcer instance."""
    global _runtime_enforcer
    if _runtime_enforcer is None:
        _runtime_enforcer = RuntimeEnforcer(settings=settings)
    return _runtime_enforcer


def correlation_id(request: Request) -> str:
    return getattr(request.state, "correlation_id", "ares-" + secrets.token_hex(6))


def api_error(
    request: Request, status_code: int, code: str, message: str, *, validation_errors: dict | None = None
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "code": code,
            "message": message,
            "correlation_id": correlation_id(request),
            "validation_errors": validation_errors,
            "retryable": False,
        },
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    if settings.auto_create_schema:
        try:
            Base.metadata.create_all(bind=engine)
        except Exception:
            pass
    app.state.orchestrator = orchestrator
    if settings.environment != "test":
        try:
            await orchestrator.start()
        except Exception:
            pass
    try:
        yield
    finally:
        if settings.environment != "test":
            try:
                await orchestrator.stop()
            except Exception:
                pass


app = FastAPI(title="ARES Security API", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rate_limiter = MemoryRateLimiter()
app.add_middleware(
    RateLimitMiddleware,
    limiter=rate_limiter,
    enabled=getattr(settings, "rate_limit_enabled", True),
    requests_per_window=getattr(settings, "rate_limit_requests", 120),
    window_seconds=getattr(settings, "rate_limit_window_seconds", 60),
)

hmac_verifier = HmacVerifier(
    default_secret=getattr(settings, "hmac_secret", settings.api_key),
    max_clock_skew_seconds=getattr(settings, "hmac_max_clock_skew", 300),
)
app.add_middleware(
    HmacVerificationMiddleware,
    verifier=hmac_verifier,
    enforce_mode=getattr(settings, "hmac_enforce_mode", "permissive"),
)

# Unified ARES Vector Security, ML Analytics, Hardening, and Gateway Routers.
# Every one of these routes is protected by API-key authentication; only the health
# probes below are public.
_auth = [Depends(verify_api_key)]
app.include_router(ares_dashboard.router, dependencies=_auth)
app.include_router(ares_stats.router, dependencies=_auth)
app.include_router(ares_search.router, dependencies=_auth)
app.include_router(ares_harden.router, dependencies=_auth)
app.include_router(ares_corpus.router, dependencies=_auth)
app.include_router(ares_tests.router, dependencies=_auth)
app.include_router(ares_gateway.router, dependencies=_auth)
app.include_router(ares_models.router, dependencies=_auth)
app.include_router(ares_moderation.router, dependencies=_auth)
app.include_router(ares_optimizer.router, dependencies=_auth)

# Explicit v1 aliases for frontend compatibility
@app.get("/api/v1/stats", tags=["stats"], dependencies=_auth)
async def v1_stats():
    return await ares_stats.get_stats()

@app.post("/api/v1/stats/retrain", tags=["stats"], dependencies=_auth)
async def v1_stats_retrain():
    return await ares_stats.retrain_stats()

@app.post("/api/v1/search", tags=["search"], dependencies=_auth)
async def v1_search(req: ares_search.SearchRequest):
    return await ares_search.search_attacks(req)

@app.post("/api/v1/harden", tags=["harden"], dependencies=_auth)
async def v1_harden(req: ares_harden.HardenRequest):
    return await ares_harden.harden_prompt(req)


# Prometheus metrics endpoint - secured behind API key auth
@app.get("/metrics", tags=["observability"], dependencies=_auth)
async def metrics_endpoint():
    """Prometheus metrics endpoint (requires API key)."""
    return Response(content=get_metrics_output(), media_type="text/plain")


@app.get("/healthz", tags=["health"])
@app.get("/health", tags=["health"])
async def healthz() -> dict:
    """Unified service health probe with dependency checks."""
    from app.database import engine
    from sqlalchemy import text
    
    checks = {}
    overall_status = "ok"
    
    # Check database
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        checks["database"] = {"status": "ok", "detail": "Connection successful"}
    except Exception:
        checks["database"] = {"status": "error", "detail": "Connection failed"}
        overall_status = "degraded"
    
    # Check Qdrant (if configured)
    try:
        if settings.qdrant_url:
            from ares.vectordb.store import QdrantStore
            store = QdrantStore(settings=settings)
            collections = store.client.get_collections()
            checks["qdrant"] = {"status": "ok", "detail": f"{len(collections.collections)} collections"}
        else:
            checks["qdrant"] = {"status": "not_configured", "detail": "QDRANT_URL not set"}
    except Exception:
        checks["qdrant"] = {"status": "error", "detail": "Connection failed"}
        overall_status = "degraded"
    
    # Check LLM providers (basic config check)
    llm_status = []
    if settings.groq_api_key:
        llm_status.append("groq")
    if settings.gemini_api_key:
        llm_status.append("gemini")
    if settings.openai_api_key:
        llm_status.append("openai")
    if settings.nvidia_api_key:
        llm_status.append("nvidia")
    
    checks["llm_providers"] = {
        "status": "ok" if llm_status else "not_configured",
        "configured": llm_status,
    }
    
    return {
        "status": overall_status,
        "service": "ARES Unified API",
        "version": "0.1.0",
        "checks": checks,
    }


@app.middleware("http")
async def attach_correlation_id(request: Request, call_next):
    corr_id = request.headers.get("X-Correlation-ID") or "ares-" + secrets.token_hex(6)
    org_id = request.headers.get("X-Ares-Organization-Id")
    request.state.correlation_id = corr_id
    request.state.organization_id = org_id
    set_correlation_id(corr_id)
    set_organization_id(org_id)
    try:
        response = await call_next(request)
    finally:
        set_correlation_id(None)
        set_organization_id(None)
    response.headers["X-Correlation-ID"] = corr_id
    return response


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    errors: dict[str, list[str]] = {}
    for error in exc.errors():
        field = (
            ".".join(str(part) for part in error["loc"] if part not in {"body", "query", "path"}) or "request"
        )
        errors.setdefault(field, []).append(error["msg"])
    return api_error(
        request, 422, "validation_failed", "One or more fields need attention.", validation_errors=errors
    )


@app.exception_handler(HTTPException)
async def http_error_handler(request: Request, exc: HTTPException):
    code = {401: "unauthorized", 403: "forbidden", 404: "not_found", 409: "invalid_transition"}.get(
        exc.status_code, "request_failed"
    )
    return api_error(request, exc.status_code, code, str(exc.detail))


def public_configuration(configuration) -> dict:
    """Only metadata is persisted; raw prompts remain request-scoped."""
    return {
        "test_name": configuration.test_name,
        "target_application": configuration.target_application,
        "provider": configuration.provider,
        "model": configuration.model,
        "attack_source": configuration.attack_source,
        "attack_categories": configuration.attack_categories,
        "temperature": configuration.temperature,
        "max_response_tokens": configuration.max_response_tokens,
        "variation_count": configuration.variation_count,
        "include_hardened_comparison": configuration.include_hardened_comparison,
    }


def fingerprint(configuration) -> str:
    material = f"{configuration.system_prompt}\x00{configuration.user_prompt}".encode()
    return hashlib.sha256(settings.prompt_fingerprint_secret.encode() + material).hexdigest()


def require_project(db: Session, user: CurrentUser, project_id: str | None) -> Project:
    query = select(Project).where(Project.organization_id == user.organization_id)
    if project_id:
        query = query.where(Project.id == project_id)
    project = db.scalar(query.order_by(Project.created_at).limit(1))
    if project is None:
        raise HTTPException(status_code=404, detail="Project was not found in your organization.")
    return project


def load_test(db: Session, user: CurrentUser, test_id: str) -> TestRun:
    run = db.scalar(
        select(TestRun)
        .options(selectinload(TestRun.findings), selectinload(TestRun.evidence))
        .where(TestRun.id == test_id, TestRun.organization_id == user.organization_id)
    )
    if run is None:
        raise HTTPException(status_code=404, detail="Test run was not found.")
    return run


def test_response(run: TestRun) -> TestRunResponse:
    return TestRunResponse(
        id=run.id,
        project_id=run.project_id,
        status=run.status.value,
        created_at=run.created_at,
        completed_at=run.completed_at,
        duration_milliseconds=run.duration_milliseconds,
        token_estimate=run.token_estimate,
        correlation_id=run.correlation_id,
        failure_reason=run.failure_reason,
        configuration=run.configuration,
        findings=[
            {
                "id": item.id,
                "category": item.category,
                "severity": item.severity.value,
                "risk_score": item.risk_score,
                "attack_succeeded": item.attack_succeeded,
                "runtime_classification": item.runtime_classification,
                "summary": item.safe_summary,
            }
            for item in run.findings
        ],
        evidence=[
            {
                "id": item.id,
                "source": item.source,
                "category": item.category,
                "summary": item.summary,
                "similarity": item.similarity,
                "redacted": item.redacted,
            }
            for item in run.evidence
        ],
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "ares-api"}


@app.get("/api/v1/projects", response_model=list[ProjectResponse])
def list_projects(user: CurrentUser = Depends(get_current_user), db: Session = Depends(get_db)):
    rows = db.scalars(
        select(Project).where(Project.organization_id == user.organization_id).order_by(Project.created_at)
    ).all()
    return [
        ProjectResponse(id=row.id, name=row.name, description=row.description, created_at=row.created_at)
        for row in rows
    ]


@app.post("/api/v1/projects", response_model=ProjectResponse, status_code=201)
def create_project(
    payload: ProjectCreate, user: CurrentUser = Depends(get_current_user), db: Session = Depends(get_db)
):
    existing = db.scalar(
        select(Project).where(Project.organization_id == user.organization_id, Project.name == payload.name)
    )
    if existing:
        raise HTTPException(status_code=409, detail="A project with this name already exists.")
    project = Project(
        organization_id=user.organization_id,
        created_by_user_id=user.id,
        name=payload.name,
        description=payload.description,
    )
    db.add(project)
    db.commit()
    db.refresh(project)
    return ProjectResponse(
        id=project.id, name=project.name, description=project.description, created_at=project.created_at
    )


@app.post("/api/v1/projects/{project_id}/policies", response_model=PolicyResponse, status_code=201)
def create_policy(
    project_id: str,
    payload: PolicyCreate,
    user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    project = require_project(db, user, project_id)
    policy = Policy(project_id=project.id, name=payload.name, rules=payload.rules)
    db.add(policy)
    db.commit()
    db.refresh(policy)
    return PolicyResponse(
        id=policy.id,
        project_id=policy.project_id,
        name=policy.name,
        version=policy.version,
        is_active=policy.is_active,
        rules=policy.rules,
        created_at=policy.created_at,
    )


@app.post("/api/v1/tests", response_model=CreateTestResponse, status_code=201)
async def create_test(
    payload: CreateTestRequest,
    request: Request,
    user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    project = require_project(db, user, payload.project_id)
    if not orchestrator.can_accept(payload.configuration.provider):
        return api_error(
            request,
            503,
            "orchestration_unavailable",
            orchestrator.configuration_error(payload.configuration.provider),
        )
    execution_payload = payload.configuration.model_dump()
    run = TestRun(
        organization_id=user.organization_id,
        project_id=project.id,
        requested_by_user_id=user.id,
        status=TestRunStatus.queued,
        configuration=public_configuration(payload.configuration),
        prompt_fingerprint=fingerprint(payload.configuration),
        encrypted_execution_payload=PayloadCipher(settings).encrypt(execution_payload),
        payload_expires_at=payload_expiry(),
        correlation_id=correlation_id(request),
    )
    db.add(run)
    db.commit()
    db.refresh(run)
    await orchestrator.enqueue(run.id)
    return CreateTestResponse(
        test_id=run.id,
        project_id=run.project_id,
        status=run.status.value,
        correlation_id=run.correlation_id,
        created_at=run.created_at,
    )


@app.get("/api/v1/tests/recent", response_model=RecentTestsResponse)
def recent_tests(
    limit: int = 20, user: CurrentUser = Depends(get_current_user), db: Session = Depends(get_db)
):
    if not 1 <= limit <= 100:
        raise HTTPException(status_code=422, detail="limit must be between 1 and 100.")
    rows = db.scalars(
        select(TestRun)
        .options(selectinload(TestRun.findings))
        .where(TestRun.organization_id == user.organization_id)
        .order_by(TestRun.created_at.desc())
        .limit(limit)
    ).all()
    items = [
        {
            "id": row.id,
            "timestamp": row.created_at,
            "project_id": row.project_id,
            "category": row.configuration["attack_categories"][0] if row.configuration.get("attack_categories") else "instruction_override",
            "provider": row.configuration.get("provider", "nvidia_nim"),
            "model": row.configuration.get("model", "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"),
            "risk_score": max((f.risk_score for f in row.findings), default=0),
            "status": row.status.value,
        }
        for row in rows
    ]
    if not items:
        from pathlib import Path
        import json
        nemotron_path = Path(__file__).parent.parent / "nemotron_robustness_report.json"
        if nemotron_path.exists():
            try:
                ndata = json.loads(nemotron_path.read_text(encoding="utf-8"))
                ts_str = ndata.get("timestamp", datetime.now(timezone.utc).isoformat())
                try:
                    ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                except Exception:
                    ts = datetime.now(timezone.utc)
                asr = ndata.get("overall_asr", 0.0)
                items.append({
                    "id": ndata.get("report_id", "eval-nemotron-redteam"),
                    "timestamp": ts,
                    "project_id": "redteam-background-suite",
                    "category": "instruction_override",
                    "provider": "nvidia_nim",
                    "model": ndata.get("victim_model", "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"),
                    "risk_score": int(asr * 100),
                    "status": "completed",
                })
            except Exception:
                pass
    return RecentTestsResponse(items=items)


@app.get("/api/v1/tests/{test_id}", response_model=TestRunResponse)
def get_test(test_id: str, user: CurrentUser = Depends(get_current_user), db: Session = Depends(get_db)):
    return test_response(load_test(db, user, test_id))


@app.post("/api/v1/tests/{test_id}/cancel", response_model=CancelTestResponse)
def cancel_test(
    test_id: str,
    request: Request,
    user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    run = load_test(db, user, test_id)
    if run.status in {TestRunStatus.queued, TestRunStatus.running}:
        run.status = TestRunStatus.cancelled
        run.completed_at = datetime.now(timezone.utc)
        db.commit()
        db.refresh(run)
    return CancelTestResponse(test_id=run.id, status=run.status.value, correlation_id=correlation_id(request))


@app.get("/api/v1/dashboard/summary", response_model=DashboardSummary)
def dashboard_summary(user: CurrentUser = Depends(get_current_user), db: Session = Depends(get_db)):
    runs = (
        db.scalar(select(func.count(TestRun.id)).where(TestRun.organization_id == user.organization_id)) or 0
    )
    blocked = (
        db.scalar(
            select(func.count(TestRun.id)).where(
                TestRun.organization_id == user.organization_id, TestRun.status == TestRunStatus.blocked
            )
        )
        or 0
    )
    projects = (
        db.scalar(select(func.count(Project.id)).where(Project.organization_id == user.organization_id)) or 0
    )
    findings = (
        db.scalar(
            select(func.count(Finding.id))
            .join(TestRun)
            .where(TestRun.organization_id == user.organization_id)
        )
        or 0
    )

    # Attack success rate: % of findings where the attack succeeded
    succeeded = (
        db.scalar(
            select(func.count(Finding.id))
            .join(TestRun)
            .where(
                TestRun.organization_id == user.organization_id,
                Finding.attack_succeeded.is_(True),
            )
        )
        or 0
    )
    success_rate = round(succeeded * 100 / findings) if findings > 0 else 0
    success_severity = "Safe" if success_rate == 0 else ("Low" if success_rate < 25 else ("Medium" if success_rate < 50 else "High"))

    # Average risk score across all findings
    avg_risk = (
        db.scalar(
            select(func.avg(Finding.risk_score))
            .join(TestRun)
            .where(TestRun.organization_id == user.organization_id)
        )
        or 0.0
    )
    avg_risk_int = round(avg_risk)
    risk_severity = "Safe" if avg_risk_int < 20 else ("Low" if avg_risk_int < 40 else ("Medium" if avg_risk_int < 60 else ("High" if avg_risk_int < 80 else "Critical")))

    return DashboardSummary(
        metrics=[
            {
                "label": "Tests executed",
                "value": str(runs),
                "change": "Live data",
                "trend": "flat",
                "status": "Safe",
                "description": "Controlled test runs in this organisation",
            },
            {
                "label": "Tests blocked",
                "value": str(blocked),
                "change": "Live data",
                "trend": "flat",
                "status": "Safe",
                "description": "Runs blocked by enforcement",
            },
            {
                "label": "Attack success rate",
                "value": f"{success_rate}%",
                "change": f"{succeeded} of {findings} findings",
                "trend": "up" if success_rate > 0 else "flat",
                "status": success_severity,
                "description": "Percentage of completed tests where the attack succeeded",
            },
            {
                "label": "Avg risk score",
                "value": str(avg_risk_int),
                "change": "Across all findings",
                "trend": "up" if avg_risk_int > 50 else "flat",
                "status": risk_severity,
                "description": "Mean risk score across all security findings (0–100)",
            },
        ],
        corpus_size=findings,
        protected_applications=projects,
        generated_at=datetime.now(timezone.utc),
    )


@app.get("/api/v1/dashboard/trends")
def dashboard_trends(user: CurrentUser = Depends(get_current_user), db: Session = Depends(get_db)) -> dict:
    from datetime import date, timedelta

    today = date.today()
    window_start = datetime(today.year, today.month, today.day, tzinfo=timezone.utc) - timedelta(days=13)

    # Fetch all runs in the 14-day window for this org (with findings eager-loaded)
    runs = db.scalars(
        select(TestRun)
        .options(selectinload(TestRun.findings))
        .where(
            TestRun.organization_id == user.organization_id,
            TestRun.created_at >= window_start,
        )
    ).all()

    # Build a day-keyed dictionary covering all 14 days (so days with no runs still appear)
    points: dict[date, dict] = {
        today - timedelta(days=i): {"tested": 0, "blocked": 0, "successful": 0, "incidents": 0}
        for i in range(13, -1, -1)
    }

    for run in runs:
        run_date = run.created_at.date()
        if run_date not in points:
            continue
        bucket = points[run_date]
        bucket["tested"] += 1
        if run.status == TestRunStatus.blocked:
            bucket["blocked"] += 1
        if any(f.attack_succeeded for f in run.findings):
            bucket["successful"] += 1
        # Runtime incidents come from the enforcement gateway (future). Zero for now.

    return {
        "points": [
            {
                "date": day.isoformat(),
                "tested": bucket["tested"],
                "blocked": bucket["blocked"],
                "successful": bucket["successful"],
                "incidents": bucket["incidents"],
            }
            for day, bucket in points.items()
        ]
    }


@app.get("/api/v1/dashboard/categories")
def dashboard_categories(
    user: CurrentUser = Depends(get_current_user), db: Session = Depends(get_db)
) -> dict:
    rows = db.execute(
        select(
            Finding.category,
            func.count(Finding.id),
            func.sum(func.cast(Finding.attack_succeeded, Integer)),
        )
        .join(TestRun)
        .where(TestRun.organization_id == user.organization_id)
        .group_by(Finding.category)
    ).all()
    items = []
    for category, tests, successful in rows:
        success_count = int(successful or 0)
        total = int(tests)
        items.append(
            {
                "category": category,
                "tests": total,
                "successful": success_count,
                "blocked": 0,
                "success_rate": round(success_count * 100 / total) if total else 0,
            }
        )
    return {"items": items}


@app.get("/api/v1/incidents/recent", dependencies=[Depends(get_current_user)])
def recent_incidents() -> dict:
    # Runtime enforcement incidents are added with the enforcement gateway, not fabricated here.
    return {"items": [], "next_cursor": None}


@app.get("/api/v1/hardening/comparison")
def hardening_comparison(user: CurrentUser = Depends(get_current_user), db: Session = Depends(get_db)) -> dict:
    from datetime import date

    # Fetch all completed runs for this org with their findings
    runs = db.scalars(
        select(TestRun)
        .options(selectinload(TestRun.findings))
        .where(
            TestRun.organization_id == user.organization_id,
            TestRun.status == TestRunStatus.completed,
        )
    ).all()

    if not runs:
        from pathlib import Path
        import json
        report_path = Path(__file__).parent.parent / "prompt_hardening_report.json"
        if report_path.exists():
            try:
                data = json.loads(report_path.read_text(encoding="utf-8"))
                if "baseline_attack_success_rate" in data:
                    base = int(round(float(data["baseline_attack_success_rate"]) * 100))
                elif "baseline_robustness" in data:
                    base = max(0, int(round(100.0 - float(data["baseline_robustness"]))))
                else:
                    base = 0

                if "hardened_attack_success_rate" in data:
                    hard = int(round(float(data["hardened_attack_success_rate"]) * 100))
                elif "hardened_robustness" in data:
                    hard = max(0, int(round(100.0 - float(data["hardened_robustness"]))))
                else:
                    hard = 0

                delta = int(round(float(data.get("robustness_delta", max(0, base - hard)))))
                raw_ts = data.get("timestamp")
                return {
                    "baseline_success_rate": base,
                    "hardened_success_rate": hard,
                    "improvement_points": max(0, delta),
                    "tests_included": int(data.get("total_iterations", data.get("iterations_run", 0))),
                    "last_cycle": str(raw_ts)[:10] if raw_ts else None,
                }
            except Exception:
                pass
        return {
            "baseline_success_rate": 0,
            "hardened_success_rate": 0,
            "improvement_points": 0,
            "tests_included": 0,
            "last_cycle": None,
        }

    # Baseline = runs that did NOT request a hardened comparison
    baseline_runs = [r for r in runs if not r.configuration.get("include_hardened_comparison", False)]
    # Hardened = runs that DID request a hardened comparison
    hardened_runs = [r for r in runs if r.configuration.get("include_hardened_comparison", False)]

    def success_rate(run_list: list) -> int:
        if not run_list:
            return 0
        succeeded = sum(1 for r in run_list if any(f.attack_succeeded for f in r.findings))
        return round(succeeded * 100 / len(run_list))

    baseline_rate = success_rate(baseline_runs or runs)  # fall back to all runs if no split
    hardened_rate = success_rate(hardened_runs) if hardened_runs else 0
    improvement = max(0, baseline_rate - hardened_rate) if hardened_runs else 0

    last_completed = max(
        (r.completed_at for r in runs if r.completed_at), default=None
    )

    return {
        "baseline_success_rate": baseline_rate,
        "hardened_success_rate": hardened_rate,
        "improvement_points": improvement,
        "tests_included": len(runs),
        "last_cycle": last_completed.date().isoformat() if last_completed else None,
    }


@app.get("/api/v1/providers/health", dependencies=[Depends(get_current_user)])
def provider_health() -> dict:
    now = datetime.now(timezone.utc)
    openai_configured = orchestrator._provider_is_configured("OpenAI")
    groq_configured = orchestrator._provider_is_configured("Groq")
    ollama_configured = orchestrator._provider_is_configured("Ollama")
    return {
        "items": [
            {
                "provider": "OpenAI",
                "name": "OpenAI",
                "status": "Operational" if openai_configured else "NotConfigured",
                "detail": "Server-side Responses adapter configured."
                if openai_configured
                else "Server-side credential is not configured.",
                "checked_at": now,
            },
            {
                "provider": "Groq",
                "name": "Groq",
                "status": "Operational" if groq_configured else "NotConfigured",
                "detail": "High-speed LPU inference adapter configured."
                if groq_configured
                else "GROQ_API_KEY is not configured.",
                "checked_at": now,
            },
            {
                "provider": "Ollama",
                "name": "Ollama (Local GPU)",
                "status": "Operational" if ollama_configured else "NotConfigured",
                "detail": f"Local GPU daemon configured at {settings.ollama_base_url}."
                if ollama_configured
                else "Local Ollama daemon is not configured.",
                "checked_at": now,
            },
        ]
    }


@app.get("/api/v1/models", dependencies=[Depends(get_current_user)])
def models(provider: str) -> dict:
    p_norm = provider.lower().replace("_", "")
    items = []
    if p_norm in ("openai", "open_ai"):
        items = [
            {"provider": "OpenAI", "id": "gpt-4.1-mini", "display_name": "GPT-4.1 mini", "context_window": 128000, "is_available": orchestrator._provider_is_configured("OpenAI")},
            {"provider": "OpenAI", "id": "gpt-4o-mini", "display_name": "GPT-4o mini", "context_window": 128000, "is_available": orchestrator._provider_is_configured("OpenAI")},
        ]
    elif p_norm == "groq":
        is_avail = orchestrator._provider_is_configured("Groq")
        items = [
            {"provider": "Groq", "id": "llama-3.1-8b-instant", "display_name": "Llama 3.1 8B Instant", "context_window": 128000, "is_available": is_avail},
            {"provider": "Groq", "id": "llama-3.1-70b-versatile", "display_name": "Llama 3.1 70B Versatile", "context_window": 128000, "is_available": is_avail},
            {"provider": "Groq", "id": "mixtral-8x7b-32768", "display_name": "Mixtral 8×7B", "context_window": 32768, "is_available": is_avail},
        ]
    elif p_norm in ("ollama", "local"):
        is_avail = orchestrator._provider_is_configured("Ollama")
        items = [
            {"provider": "Ollama", "id": "llama3.1:8b", "display_name": "Llama 3.1 8B (Local RTX)", "context_window": 128000, "is_available": is_avail},
            {"provider": "Ollama", "id": "mistral:latest", "display_name": "Mistral 7B (Local RTX)", "context_window": 32768, "is_available": is_avail},
            {"provider": "Ollama", "id": "qwen2.5:7b", "display_name": "Qwen 2.5 7B (Local RTX)", "context_window": 128000, "is_available": is_avail},
            {"provider": "Ollama", "id": "deepseek-r1:8b", "display_name": "DeepSeek R1 8B (Local Reasoning)", "context_window": 65536, "is_available": is_avail},
        ]
    elif p_norm == "gemini":
        items = [
            {"provider": "Gemini", "id": "gemini-2.0-flash", "display_name": "Gemini 2.0 Flash", "context_window": 1048576, "is_available": bool(settings.gemini_api_key)},
        ]
    elif p_norm in ("nvidianim", "nvidia"):
        items = [
            {"provider": "NvidiaNim", "id": "nvidia/nemotron-4-340b-instruct", "display_name": "Nemotron 4 340B", "context_window": 4096, "is_available": True},
            {"provider": "NvidiaNim", "id": "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning", "display_name": "Nemotron 3 Nano Omni 30B", "context_window": 32768, "is_available": True},
        ]
    return {"items": items}


def arena_progress(user: CurrentUser, db: Session) -> list[dict]:
    submissions = db.scalars(
        select(ArenaSubmission)
        .where(ArenaSubmission.user_id == user.id)
        .order_by(ArenaSubmission.submitted_at.desc())
    ).all()
    by_challenge: dict[str, list[ArenaSubmission]] = {}
    for submission in submissions:
        by_challenge.setdefault(submission.challenge_id, []).append(submission)
    progress = []
    for challenge in CHALLENGES:
        attempts = by_challenge.get(challenge["id"], [])
        best = max((int(item.score["total"]) for item in attempts), default=None)
        progress.append({
            "challenge_id": challenge["id"], "title": challenge["title"], "track": challenge["track"],
            "tier": challenge["tier"], "status": "Completed" if attempts else ("Locked" if challenge["is_room_locked"] else "Available"),
            "best_score": best, "last_attempt_at": attempts[0].submitted_at if attempts else None,
        })
    return progress


@app.get("/api/v1/arena/challenges")
def arena_challenges(
    track: str | None = None, tier: str | None = None, category: str | None = None,
    search: str | None = None, page: int = 1,
    user: CurrentUser = Depends(get_current_user), db: Session = Depends(get_db),
) -> dict:
    if page < 1:
        raise HTTPException(status_code=422, detail="page must be at least 1.")
    rows = CHALLENGES
    if track: rows = [item for item in rows if item["track"] == track]
    if tier: rows = [item for item in rows if item["tier"] == tier]
    if category: rows = [item for item in rows if item["category"] == category]
    if search:
        needle = search.lower()
        rows = [item for item in rows if needle in item["title"].lower() or needle in item["description"].lower()]
    page_size = 10
    return {"items": rows[(page - 1) * page_size:page * page_size], "progress": arena_progress(user, db), "total_count": len(rows), "page": page, "page_size": page_size}


@app.get("/api/v1/arena/challenges/{challenge_id}")
def arena_challenge(challenge_id: str, user: CurrentUser = Depends(get_current_user)) -> dict:
    challenge = CHALLENGE_BY_ID.get(challenge_id)
    if challenge is None:
        raise HTTPException(status_code=404, detail="Challenge was not found.")
    return challenge


@app.get("/api/v1/arena/rooms")
def arena_rooms(user: CurrentUser = Depends(get_current_user)) -> dict:
    return {"items": ROOMS}


@app.get("/api/v1/arena/rooms/{room_id}")
def arena_room(room_id: str, user: CurrentUser = Depends(get_current_user)) -> dict:
    room = next((item for item in ROOMS if item["id"] == room_id), None)
    if room is None:
        raise HTTPException(status_code=404, detail="Room was not found.")
    return room


@app.get("/api/v1/arena/paths")
def arena_paths(user: CurrentUser = Depends(get_current_user)) -> dict:
    return {"items": PATHS}


@app.get("/api/v1/arena/progress")
def arena_my_progress(user: CurrentUser = Depends(get_current_user), db: Session = Depends(get_db)) -> dict:
    return {"items": arena_progress(user, db)}


@app.get("/api/v1/arena/profile")
def arena_profile(user: CurrentUser = Depends(get_current_user), db: Session = Depends(get_db)) -> dict:
    progress = arena_progress(user, db)
    completed = [item for item in progress if item["status"] == "Completed"]
    attacker = sum(item["track"] == "Attacker" for item in completed)
    defender = len(completed) - attacker
    xp = sum(item["best_score"] or 0 for item in completed)
    return {"id": user.id, "display_name": user.email.split("@")[0], "initials": user.email[:2].upper(), "xp_total": xp, "level": xp // 500 + 1, "xp_this_level": xp % 500, "xp_to_next_level": 500, "badges": [], "challenges_solved": len(completed), "attacker_solved": attacker, "defender_solved": defender, "member_since": datetime.now(timezone.utc)}


@app.post("/api/v1/arena/submissions", status_code=201)
def arena_submit(
    payload: ArenaSubmissionCreate, user: CurrentUser = Depends(get_current_user), db: Session = Depends(get_db)
) -> dict:
    challenge = CHALLENGE_BY_ID.get(payload.challenge_id)
    if challenge is None:
        raise HTTPException(status_code=404, detail="Challenge was not found.")
    run = db.scalar(
        select(TestRun)
        .options(selectinload(TestRun.findings), selectinload(TestRun.evidence))
        .where(TestRun.id == payload.test_run_id, TestRun.requested_by_user_id == user.id)
    )
    if run is None:
        raise HTTPException(status_code=404, detail="Controlled test run was not found.")
    if run.status not in {TestRunStatus.completed, TestRunStatus.blocked}:
        raise HTTPException(status_code=409, detail="Complete the controlled test before submitting it.")
    existing = db.scalar(select(ArenaSubmission).where(ArenaSubmission.user_id == user.id, ArenaSubmission.challenge_id == payload.challenge_id, ArenaSubmission.test_run_id == run.id))
    if existing is None:
        existing = ArenaSubmission(
            user_id=user.id,
            challenge_id=payload.challenge_id,
            test_run_id=run.id,
            score=score_run(run, challenge["track"], challenge=challenge),
        )
        db.add(existing)
        db.commit()
        db.refresh(existing)
    attempts = db.scalars(select(ArenaSubmission).where(ArenaSubmission.user_id == user.id, ArenaSubmission.challenge_id == payload.challenge_id)).all()
    best = max(int(item.score["total"]) for item in attempts)
    next_challenge = next((item["id"] for item in CHALLENGES if item["track"] == challenge["track"] and item["id"] != challenge["id"]), None)
    return {"submission": {"id": existing.id, "challenge_id": existing.challenge_id, "user_id": existing.user_id, "test_run_id": existing.test_run_id, "score": existing.score, "submitted_at": existing.submitted_at, "is_best": int(existing.score["total"]) == best}, "badge_unlocked": False, "badge": None, "next_challenge_id": next_challenge}


@app.get("/api/v1/arena/challenges/{challenge_id}/submissions")
def arena_submissions(challenge_id: str, user: CurrentUser = Depends(get_current_user), db: Session = Depends(get_db)) -> dict:
    rows = db.scalars(select(ArenaSubmission).where(ArenaSubmission.user_id == user.id, ArenaSubmission.challenge_id == challenge_id).order_by(ArenaSubmission.submitted_at.desc())).all()
    best = max((int(item.score["total"]) for item in rows), default=0)
    return {"items": [{"id": item.id, "challenge_id": item.challenge_id, "user_id": item.user_id, "test_run_id": item.test_run_id, "score": item.score, "submitted_at": item.submitted_at, "is_best": int(item.score["total"]) == best} for item in rows]}
