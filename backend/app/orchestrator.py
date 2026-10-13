import asyncio
import logging
import time
from datetime import datetime, timezone

from sqlalchemy import select

from app.analysis import analyze_response
from app.config import Settings
from app.database import SessionLocal
from app.models import Evidence, Finding, TestRun, TestRunStatus
from app.providers import ProviderRegistry
from app.providers.base import ProviderExecutionError
from app.security import PayloadCipher

logger = logging.getLogger(__name__)


class TestOrchestrator:
    """Durable queue metadata with short-lived encrypted execution payloads.

    The worker deletes plaintext-equivalent payloads as soon as a run reaches a terminal state.
    """

    __test__ = False

    def __init__(self, settings: Settings):
        self._settings = settings
        self.providers = ProviderRegistry(settings)
        self._queue: asyncio.Queue[str | None] = asyncio.Queue()
        self._workers: list[asyncio.Task] = []

    def can_accept(self, provider: str) -> bool:
        return PayloadCipher.is_configured(self._settings) and self._provider_is_configured(provider)

    def configuration_error(self, provider: str) -> str:
        if not PayloadCipher.is_configured(self._settings):
            return "Secure execution payload encryption is not configured on this server."
        try:
            self.providers.ensure_configured(provider)
        except ProviderExecutionError as exc:
            return str(exc)
        return "The test runner is unavailable."

    def _provider_is_configured(self, provider: str) -> bool:
        try:
            self.providers.ensure_configured(provider)
        except ProviderExecutionError:
            return False
        return True

    async def start(self) -> None:
        if not self._settings.run_worker:
            return
        with SessionLocal() as db:
            queued = db.scalars(
                select(TestRun.id).where(
                    TestRun.status == TestRunStatus.queued,
                    TestRun.encrypted_execution_payload.is_not(None),
                )
            ).all()
        self._workers = [asyncio.create_task(self._work(), name="ares-test-worker")]
        for test_id in queued:
            await self.enqueue(test_id)

    async def stop(self) -> None:
        for _ in self._workers:
            await self._queue.put(None)
        if self._workers:
            await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers = []

    async def enqueue(self, test_id: str) -> None:
        if self._settings.run_worker:
            await self._queue.put(test_id)

    async def _work(self) -> None:
        while True:
            test_id = await self._queue.get()
            try:
                if test_id is None:
                    return
                await self._execute(test_id)
            except Exception:
                logger.exception("Test worker recovered from an unexpected failure")
                try:
                    self._record_worker_failure(test_id)
                except Exception:
                    logger.exception("Could not mark run '%s' as failed", test_id)
            finally:
                self._queue.task_done()

    def _record_worker_failure(self, test_id: str | None) -> None:
        if test_id is None:
            return
        with SessionLocal() as db:
            run = db.get(TestRun, test_id)
            if run is None or run.status not in (TestRunStatus.queued, TestRunStatus.running):
                return
            self._fail(run, "The test runner encountered an internal error while executing this run.")
            db.commit()

    async def _execute(self, test_id: str) -> None:
        started = time.perf_counter()
        with SessionLocal() as db:
            run = db.get(TestRun, test_id)
            if run is None or run.status != TestRunStatus.queued:
                return
            expires_at = run.payload_expires_at
            if expires_at is not None and expires_at.tzinfo is None:
                expires_at = expires_at.replace(tzinfo=timezone.utc)
            if expires_at is None or expires_at <= datetime.now(timezone.utc):
                self._fail(run, "The protected execution payload expired before the test could start.")
                db.commit()
                return
            run.status = TestRunStatus.running
            run.started_at = datetime.now(timezone.utc)
            db.commit()
            try:
                payload = PayloadCipher(self._settings).decrypt(run.encrypted_execution_payload or "")
                result = await self.providers.execute(payload["provider"], payload, run.requested_by_user_id)
            except ProviderExecutionError as exc:
                db.refresh(run)
                if run.status != TestRunStatus.cancelled:
                    self._fail(run, str(exc))
                    db.commit()
                return
            except RuntimeError:
                db.refresh(run)
                if run.status != TestRunStatus.cancelled:
                    self._fail(run, "Provider execution could not complete safely. Check server-side configuration or retry.")
                    db.commit()
                return

            db.refresh(run)
            if run.status == TestRunStatus.cancelled:
                run.encrypted_execution_payload = None
                db.commit()
                return
            analysis = await analyze_response(
                result.output_text,
                payload["attack_categories"],
                attack_prompt=payload.get("user_prompt", ""),
                system_prompt=payload.get("system_prompt", ""),
                settings=self._settings,
            )
            run.findings.append(
                Finding(
                    category=payload["attack_categories"][0],
                    severity=analysis.severity,
                    risk_score=analysis.risk_score,
                    attack_succeeded=analysis.attack_succeeded,
                    runtime_classification=analysis.classification,
                    safe_summary=analysis.safe_summary,
                )
            )
            if analysis.evidence_items:
                for ev in analysis.evidence_items:
                    run.evidence.append(
                        Evidence(
                            source=ev.get("source", "ARES Evaluation"),
                            category=ev.get("category", payload["attack_categories"][0]),
                            summary=ev.get("summary", "Assessed by security engine."),
                            similarity=ev.get("similarity"),
                            redacted=True,
                        )
                    )
            else:
                run.evidence.append(
                    Evidence(
                        source=payload.get("provider", "Provider API"),
                        category=payload["attack_categories"][0],
                        summary="Provider output was assessed in memory; raw output was not retained.",
                        similarity=None,
                        redacted=True,
                    )
                )

            run.status = TestRunStatus.completed
            run.completed_at = datetime.now(timezone.utc)
            run.duration_milliseconds = int((time.perf_counter() - started) * 1000)
            run.token_estimate = result.token_estimate
            run.encrypted_execution_payload = None
            run.payload_expires_at = None
            db.commit()

    @staticmethod
    def _fail(run: TestRun, message: str) -> None:
        run.status = TestRunStatus.failed
        run.failure_reason = message
        run.completed_at = datetime.now(timezone.utc)
        run.encrypted_execution_payload = None
        run.payload_expires_at = None
