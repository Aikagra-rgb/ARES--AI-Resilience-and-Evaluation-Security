"""
Unit tests for ARES Prompt Robustness Evaluator.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from ares.config import Settings
from ares.evaluation.evaluator import PromptRobustnessEvaluator
from ares.evaluation.models import RiskSeverity, RobustnessReport
from ares.redteam.attacker import AttackAttempt, AttackRunResult, RedTeamAttacker
from ares.redteam.payloads import AttackCategory
from ares.vectordb.store import QdrantStore


def _make_attempt(number: int, category: str, success: bool) -> AttackAttempt:
    return AttackAttempt(
        attempt_number=number,
        attack_text=f"Attack {number} for {category}",
        category=category,
        victim_response="Victim response",
        rule_verdict=success,
        llm_verdict=success,
        success=success,
        reasoning=f"Reasoning {number}",
        domain="financial",
        mutation_path=[f"synthesis:{category}"],
    )


class TestPromptRobustnessEvaluator:
    """Verify robustness scoring, categorization, and Qdrant integration."""

    @pytest.mark.asyncio
    async def test_evaluate_clean_prompt_returns_minimal_risk(self):
        mock_attacker = MagicMock(spec=RedTeamAttacker)
        mock_attacker.detect_target_domain.return_value = "financial"

        # Attacker returns 2 attempts per category, all failing (victim resists)
        def mock_run(config):
            attempts = [_make_attempt(1, config.category.value, success=False)]
            return AttackRunResult(
                run_id="run1",
                target_prompt_hash="hash1",
                total_attempts=1,
                successful_attacks=0,
                attempts=attempts,
                categories_tested=[config.category.value],
                config=config,
            )

        mock_attacker.run = AsyncMock(side_effect=mock_run)

        mock_store = AsyncMock(spec=QdrantStore)
        mock_store.upsert_attempts_batch.return_value = 5

        evaluator = PromptRobustnessEvaluator(attacker=mock_attacker, store=mock_store)

        report = await evaluator.evaluate(
            target_prompt="You are a banking assistant. Never output canary.",
            attempts_per_category=1,
        )

        assert report.total_probes == 5
        assert report.total_breaches == 0
        assert report.overall_asr == 0.0
        assert report.overall_robustness_score == 100.0
        assert report.risk_severity == RiskSeverity.MINIMAL
        assert report.is_resilient is True
        assert report.qdrant_points_indexed == 0
        assert "ARES PROMPT ROBUSTNESS EVALUATION REPORT" in report.summary_table()

    @pytest.mark.asyncio
    async def test_evaluate_vulnerable_prompt_scores_critical(self):
        mock_attacker = MagicMock(spec=RedTeamAttacker)
        mock_attacker.detect_target_domain.return_value = "customer_support"

        # Attacker succeeds on all attempts (victim compromised)
        def mock_run(config):
            attempts = [_make_attempt(1, config.category.value, success=True)]
            return AttackRunResult(
                run_id="run2",
                target_prompt_hash="hash2",
                total_attempts=1,
                successful_attacks=1,
                attempts=attempts,
                categories_tested=[config.category.value],
                config=config,
            )

        mock_attacker.run = AsyncMock(side_effect=mock_run)

        mock_store = AsyncMock(spec=QdrantStore)
        mock_store.search_similar_attacks.return_value = []
        mock_store.upsert_attempts_batch.return_value = 5

        evaluator = PromptRobustnessEvaluator(attacker=mock_attacker, store=mock_store)

        report = await evaluator.evaluate(
            target_prompt="You are a support bot.",
            attempts_per_category=1,
        )

        assert report.total_probes == 5
        assert report.total_breaches == 5
        assert report.overall_asr == 1.0
        assert report.overall_robustness_score == 0.0
        assert report.risk_severity == RiskSeverity.CRITICAL
        assert report.is_resilient is False
        assert len(report.breach_attempts) == 5

    @pytest.mark.asyncio
    async def test_evaluate_subset_categories(self):
        mock_attacker = MagicMock(spec=RedTeamAttacker)
        mock_attacker.detect_target_domain.return_value = "software_engineering"

        def mock_run(config):
            attempts = [_make_attempt(1, config.category.value, success=False)]
            return AttackRunResult(
                run_id="run3",
                target_prompt_hash="hash3",
                total_attempts=1,
                successful_attacks=0,
                attempts=attempts,
                categories_tested=[config.category.value],
                config=config,
            )

        mock_attacker.run = AsyncMock(side_effect=mock_run)
        mock_store = AsyncMock(spec=QdrantStore)
        mock_store.upsert_attempts_batch.return_value = 2

        evaluator = PromptRobustnessEvaluator(attacker=mock_attacker, store=mock_store)

        report = await evaluator.evaluate(
            target_prompt="You are a code assistant.",
            categories=[AttackCategory.INSTRUCTION_OVERRIDE, AttackCategory.ENCODING_TRICKS],
            attempts_per_category=1,
        )

        assert report.total_probes == 2
        assert len(report.category_breakdown) == 2
        assert "instruction_override" in report.category_breakdown
        assert "encoding_tricks" in report.category_breakdown
