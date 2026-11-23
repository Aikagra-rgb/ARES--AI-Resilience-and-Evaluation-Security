"""Automated red-team evaluation harness for ARES operator-controlled targets."""

from .assertions import assert_allowed, assert_blocked, assert_sanitized
from .attack_generator import TargetAttackCase, TargetAttackGenerator

__all__ = [
    "TargetAttackCase",
    "TargetAttackGenerator",
    "assert_blocked",
    "assert_allowed",
    "assert_sanitized",
]
