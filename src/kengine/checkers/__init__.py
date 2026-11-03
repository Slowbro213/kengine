"""Claim checkers (spec §5.5)."""

from __future__ import annotations

from kengine.checkers.arithmetic import ArithmeticChecker
from kengine.checkers.base import Checker
from kengine.checkers.logic import LogicChecker
from kengine.checkers.restatement import RestatementChecker
from kengine.checkers.structure import StructureChecker


def default_checkers() -> list[Checker]:
    """The M2 checkers, in the order they run: structure first, so a malformed claim is
    caught before anything evaluates it."""
    return [StructureChecker(), ArithmeticChecker(), RestatementChecker(), LogicChecker()]
