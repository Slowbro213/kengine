"""Belief statuses (spec §5.4)."""

from __future__ import annotations

from enum import StrEnum


class Status(StrEnum):
    UNCHECKED = "unchecked"
    VERIFIED = "verified"
    SUPPORTED = "supported"
    CRITIC_ACCEPTED = "critic_accepted"
    ASSUMED = "assumed"
    CONTRADICTED = "contradicted"
    RETRACTED = "retracted"
    INVALIDATED = "invalidated"


# No status changes out of these in v1.
TERMINAL: frozenset[Status] = frozenset({Status.CONTRADICTED, Status.RETRACTED, Status.INVALIDATED})
