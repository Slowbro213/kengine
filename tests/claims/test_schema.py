import pytest
from pydantic import ValidationError

from kengine.beliefs.status import TERMINAL, Status
from kengine.claims.extraction import ExtractedClaim, Extraction, ProposedAction
from kengine.claims.schema import Claim, ClaimType, Computation, Source, SourceKind

SRC = Source(kind=SourceKind.MODEL)
COMP = Computation(expression="2 + 2", claimed_result="4")


def test_terminal_statuses():
    assert TERMINAL == {Status.CONTRADICTED, Status.RETRACTED, Status.INVALIDATED}


def test_computation_claim_requires_computation():
    with pytest.raises(ValidationError):
        Claim(id="b1", type=ClaimType.COMPUTATION, text="x", source=SRC, step=1)
    claim = Claim(
        id="b1", type=ClaimType.COMPUTATION, text="x", computation=COMP, source=SRC, step=1
    )
    assert claim.computation == COMP


def test_non_computation_claim_rejects_computation():
    with pytest.raises(ValidationError):
        Claim(id="b1", type=ClaimType.FACT, text="x", computation=COMP, source=SRC, step=1)


def test_confidence_bounds():
    with pytest.raises(ValidationError):
        Claim(id="b1", type=ClaimType.FACT, text="x", confidence=1.2, source=SRC, step=1)
    claim = Claim(id="b1", type=ClaimType.FACT, text="x", confidence=0.5, source=SRC, step=1)
    assert claim.confidence == 0.5


def test_claim_id_pattern():
    assert Claim(id="b12", type=ClaimType.FACT, text="x", source=SRC, step=1).id == "b12"
    with pytest.raises(ValidationError):
        Claim(id="c1", type=ClaimType.FACT, text="x", source=SRC, step=1)


def test_claim_is_frozen():
    claim = Claim(id="b1", type=ClaimType.FACT, text="x", source=SRC, step=1)
    with pytest.raises(ValidationError):
        claim.text = "y"  # type: ignore[misc]


def test_extracted_ref_pattern_and_limits():
    assert ExtractedClaim(ref="c1", type="fact", text="x").ref == "c1"
    with pytest.raises(ValidationError):
        ExtractedClaim(ref="x", type="fact", text="x")
    with pytest.raises(ValidationError):
        ExtractedClaim(ref="c1", type="fact", text="")
    with pytest.raises(ValidationError):
        ExtractedClaim(ref="c1", type="fact", text="a" * 401)
    assert ExtractedClaim(ref="c1", type="inference", text="x", premises=["c2", "b7"]).premises
    with pytest.raises(ValidationError):
        ExtractedClaim(ref="c1", type="inference", text="x", premises=["zz"])
    claims = [ExtractedClaim(ref=f"c{i}", type="fact", text="x") for i in range(13)]
    with pytest.raises(ValidationError):
        Extraction(claims=claims, action=ProposedAction(kind="continue"))


def test_extracted_computation_iff_type():
    with pytest.raises(ValidationError):
        ExtractedClaim(ref="c1", type="computation", text="x")
    with pytest.raises(ValidationError):
        ExtractedClaim(ref="c1", type="fact", text="x", computation=COMP)


def test_extracted_type_excludes_task_only_types():
    with pytest.raises(ValidationError):
        ExtractedClaim(ref="c1", type="given", text="x")  # type: ignore[arg-type]


def test_extraction_json_schema_is_self_contained():
    s = Extraction.model_json_schema()
    assert s["properties"]["claims"]["maxItems"] == 12
    assert "computation" in s["$defs"]["ExtractedClaim"]["properties"]
