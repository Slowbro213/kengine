import pytest
from pydantic import ValidationError

from kengine.beliefs.status import TERMINAL, Status
from kengine.claims.extraction import (
    ExtractedClaim,
    ExtractedFormal,
    Extraction,
    ProposedAction,
    to_formal,
)
from kengine.claims.formal import FormalSyntaxError, JsonStatement
from kengine.claims.schema import Claim, ClaimType, Computation, Formal, Source, SourceKind

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


def test_formal_is_canonicalized_and_split():
    f = Formal(statements=("∀x (Dog(x) → Mammal(x))", "a & b"))
    assert f.statements == ("dog(x) -> mammal(x)", "a", "b")
    assert f.mode == "holds"


def test_formal_rejects_bad_syntax():
    with pytest.raises(ValidationError, match="unbalanced"):
        Formal(statements=("dog(rex",))


def test_claim_round_trips_formal():
    c = Claim(
        id="b1",
        type=ClaimType.GIVEN,
        text="Rex is a dog",
        source=Source(kind=SourceKind.TASK),
        step=0,
        formal=Formal(statements=("dog(rex)",)),
    )
    assert Claim.model_validate_json(c.model_dump_json()) == c


def test_extracted_formal_accepts_both_serializations():
    e = Extraction.model_validate(
        {
            "claims": [
                {
                    "ref": "c1",
                    "type": "inference",
                    "text": "Rex is a mammal",
                    "premises": ["b1", "b2"],
                    "formal": {"statements": ["mammal(rex)", {"if": [], "then": "dog(rex)"}]},
                }
            ],
            "action": {"kind": "continue"},
        }
    )
    f = e.claims[0].formal
    assert f is not None and isinstance(f.statements[1], JsonStatement)
    assert to_formal(f).statements == ("mammal(rex)", "dog(rex)")


def test_extracted_formal_is_not_validated_until_converted():
    c = ExtractedClaim.model_validate(
        {"ref": "c1", "type": "fact", "text": "t", "formal": {"statements": ["dog(rex"]}}
    )
    assert c.formal is not None
    with pytest.raises(FormalSyntaxError, match="unbalanced"):
        to_formal(c.formal)


def test_to_formal_keeps_mode():
    f = to_formal(ExtractedFormal(mode="undetermined", statements=["barks(rex)"]))
    assert f.mode == "undetermined"
