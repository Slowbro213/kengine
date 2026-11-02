from pathlib import Path

from kengine.audit.events import BeliefAddedEvent, FeedbackEvent, ModelCallEvent
from kengine.audit.session import Session
from kengine.backend.types import Completion, Message
from kengine.claims.schema import Claim, ClaimType, Source, SourceKind


def test_session_create_and_event_roundtrip(tmp_path: Path):
    sess = Session.create(tmp_path)
    assert (sess.dir / "beliefs.sqlite").exists()
    assert sess.dir.parent == tmp_path
    sess.log.append(FeedbackEvent(step=1, items=["x"]))
    reopened = Session.open(sess.dir)
    event = reopened.log.read()[0]
    assert isinstance(event, FeedbackEvent) and event.items == ["x"]


def test_event_union_roundtrips_variants(tmp_path: Path):
    sess = Session.create(tmp_path)
    claim = Claim(
        id="b1", type=ClaimType.FACT, text="x", source=Source(kind=SourceKind.MODEL), step=1
    )
    sess.log.append(BeliefAddedEvent(step=1, claim=claim))
    sess.log.append(
        ModelCallEvent(
            step=1,
            role="think",
            messages=[Message(role="user", content="hi")],
            completion=Completion(content="yo"),
            wall_s=0.5,
        )
    )
    events = sess.log.read()
    assert [e.kind for e in events] == ["belief_added", "model_call"]
    assert isinstance(events[0], BeliefAddedEvent) and events[0].claim == claim
    assert events[0].ts.tzinfo is not None
