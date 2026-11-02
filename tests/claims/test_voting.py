from kengine.claims.extraction import ExtractableType, ExtractedClaim, Extraction, ProposedAction
from kengine.claims.voting import VoteResult, default_k, vote

CONTINUE = ProposedAction(kind="continue")


def c(ref: str, type_: ExtractableType, text: str, premises: list[str] | None = None):
    return ExtractedClaim(ref=ref, type=type_, text=text, premises=premises or [])


def sample(*claims: ExtractedClaim, action: ProposedAction = CONTINUE) -> Extraction:
    return Extraction(claims=list(claims), action=action)


def test_single_sample_k1_keeps_everything():
    s = sample(c("c1", "fact", "a is 1"), c("c2", "inference", "b is 2", ["c1"]))
    result = vote([s], k=1)
    assert result == VoteResult(claims=s.claims, action=CONTINUE, dropped=[], n_valid=1)


def test_claim_in_one_of_three_samples_is_dropped_and_links_removed():
    s1 = sample(
        c("c1", "fact", "Tom has 3 apples"), c("c2", "inference", "Tom has 6 apples", ["c1"])
    )
    s2 = sample(c("c1", "inference", "Tom has 6 apples."))
    s3 = sample(c("c1", "inference", "Tom has 6 apples."))
    result = vote([s1, s2, s3], k=2)
    assert [(x.ref, x.premises) for x in result.claims] == [("c2", [])]
    assert result.dropped == ["c1"] and result.n_valid == 3


def test_b_refs_survive_premise_filtering():
    s = sample(c("c1", "inference", "x follows", ["b3", "c9"]))
    assert vote([s, s], k=2).claims[0].premises == ["b3", "c9"]


def test_type_mismatch_does_not_match():
    s1 = sample(c("c1", "fact", "the sky is blue"))
    s2 = sample(c("c1", "assumption", "the sky is blue"))
    result = vote([s1, s2], k=2)
    assert result.claims == [] and result.dropped == ["c1"]


def test_anchor_is_most_supported_sample_not_first():
    s1 = sample(
        c("c1", "fact", "zebras are purple"),
        c("c2", "fact", "moon is cheese"),
        c("c3", "fact", "rain falls up"),
    )
    s2 = sample(c("c1", "fact", "the train leaves at 9"), c("c2", "fact", "the trip takes 3 hours"))
    s3 = sample(
        c("c5", "fact", "The train leaves at 9."), c("c6", "fact", "the trip takes 3 hours")
    )
    result = vote([s1, s2, s3], k=2)
    assert [x.ref for x in result.claims] == ["c1", "c2"]
    assert [x.text for x in result.claims] == ["the train leaves at 9", "the trip takes 3 hours"]
    assert result.dropped == []


def test_action_majority_and_precondition_mapping():
    claims = [c("c1", "fact", "a is 1"), c("c2", "fact", "b is 2")]
    s1 = sample(*claims, c("c3", "fact", "only here"), action=CONTINUE)
    s2 = sample(
        c("c1", "fact", "b is 2"),
        c("c2", "fact", "a is 1"),
        c("c3", "fact", "something unrelated entirely"),
        action=ProposedAction(kind="final_answer", answer="3", preconditions=["c1", "c3", "b4"]),
    )
    s3 = sample(*claims, action=ProposedAction(kind="final_answer", answer="4"))
    result = vote([s1, s2, s3], k=2)
    assert result.dropped == ["c3"]
    assert result.action is not None
    assert (result.action.kind, result.action.answer) == ("final_answer", "3")
    # s2's c1 is the anchor's c2; s2's c3 matches no kept claim.
    assert result.action.preconditions == ["c2", "b4"]


def test_anchor_action_preconditions_filtered_by_dropped():
    s1 = sample(
        c("c1", "fact", "a is 1"),
        c("c2", "fact", "only here"),
        action=ProposedAction(kind="final_answer", answer="1", preconditions=["c1", "c2", "b1"]),
    )
    s2 = sample(c("c1", "fact", "a is 1"), action=ProposedAction(kind="final_answer", answer="1"))
    result = vote([s1, s2], k=2)
    assert result.action is not None and result.action.preconditions == ["c1", "b1"]


def test_action_tie_goes_to_anchor():
    s1 = sample(c("c1", "fact", "a"), action=ProposedAction(kind="final_answer", answer="1"))
    s2 = sample(action=CONTINUE)
    result = vote([s1, s2], k=1)
    assert result.action is not None and result.action.kind == "final_answer"


def test_empty_samples():
    assert vote([], 1) == VoteResult(claims=[], action=None, dropped=[], n_valid=0)


def test_default_k():
    assert [default_k(n) for n in (1, 3, 5)] == [1, 2, 3]
