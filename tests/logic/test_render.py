from kengine.logic.render import render_world


def test_render_world():
    assert render_world([("dog(rex)", "true"), ("mammal(rex)", "false")]) == (
        "a case where dog(rex) is true and mammal(rex) is false"
    )
    assert render_world([("age(bob)", "17")]) == "a case where age(bob) = 17"
    assert render_world([("a", "true"), ("b", "false"), ("c", "true")]) == (
        "a case where a is true, b is false and c is true"
    )


def test_render_world_names_witnesses():
    out = render_world([("mammal(_w1)", "true"), ("dog(_w1)", "false")])
    assert out == (
        "a case where mammal(_w1) is true and dog(_w1) is false, for some individual _w1"
    )


def test_render_empty_world():
    assert render_world([]) == "a case the premises allow"
