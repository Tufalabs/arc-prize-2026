"""``ullm.reasoning_register``: the blanked/control register classifier.

Covers the contract a caller relies on -- the two registers land on the
right sides, a list scores as one text, empty input is uninformative --
and the property the fit is built for: the vocabulary carries no game
words, so a text about something other than a grid puzzle is still read
by its register and not by which game it plays.

The opening model is covered for the one thing that makes it a separate
model rather than a recalibration: a control OPENING, which is short and
starts with "Let's", is read correctly by it and misread by the corpus
model, whose "let's" weight points the other way.
"""

from ullm import reasoning_register

# Register samples about a subject none of the fitted runs contain, so a
# score here can only come from function words and discourse markers.
BLANKED_DEBUGGING = """
We need find why server returns 500. Need check logs first. Maybe config not
loaded. Let's grep for error. Could be missing env var. We have access to
container, so let's inspect. Not sure yet, but likely database connection
refused because port wrong. Let's test with curl.
"""

CONTROL_DEBUGGING = """
Let me start by looking at what the server is actually returning. Hmm, the
500 suggests an unhandled exception somewhere in the request path. Wait, let
me think about this more carefully. What if the configuration file is not
being loaded at all? I should check the logs first, and then I can look at
the environment variables that the container is reading.
"""


def test_registers_land_on_their_sides() -> None:
    """A blanked-register text scores near 1, a control-register text near 0."""
    assert reasoning_register.blanked_score(BLANKED_DEBUGGING) > 0.7
    assert reasoning_register.blanked_score(CONTROL_DEBUGGING) < 0.3


def test_a_list_is_scored_as_one_text() -> None:
    """The register is a property of the corpus, so parts and whole agree."""
    parts = [line for line in CONTROL_DEBUGGING.strip().split(". ") if line]
    assert reasoning_register.blanked_score(parts) == reasoning_register.blanked_score(". ".join(parts))


def test_empty_input_is_uninformative() -> None:
    """No tokens, no evidence: exactly the midpoint, not an error."""
    assert reasoning_register.blanked_score("") == 0.5
    assert reasoning_register.blanked_score([]) == 0.5
    assert reasoning_register.blanked_score("12345 !!! ---") == 0.5


def test_vocabulary_carries_no_subject() -> None:
    """No game word carries weight: the fit cannot have learnt the games."""
    model = reasoning_register.shipped()
    subject_words = {
        "row",
        "col",
        "column",
        "rows",
        "columns",
        "grid",
        "board",
        "level",
        "game",
        "pixel",
        "blue",
        "cell",
    }
    assert not subject_words & set(model.weights)
    assert len(model.weights) <= 40, "the list is meant to stay short"


# A short control opening: the register is carried by the articles and
# prepositions ("the ... of the"), not by the verb it happens to start with.
CONTROL_OPENING = "Let's start by checking the current state of the deployment."
BLANKED_OPENING = "We need check deployment. Need inspect config. Let's run status."


def test_opening_model_reads_short_openings() -> None:
    """Both openings land on their own side, at a length the corpus model cannot use."""
    assert reasoning_register.opening_blanked_score(BLANKED_OPENING) > 0.7
    assert reasoning_register.opening_blanked_score(CONTROL_OPENING) < 0.3


def test_the_two_models_disagree_where_they_must() -> None:
    """The corpus model misreads a control opening; that is why this one exists.

    Not a defect of either: "let's" is a blanked marker over a whole game and
    a control marker at its opening, so one fit cannot serve both slices. The
    assertion pins the disagreement so a refit that erased it would be seen.
    """
    assert reasoning_register.blanked_score(CONTROL_OPENING) > 0.5
    assert reasoning_register.opening_blanked_score(CONTROL_OPENING) < 0.5
