from src.prompt_templates import render_staff_question


def test_context_replacement() -> None:
    rendered = render_staff_question(context="Manager approval is required.", question="How do I request leave?")
    assert "Manager approval is required." in rendered


def test_question_replacement() -> None:
    rendered = render_staff_question(context="Staff handbook", question="Where is the leave policy?")
    assert "Where is the leave policy?" in rendered


def test_json_structure_is_rendered() -> None:
    rendered = render_staff_question(context="Context", question="Question")
    assert '{"answer": "...", "source": "..."}' in rendered


def test_no_unresolved_placeholders() -> None:
    rendered = render_staff_question(context="Context", question="Question")
    assert "{context}" not in rendered
    assert "{question}" not in rendered
    assert "{{" not in rendered
    assert "}}" not in rendered


def test_different_inputs_produce_different_renders() -> None:
    first = render_staff_question(context="First context", question="First question")
    second = render_staff_question(context="Second context", question="Second question")
    assert first != second
