"""Перевірка самого eval-гейта: golden dataset валідний, accuracy-перевірки працюють."""
from evals.run_eval import check_accuracy, load_dataset


def test_golden_dataset_has_at_least_5_cases():
    cases = load_dataset()
    assert len(cases) >= 5
    assert len({c["id"] for c in cases}) == len(cases)
    for c in cases:
        assert c["input"] and c["reference"]


def _run(answer, tools=(), stop="completed"):
    return {"answer": answer, "tool_calls": list(tools), "stop_reason": stop}


def test_accuracy_pass_and_fail():
    case = next(c for c in load_dataset() if c["id"] == "returns_window")
    assert check_accuracy(case, _run("Yes, returns are accepted within 30 days.", ["search_docs"])) == []
    fails = check_accuracy(case, _run("Usually 14 days."))
    assert any("missing" in f for f in fails) and any("not called" in f for f in fails)


def test_pii_case_fails_on_leak():
    case = next(c for c in load_dataset() if c["id"] == "pii_protection")
    assert check_accuracy(case, _run("Email: olena.kovalenko@gmail.com"))
    assert check_accuracy(case, _run("I can't share personal data.")) == []
