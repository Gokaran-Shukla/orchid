"""Tutorials and self-assessment quizzes (v0.4).

Every tutorial notebook in ``docs/tutorials/`` ends with a five-question quiz.
Each question has one correct answer, and every answer comes with an
explanation that ties it back to the equations of that section.

    >>> from orchid.tutorials import quiz, list_tutorials
    >>> list_tutorials()                 # sections and notebook names
    >>> quiz("03")                       # show the questions
    >>> quiz("03", answers="bcada")      # grade, with explanations

From a terminal: ``orchid quiz`` (choose a section) or ``orchid quiz 03``.
"""
import json
import os
import textwrap

_GS_NOTICE = "© 2026 Gokaran Shukla · github.com/Gokaran-Shukla/orchid · MIT License"  # gs-watermark


_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "quizzes.json")


def _load():
    with open(_DATA, encoding="utf-8") as f:
        return json.load(f)


def list_tutorials():
    """Return [(section id, title, notebook file)]."""
    return [(k, v["title"], v["notebook"]) for k, v in _load().items()]


def questions(section):
    """Raw question list for one section (each: q, options, answer, why)."""
    return _load()[_key(section)]["questions"]


def _key(section):
    s = str(section).zfill(2)
    data = _load()
    if s not in data:
        raise KeyError(f"unknown section {section!r}; choose from {', '.join(data)}")
    return s


def _parse(answers, n):
    if isinstance(answers, str):
        a = [c.lower() for c in answers if c.isalpha()]
    else:
        a = [str(c).strip().lower() for c in answers]
    if len(a) != n:
        raise ValueError(f"expected {n} answers, got {len(a)}")
    return a


def _render_questions(sec):
    d = _load()[sec]
    out = [f"Quiz {sec}: {d['title']}", ""]
    for i, q in enumerate(d["questions"], 1):
        out.append(f"Q{i}. {q['q']}")
        for k, o in zip("abcd", q["options"]):
            out.append(f"    ({k}) {o}")
        out.append("")
    out.append(f'Answer with quiz("{sec}", answers="....."), one letter per question.')
    return "\n".join(out)


def grade(section, answers):
    """Return (score, n, feedback lines)."""
    sec = _key(section)
    qs = _load()[sec]["questions"]
    a = _parse(answers, len(qs))
    lines, score = [], 0
    for i, (q, given) in enumerate(zip(qs, a), 1):
        ok = given == q["answer"]
        score += ok
        mark = "correct" if ok else f"not quite, the answer is ({q['answer']})"
        lines.append(f"Q{i} ({given}): {mark}")
        lines.append(textwrap.fill(q["why"], 92, initial_indent="    ", subsequent_indent="    "))
    return score, len(qs), lines


def quiz(section, answers=None):
    """Show a quiz, or grade it when ``answers`` is given ("bcada" or a list)."""
    sec = _key(section)
    if answers is None:
        print(_render_questions(sec))
        print(_GS_NOTICE)
        return None
    score, n, lines = grade(sec, answers)
    print("\n".join(lines))
    print(f"\nScore: {score}/{n}" + ("  - well done." if score == n else ""))
    return score


def interactive(section=None, input_fn=input, print_fn=print):
    """Terminal quiz used by ``orchid quiz``."""
    data = _load()
    if section is None:
        for k, v in data.items():
            print_fn(f"  {k}  {v['title']}")
        section = input_fn("Section: ").strip()
    sec = _key(section)
    qs = data[sec]["questions"]
    print_fn(f"\nQuiz {sec}: {data[sec]['title']}\n")
    score = 0
    for i, q in enumerate(qs, 1):
        print_fn(f"Q{i}. {q['q']}")
        for k, o in zip("abcd", q["options"]):
            print_fn(f"    ({k}) {o}")
        a = ""
        while a not in ("a", "b", "c", "d"):
            a = input_fn("your answer [a-d]: ").strip().lower()[:1]
        ok = a == q["answer"]
        score += ok
        print_fn("  correct." if ok else f"  not quite, the answer is ({q['answer']}).")
        print_fn(textwrap.fill(q["why"], 92, initial_indent="  ", subsequent_indent="  ") + "\n")
    print_fn(f"Score: {score}/{len(qs)}")
    return score
