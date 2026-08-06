"""Every text file this engine reads or writes names its encoding.

Reported from an operator's console on Windows, where every section reference
in a recorded finding grew a stray capital letter in front of it:

    mark tracks mid within <U+0412>§1.4's threshold ...

`§` is U+00A7, which in UTF-8 is the two bytes C2 A7. Read back under cp1251
those decode as U+0412 (Cyrillic Ve) followed by U+00A7. The file was UTF-8,
the reader was not, and nothing in between said so. Under a stricter default
it does not even garble: an ASCII locale raises UnicodeDecodeError and the
command dies outright, which is how this file's own test demonstrates it.

Python's default is the trap: `open()`, `read_text()` and `write_text()` use
`locale.getencoding()` unless told otherwise, so the same code is correct on
CI (UTF-8), correct on the maintainer's machine (UTF-8), and wrong on the
operator's. It is invisible to every test that reads a file it just wrote,
because the same wrong encoding round-trips cleanly.

Two of the sites this found were more than cosmetic:

  - `shadow/providers.py` reads the address list, whose `frame` field is prose
    an operator is *required* to hand-write. A frame typed on Windows and read
    in the Linux container is a different string — and that string is written
    into the journal as the permanent sampling-frame provenance of every
    calibration score computed from it (OPEN-QUESTIONS B4).
  - `shadow/cli.py` writes that file's template, so the round trip could be
    broken from either end.

So this is an AST sweep rather than a round-trip test. A round-trip test
cannot catch it — writing and reading with the same wrong encoding succeeds —
and the defect is a *property of the source*: every text operation must state
what bytes it means. Checking the property directly is the only way to keep
it, and it covers files nobody has written a test for yet.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

#: Only shipped code. Tests read and write their own tmp files and never
#: exchange them with another machine, so the property does not apply and
#: enforcing it there would be noise that gets the sweep switched off.
#: `scripts/` is here because its files are operator tools run on exactly the
#: machines where the locale differs from the container's — the audit found
#: it excluded (F-13), which is the one directory the sweep exists for.
ROOTS = ("risk_engine", "qa", "scripts")

TEXT_IO = ("read_text", "write_text", "open")


def _call_name(node: ast.Call) -> str:
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    if isinstance(node.func, ast.Name):
        return node.func.id
    return ""


def _unsafe_calls(path: pathlib.Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    out: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or _call_name(node) not in TEXT_IO:
            continue
        if "encoding" in {k.arg for k in node.keywords}:
            continue
        if _call_name(node) == "open":
            # Binary mode has no encoding and must not be asked for one.
            mode = ""
            if len(node.args) > 1 and isinstance(node.args[1], ast.Constant):
                mode = str(node.args[1].value)
            if "b" in mode:
                continue
        out.append(f"{path}:{node.lineno} {_call_name(node)}()")
    return out


def _shipped_sources() -> list[pathlib.Path]:
    repo = pathlib.Path(__file__).resolve().parents[2]
    files: list[pathlib.Path] = []
    for root in ROOTS:
        files += [p for p in sorted((repo / root).rglob("*.py"))
                  if "tests" not in p.parts]
    return files


def test_the_sweep_actually_looks_at_something():
    """A guard on the guard: an empty file list makes the test below pass
    vacuously, which is precisely how a sweep stops protecting anything."""
    files = _shipped_sources()
    assert len(files) > 20, files
    assert any(p.name == "providers.py" for p in files)


@pytest.mark.parametrize("path", _shipped_sources(), ids=lambda p: p.name)
def test_text_io_names_its_encoding(path):
    unsafe = _unsafe_calls(path)
    assert not unsafe, (
        "text file operations without an explicit encoding, which use the "
        "platform locale and therefore differ between the operator's machine "
        "and the container:\n  " + "\n  ".join(unsafe) +
        "\nPass encoding='utf-8' (or open in binary mode)."
    )


def test_the_detector_recognises_the_shape_it_is_looking_for(tmp_path):
    """Pins the detector against a known-bad and known-good file, so a
    refactor cannot quietly turn it into a function that always returns []."""
    bad = tmp_path / "bad.py"
    bad.write_text(
        "import pathlib\n"
        "pathlib.Path('x').read_text()\n"
        "open('y', 'w')\n",
        encoding="utf-8",
    )
    assert len(_unsafe_calls(bad)) == 2

    good = tmp_path / "good.py"
    good.write_text(
        "import pathlib\n"
        "pathlib.Path('x').read_text(encoding='utf-8')\n"
        "open('y', 'w', encoding='utf-8')\n"
        "open('z', 'rb')\n",  # binary needs no encoding
        encoding="utf-8",
    )
    assert _unsafe_calls(good) == []


def test_the_shipped_findings_file_survives_a_utf8_read():
    """The file that produced the bug report. It carries literal `§`
    characters, and reading it as anything but UTF-8 mangles them."""
    from risk_engine.market.findings import load_findings

    findings = load_findings()
    joined = " ".join(f.detail for f in findings.values())
    assert "§" in joined, "expected section references in the recorded details"
    # The exact mojibake, asserted absent rather than inferred from the read
    # merely succeeding. Written as escapes because the characters themselves
    # are homoglyphs of ASCII letters -- the property that makes this class of
    # bug hard to see in the first place, and that ruff's RUF001 flags.
    # Derived from the bytes rather than typed, which is both unambiguous and
    # a statement of the mechanism: these ARE what UTF-8's C2 A7 becomes.
    utf8_section_bytes = "§".encode()
    assert utf8_section_bytes == b"\xc2\xa7"
    for legacy in ("cp1251", "cp1252"):
        assert utf8_section_bytes.decode(legacy) not in joined
