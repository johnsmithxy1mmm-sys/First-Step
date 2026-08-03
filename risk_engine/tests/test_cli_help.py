"""Every command this repo ships can be asked what it does.

This file exists because one of them could not. `shadow icc --help` died with

    TypeError: %o format: an integer is required, not dict

for a reason nothing in the code review process would have caught: argparse
runs every help string through `%` expansion, so a literal percent sign is a
format spec. `"a 95% lower confidence bound"` parses as `% lo` -- space flag,
`l` length modifier, `o` octal conversion -- and argparse hands it a dict.

The defect is invisible until someone types `--help`, and no other test path
formats help text. It also lands on the worst possible person: someone who
does not know the command yet. And it is latent everywhere, because writing
"95%" in a help string is the natural thing to do in a codebase whose whole
subject is percentages -- `grep` finds that spelling in six other places, all
of them currently in `print` calls where it is harmless, any of which could be
moved into a `help=` next month.

So this walks the commands rather than asserting on the one that broke: it
discovers subcommands from the usage line, which is exactly the set a user can
type, and renders each one's help. It asserts nothing about the wording. The
assertion is that the command answers at all.
"""

from __future__ import annotations

import io
import re
from contextlib import redirect_stderr, redirect_stdout

import pytest

# Import the entry points, not module paths: a renamed module then fails here
# as a collection error rather than as a silently skipped test.
from risk_engine.market.collect_addresses import main as collect_main
from risk_engine.market.probe_isolated_funding import main as probe_main
from risk_engine.market.verify import main as verify_main
from risk_engine.service.app import main as service_main
from risk_engine.shadow.cli import main as shadow_main
from risk_engine.validation.cli import main as validation_main
from risk_engine.validation.power import main as power_main

ENTRY_POINTS = {
    "shadow": shadow_main,
    "validation": validation_main,
    "power": power_main,
    "verify": verify_main,
    "collect-addresses": collect_main,
    "probe-isolated-funding": probe_main,
    "service": service_main,
}


def _render_help(main, argv: list[str]) -> str:
    """Run `main` with a help flag and return what the user would see.

    `--help` exits, so `SystemExit` is the success path; the failure this
    guards against is a `TypeError` escaping from argparse's own formatter
    before the exit ever happens.
    """
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        with pytest.raises(SystemExit) as exc:
            main(argv)
    assert exc.value.code == 0, f"{argv} exited {exc.value.code}: {err.getvalue()}"
    return out.getvalue()


def _subcommands(usage: str) -> list[str]:
    """The subcommand names argparse advertises, read off its own usage line.

    Read from the rendered output rather than by introspecting the parser,
    because every entry point in this repo builds its parser inside `main`
    and there is no object to introspect. The upside is that this tests the
    set of words a user can actually type.
    """
    match = re.search(r"\{([\w,-]+)\}", usage)
    return match.group(1).split(",") if match else []


@pytest.mark.parametrize("name", sorted(ENTRY_POINTS))
def test_the_top_level_help_renders(name):
    assert _render_help(ENTRY_POINTS[name], ["--help"]).strip()


@pytest.mark.parametrize("name", sorted(ENTRY_POINTS))
def test_every_subcommand_help_renders(name):
    """The `icc` failure was here and not at the top level: the top-level help
    prints only each subparser's one-line `help=`, so a broken help string in
    an *argument* stays hidden until that subcommand is asked directly."""
    main = ENTRY_POINTS[name]
    subs = _subcommands(_render_help(main, ["--help"]))
    for sub in subs:
        assert _render_help(main, [sub, "--help"]).strip(), f"{name} {sub}"


def test_the_discovery_finds_the_subcommands_it_is_supposed_to():
    """A guard on the guard. `_subcommands` returning `[]` would make the test
    above pass vacuously for every command, which is precisely the failure
    mode that lets a regression through unnoticed."""
    subs = _subcommands(_render_help(shadow_main, ["--help"]))
    assert "icc" in subs, subs
    assert len(subs) >= 4, subs


def test_a_literal_percent_in_a_help_string_is_what_broke_it():
    """Pins the mechanism, so a future reader does not have to rediscover why
    `%%` appears in `cli.py`. Not a test of our code -- a test of the claim
    this file's docstring makes about argparse."""
    import argparse

    p = argparse.ArgumentParser(prog="t", add_help=False)
    p.add_argument("--x", help="a 95% lower bound")
    with pytest.raises(TypeError):
        p.format_help()

    ok = argparse.ArgumentParser(prog="t", add_help=False)
    ok.add_argument("--x", help="a 95%% lower bound")
    assert "95%" in ok.format_help()


def _documented_compose_runs() -> list[tuple[str, str]]:
    """Every `docker compose run` line the repo prints at an operator.

    Backslash continuations are folded first: the commands are wrapped for
    width, and a line-at-a-time scan would see the flags and the service name
    as separate commands.
    """
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[2]
    found: list[tuple[str, str]] = []
    for rel in ("deploy/README.md", "deploy/docker-compose.yml"):
        text = (root / rel).read_text(encoding="utf-8")
        # Continuations inside a YAML comment carry a `#` on the next line.
        joined = re.sub(r"\\\s*\n\s*#?\s*", " ", text)
        for line in joined.splitlines():
            if "docker compose run" in line:
                found.append((rel, line.strip().lstrip("# ")))
    return found


def test_the_documented_one_off_commands_can_actually_start():
    """The engine image's ENTRYPOINT is `python3 -m risk_engine.service`, so
    `docker compose run engine -m risk_engine.shadow progress` APPENDS to it
    and dies on an argparse usage message before reaching the module named.

    Both files shipped exactly that line. One of them carried a comment
    explaining how a *different* defect in the same command had been fixed --
    a missing `SHADOW_DSN` -- which is how a command can be corrected and stay
    unrunnable. Checked here rather than by eye because the failure is
    invisible in review and lands entirely on the operator.
    """
    runs = _documented_compose_runs()
    assert runs, "the parser found no commands; it stopped testing anything"
    for where, cmd in runs:
        if "-m risk_engine." in cmd:
            assert "--entrypoint" in cmd, f"{where}: {cmd}"


def test_the_entrypoint_this_guards_is_still_the_one_in_the_image():
    """The guard above is only meaningful while the image really does bake a
    module into its ENTRYPOINT. If that changes, this fails and the guard gets
    re-derived rather than silently protecting against nothing."""
    import pathlib

    dockerfile = (pathlib.Path(__file__).resolve().parents[2]
                  / "deploy/Dockerfile.engine").read_text(encoding="utf-8")
    assert 'ENTRYPOINT ["python3", "-m", "risk_engine.service"]' in dockerfile
