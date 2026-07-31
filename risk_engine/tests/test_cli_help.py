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
