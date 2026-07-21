"""Minimal ANSI styling for the run log.

Colour is emitted only when the destination is a real terminal, so HPC batch
logs and piped output stay free of escape codes. ``NO_COLOR`` (see no-color.org)
and ``TERM=dumb`` disable it as well.

Style is applied *after* any padding, so field alignment is unaffected by the
invisible escape sequences.

Deliberately no faint/dim style: it renders close to unreadable on terminals
whose palette already has low contrast. Ordinary text stays the terminal's
default colour, and emphasis is carried by bold and hue alone.
"""

import os
import sys

_CODES = {
    "bold": "1",
    "red": "31",
    "green": "32",
    "yellow": "33",
    "blue": "34",
    "magenta": "35",
    "cyan": "36",
}
_RESET = "\033[0m"


def supports_colour(stream=None):
    stream = stream or sys.stdout
    if os.environ.get("NO_COLOR") is not None:
        return False
    if os.environ.get("TERM", "") == "dumb":
        return False
    try:
        return bool(stream.isatty())
    except (AttributeError, ValueError):
        return False


class Style:
    """Callable styles that become no-ops when colour is unavailable."""

    def __init__(self, enabled=None, stream=None):
        self.enabled = supports_colour(stream) if enabled is None else enabled

    def _wrap(self, text, *names):
        if not self.enabled or not names:
            return text
        codes = ";".join(_CODES[n] for n in names)
        return f"\033[{codes}m{text}{_RESET}"

    def bold(self, text):
        return self._wrap(text, "bold")

    def head(self, text):
        return self._wrap(text, "bold", "cyan")

    def value(self, text):
        return self._wrap(text, "bold")

    def ok(self, text):
        return self._wrap(text, "green")

    def warn(self, text):
        return self._wrap(text, "yellow")

    def bad(self, text):
        return self._wrap(text, "bold", "red")

    def path(self, text):
        return self._wrap(text, "bold", "blue")

    def bar(self, text):
        return self._wrap(text, "cyan")
