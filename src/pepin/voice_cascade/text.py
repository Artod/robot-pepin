"""The brain's text stream cut into sentences the voice can start on, and cleaned for speech."""

from __future__ import annotations

import re

_END = re.compile(r"[.!?…]+[\"»”')\]]*(?:\s+|$)|\n+")
_SOFT = re.compile(r"[,;:]\s|\s[—–-]\s")  # noqa: RUF001 -- the dashes are meant
_MARKUP = re.compile(r"[*_#`~>|]+")
_ASIDE = re.compile(r"\[[^\]]*\]|\([^)]*\)")  # [notes] and (stage directions) are not speech
_SPACES = re.compile(r"\s+")
_WORDS = re.compile(r"\w+")
_VOWELS = re.compile("[аеёиоуыэюяaeiouy]", re.IGNORECASE)  # noqa: RUF001


class SentenceSplitter:
    """Text deltas in, whole sentences out as soon as each one ends.

    A sentence ends at ``. ! ? …`` followed by a space (``3.5`` does not end one) or at a line
    break. A first sentence still open after ``soft_chars`` is cut at its last comma or dash, so
    a long opening clause does not hold the voice back.
    """

    def __init__(self, soft_chars: int = 90) -> None:
        """``soft_chars``: an open sentence this long is cut at a comma (0: never)."""
        self.soft_chars = soft_chars
        self._buffer = ""
        self._emitted = 0

    def feed(self, text: str) -> list[str]:
        """The sentences ``text`` completes."""
        self._buffer += text
        out: list[str] = []
        while True:
            match = _END.search(self._buffer)
            if match is None or (match.end() == len(self._buffer) and match.group().strip()):
                # "... 3." at the end may still become "3.5": wait for what follows the stop
                break
            sentence = self._buffer[: match.end()].strip()
            self._buffer = self._buffer[match.end() :]
            if sentence:
                out.append(sentence)
        if (
            self.soft_chars
            and not self._emitted
            and not out
            and len(self._buffer) > self.soft_chars
        ):
            cuts = list(_SOFT.finditer(self._buffer))
            if cuts and cuts[-1].start() > 20:
                cut = cuts[-1].start() + 1
                out.append(self._buffer[:cut].strip())
                self._buffer = self._buffer[cut:]
        self._emitted += len(out)
        return out

    def flush(self) -> str:
        """Whatever is left (the end of the text), "" when nothing."""
        rest = self._buffer.strip()
        self._buffer = ""
        if rest:
            self._emitted += 1
        return rest


def speakable(text: str) -> str:
    """``text`` without what a voice must not read out: markup (asterisks, hashes, backticks)
    and asides in brackets or parentheses (a model's notes to itself)."""
    return _SPACES.sub(" ", _MARKUP.sub(" ", _ASIDE.sub(" ", text))).strip()


def clip_key(text: str) -> str:
    """How a short line is looked up among the clips made at load: lower case, the words only,
    a question kept apart from a statement ("Да?" is not "Да.")."""
    words = " ".join(_WORDS.findall(text.lower().replace("\u0451", "\u0435")))  # yo read as ie
    return words + "?" if text.rstrip().endswith("?") else words


def syllables(text: str) -> int:
    """Syllables in Russian (one per vowel letter) and in the odd Latin word."""
    return len(_VOWELS.findall(text))
