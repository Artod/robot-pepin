"""The fillers: what the robot says by itself into dead air, one engine for every kind of it.

Each context (:data:`pepin.voice_cascade.config.FILLER_CONTEXTS`) is a moment the loop finds:

- ``tool``: a slow tool called before the reply's first word: one of the tool's own lines, else
  of the common ones (:meth:`pepin.voice_cascade.config.FillerBook.for_tool`);
- ``thinking``: the brain's first token is not in ``filler_thinking_s`` after the turn closed
  and nothing has been said (:func:`thinking_due`): a short sound that fits any answer, the
  hums among them, since nobody knows yet what the answer is; wanted more the longer the
  question (:func:`thinking_chance`: none under the book's ``thinking_min_words``, a short
  question is answered fast anyway), and that chance multiplies the context's own chance to
  speak (1 - its silent chance) in one draw;
- ``again``: the reply still silent ``filler_again_s`` after its last filler (:func:`again_due`);
- ``backchannel``: "mm" in the person's pause mid-utterance (``flags.backchannel``).

:class:`FillerEngine` keeps the rules. The language of the moment is the script of the person's
last words (:func:`script_language`) or ``conversation.language`` alone; a context without lines
in it says one of the neutral hums, but for the backchannel, which then stays silent (a hum under
the person's words reads as doubt). ``tool``, ``thinking`` and ``again`` speak at most once a
reply; each context has a cooldown and a chance to let its moment pass in silence (a game's "no
bark"), and a moment passed in silence counts as one spoken; no line comes again while it is
among the last :data:`RECENT` picked, as far as the lines allow; only a line whose clip is
made is picked (synthesised on the spot it would hold the voice ahead of the reply), and none
whose clip plays longer than its context's ``filler_max_s`` (the thinking filler is a short
sound: a whole phrase may still be playing when the reply is ready). A thinking filler still
playing when the reply's first audio comes is faded out over ``filler_fade_ms`` and the reply
follows at once (:mod:`pepin.voice_cascade.player`); the other fillers stay whole and the reply
queues behind them.
"""

from __future__ import annotations

import random
import re
from collections import deque
from collections.abc import Callable, Sequence
from itertools import pairwise
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from pepin.voice_cascade.config import CascadeConfig, FillerBook

Context = Literal["tool", "thinking", "again", "backchannel"]
RECENT = 3  # a line is not picked again while among this many last picks (the lines allowing)
ONCE_PER_REPLY = frozenset(("tool", "thinking", "again"))
UNKNOWN = "und"  # the language of words in another script than Cyrillic or Latin
KEPT_REPLIES = 4  # the replies whose spent contexts are remembered (the latest ones)
EPSILON_S = 1e-9  # a moment's time compared on the clock's floats
_WORDS = re.compile(r"[^\W\d_]+")  # runs of letters


class FillerPicker:
    """Picks lines at random (an unseeded generator: each run differs), avoiding the last
    :data:`RECENT` it picked as far as the lines allow, and never the last one while another
    is there."""

    def __init__(self, rng: random.Random | None = None, recent: int = RECENT) -> None:
        """``rng``: seeded in a test; a fresh one by default."""
        self.rng = rng if rng is not None else random.Random()
        self.recent: deque[str] = deque(maxlen=max(1, recent))

    @property
    def last(self) -> str | None:
        """The line picked last (None: none yet)."""
        return self.recent[-1] if self.recent else None

    def pick(self, lines: Sequence[str], spare: Sequence[str] = ()) -> str | None:
        """One of ``lines``, the latest picks avoided; from ``spare`` when ``lines`` holds the
        last one alone; the last one again when nothing else is there; None for no lines."""
        if not lines:
            return None
        choices = self._fresh(lines) or self._fresh(spare) or list(lines)
        line = self.rng.choice(choices)
        self.recent.append(line)
        return line

    def _fresh(self, lines: Sequence[str]) -> list[str]:
        """``lines`` less as many of the latest picks as leave one; none when the last pick is
        all there is."""
        recent = list(self.recent)
        for k in range(len(recent), 0, -1):
            fresh = [line for line in lines if line not in recent[-k:]]
            if fresh:
                return fresh
        return [] if recent else list(lines)


class FillerEngine(FillerPicker):
    """The fillers' rules (the module's notes): the language of the moment, the lines of a
    context in it, once a reply, cooldowns, silence by chance, no repeats, clips only."""

    def __init__(self, rng: random.Random | None = None, language: str = "ru") -> None:
        """``language``: the conversation's at its start."""
        super().__init__(rng)
        self.language = language
        self.last_s: dict[str, float] = {}  # each context's last moment, spoken or silent
        self.spent: dict[int, set[str]] = {}  # a reply's id -> the contexts it has had

    def open(self, language: str) -> None:
        """A conversation opens: the fillers speak ``language`` until words say otherwise."""
        self.language = language

    def hear(self, text: str) -> str:
        """The person's words: the language their letters are written in becomes the
        fillers' (no letters: no change); the language now."""
        language = script_language(text)
        if language is not None:
            self.language = language
        return self.language

    def lines(
        self, book: FillerBook, context: Context, tool: str | None = None
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """The lines a filler of ``context`` is picked from in the language now, and the spare
        ones (a tool's common lines, when its own hold only the last pick)."""
        lang = self.language
        if context == "tool":
            own = book.for_tool(tool or "", lang)
            return (own, book.common.get(lang, ())) if own else (book.neutral, ())
        if context == "thinking":
            return (*book.thinking.get(lang, ()), *book.neutral), ()
        if context == "again":
            return book.again.get(lang) or book.neutral, ()
        return book.backchannel.get(lang, ()), ()

    def choose(
        self,
        book: FillerBook,
        context: Context,
        now: float,
        *,
        turn: int | None = None,
        tool: str | None = None,
        silent_p: float = 0.0,
        cooldown_s: float = 0.0,
        ready: Callable[[str], bool] | None = None,
        max_s: float | None = None,
        seconds: Callable[[str], float | None] | None = None,
    ) -> str | None:
        """The line for a moment of ``context`` (of reply ``turn``, for ``tool``), or None:
        the reply has had it already (``tool``, ``thinking``, ``again``), it comes within
        ``cooldown_s`` of the context's last one, it passes in silence (``silent_p``), or no
        line has a clip yet (``ready``) that plays at most ``max_s`` (by ``seconds``, the
        clip's length): none then, and the moment is not spent."""
        if turn is not None and context in ONCE_PER_REPLY and context in self.spent.get(turn, ()):
            return None
        last = self.last_s.get(context)
        if last is not None and now - last < cooldown_s:
            return None
        lines, spare = self.lines(book, context, tool)
        if ready is not None:
            lines, spare = tuple(filter(ready, lines)), tuple(filter(ready, spare))
        if max_s is not None and seconds is not None:
            short = _within(seconds, max_s)
            lines, spare = tuple(filter(short, lines)), tuple(filter(short, spare))
        if not (lines or spare):
            return None
        self._spend(context, turn, now)
        if silent_p > 0 and self.rng.random() < silent_p:
            return None
        return self.pick(lines or spare, spare if lines else ())

    def _spend(self, context: str, turn: int | None, now: float) -> None:
        self.last_s[context] = now
        if turn is None:
            return
        self.spent.setdefault(turn, set()).add(context)
        for old in [t for t in self.spent if t <= turn - KEPT_REPLIES]:
            del self.spent[old]


def _within(seconds: Callable[[str], float | None], max_s: float) -> Callable[[str], bool]:
    """Whether a line's clip plays at most ``max_s`` (a line without a clip does not)."""

    def short(line: str) -> bool:
        length = seconds(line)
        return length is not None and length <= max_s + EPSILON_S

    return short


def script_language(text: str) -> str | None:
    """The language the words of ``text`` are written in, by the script of each word's first
    letter: "ru" when Cyrillic words are the most, "en" when Latin ones are (a transcriber
    that follows the speech writes English in Latin letters; GigaAM writes Cyrillic alone),
    :data:`UNKNOWN` when another script's are; None for no words or a tie ("ok да")."""
    counts = {"ru": 0, "en": 0, UNKNOWN: 0}
    for word in _WORDS.findall(text):
        first = word[0]
        if "Ѐ" <= first <= "ӿ":
            counts["ru"] += 1
        elif first.isascii() or "À" <= first <= "ɏ":
            counts["en"] += 1
        else:
            counts[UNKNOWN] += 1
    most = max(counts.values())
    leaders = [language for language, n in counts.items() if n == most]
    return leaders[0] if most and len(leaders) == 1 else None


def word_count(text: str) -> int:
    """How many words ``text`` holds (runs of letters, any script)."""
    return len(_WORDS.findall(text))


def thinking_chance(words: int, min_words: int, points: Sequence[tuple[float, float]]) -> float:
    """The chance a thinking filler is wanted after an utterance of ``words`` words: none under
    ``min_words`` (a short question is answered fast anyway), else ``points`` ((words,
    probability), the words rising) interpolated linearly, held at the first and the last
    point's probability beyond them."""
    if words < min_words or not points:
        return 0.0
    if words <= points[0][0]:
        return float(points[0][1])
    for (w0, p0), (w1, p1) in pairwise(points):
        if words <= w1:
            return float(p0 + (p1 - p0) * (words - w0) / (w1 - w0))
    return float(points[-1][1])


def again_due(said_s: float | None, now: float, again_s: float) -> bool:
    """The second filler's time: ``again_s`` (above 0) since the last filler was said
    (``said_s``, None while none has been)."""
    return said_s is not None and again_s > 0 and now - said_s >= again_s - EPSILON_S


def thinking_due(closed_s: float | None, first_token: bool, now: float, after_s: float) -> bool:
    """The thinking filler's time: ``after_s`` (above 0) since the turn closed (``closed_s``;
    None for a turn without a close of its own, the robot's report or one asked early before
    it) and the brain's first token not in (``first_token``)."""
    if closed_s is None or first_token or after_s <= 0:
        return False
    return now - closed_s >= after_s - EPSILON_S


def clip_lines(config: CascadeConfig) -> tuple[str, ...]:
    """The fillers' lines a voice makes clips of, the likeliest first: the hums and the
    conversation language's lines, then (the language following the words) every other
    language's; the backchannel's only with its flag on."""
    book, back = config.fillers, config.backchannel
    first = book.lines(config.conversation_language, backchannel=back)
    if config.filler_language_from == "config":
        return first
    return tuple(dict.fromkeys((*first, *book.lines(backchannel=back))))
