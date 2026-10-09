"""Pepin's own voice loop, the cascade: local ears -> Claude -> local voice, every stage timed.

    mic (board array) --frames--> Ears (mic thread: gate, end of turn, barge-in)
        --utterance--> Whisper (worker) + Praat tag line (worker)
        --"[Артём: тише обычного] words"--> Claude (asyncio, streamed, prompt-cached, tools)
        --sentence by sentence--> Qwen3-TTS (its own thread) --PCM--> pacer --> speaker

Nothing waits for more than it needs: the first sentence goes to the voice while the brain
writes the second; the voice makes the next sentence while the one before plays; a tool's
"Сейчас гляну." is spoken while the tool runs. Every stage is behind a small protocol
(:mod:`pepin.voice_cascade.stages`) with a fake (:mod:`pepin.voice_cascade.fake`), and every
choice is a key of ``config/voice_cascade.json`` (:mod:`pepin.voice_cascade.config`): the
transcriber's model, the brain's model, thinking mode, effort and cache, the voice, and the
flags ``barge_in``, ``backchannel`` and ``prosody_tags``. The entry point is
``scripts/voice_cascade.py``, with a ``--bench`` that replays recorded utterances and prints
p50/p90 per stage (:mod:`pepin.voice_cascade.bench`, :mod:`pepin.voice_cascade.timeline`).
The Gemini Live loop (:mod:`pepin.voice_live`) stays as it is; its wake gate, pacer, audio
link, fakes and ledger are reused from there.
"""
