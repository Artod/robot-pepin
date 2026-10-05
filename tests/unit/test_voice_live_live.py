"""The google-genai adapter, offline: the setup it would send and the messages it reads."""

# ruff: noqa: RUF001 -- Russian speech is the data here, not a lookalike of Latin

from google.genai import types

from pepin.tools import TOOLS
from pepin.tools.schemas import gemini_function_declarations
from pepin.voice_live.config import LiveConfig
from pepin.voice_live.live import (
    END_CONVERSATION,
    ROBOT_REPORT,
    convert,
    declarations,
    live_config,
    system_instruction,
    usage_of,
)


def tools() -> list[types.FunctionDeclaration]:
    moving = {t.name for t in TOOLS if t.moves}
    registry = [d for d in gemini_function_declarations(TOOLS) if d["name"] != "say"]
    return declarations(registry, moving)


def test_drives_do_not_block_the_conversation_and_end_conversation_is_offered() -> None:
    by_name = {d.name: d for d in tools()}
    assert by_name["go_to"].behavior == types.Behavior.NON_BLOCKING
    assert by_name["where_am_i"].behavior == types.Behavior.BLOCKING
    assert END_CONVERSATION in by_name and "say" not in by_name
    assert by_name["go_to"].parameters is not None
    assert by_name["go_to"].parameters.required == ["place"]


def test_the_setup_asks_for_audio_transcripts_vad_compression_and_resumption() -> None:
    config = LiveConfig(transcription_languages=("ru-RU",))
    setup = live_config(config, tools(), "handle-1", system_instruction(config, None))
    assert setup.response_modalities == [types.Modality.AUDIO]
    assert setup.speech_config is not None and setup.speech_config.voice_config is not None
    voice = setup.speech_config.voice_config.prebuilt_voice_config
    assert voice is not None and voice.voice_name == config.voice
    assert setup.input_audio_transcription is not None
    assert setup.input_audio_transcription.language_codes == ["ru-RU"]
    assert setup.output_audio_transcription is not None
    vad = setup.realtime_input_config
    assert vad is not None and vad.automatic_activity_detection is not None
    assert vad.automatic_activity_detection.disabled is False
    compression = setup.context_window_compression
    assert compression is not None and compression.trigger_tokens == 16000
    assert compression.sliding_window is not None
    assert compression.sliding_window.target_tokens == 8000
    assert setup.session_resumption is not None
    assert setup.session_resumption.handle == "handle-1"
    assert "Пепин" in str(setup.system_instruction)


def test_the_instructions_carry_the_places_and_the_personas_lines_from_the_config() -> None:
    config = LiveConfig.load()
    text = system_instruction(config, ["home", "printer", "bookshelf"])
    assert "home, printer, bookshelf" in text and "call go_to at once" in text
    assert config.persona_acknowledge in text and config.persona_arrived in text
    assert "status driving" in text and ROBOT_REPORT in text  # the start, then the robot's report
    assert "never on your own" in text
    unknown = system_instruction(LiveConfig(), None)
    assert "list_places gives the names" in unknown and "style of" not in unknown
    once = system_instruction(LiveConfig(report_drive_start=False, persona_arrived="Тут"), None)
    assert "status driving" not in once and "'Тут'" in once


def test_audio_transcripts_and_the_turn_end_are_read() -> None:
    message = types.LiveServerMessage(
        server_content=types.LiveServerContent(
            model_turn=types.Content(
                parts=[
                    types.Part(inline_data=types.Blob(data=b"\x01\x02", mime_type="audio/pcm")),
                    types.Part(inline_data=types.Blob(data=b"\x03\x04", mime_type="audio/pcm")),
                ]
            ),
            input_transcription=types.Transcription(text="где ты"),
            output_transcription=types.Transcription(text="Я у"),
            turn_complete=True,
            interrupted=True,
        )
    )
    m = convert(message)
    assert m.audio == b"\x01\x02\x03\x04"
    assert (m.input_text, m.output_text) == ("где ты", "Я у")
    assert m.turn_complete and m.interrupted
    parts = m.raw["server_content"]["model_turn"]["parts"]
    assert parts[0]["inline_data"] == {"bytes": 4}  # the log keeps the size, not the audio


def test_tool_calls_cancellations_handles_and_go_away_are_read() -> None:
    m = convert(
        types.LiveServerMessage(
            tool_call=types.LiveServerToolCall(
                function_calls=[types.FunctionCall(id="c1", name="go_to", args={"place": "home"})]
            ),
            tool_call_cancellation=types.LiveServerToolCallCancellation(ids=["c0"]),
            session_resumption_update=types.LiveServerSessionResumptionUpdate(
                new_handle="h2", resumable=True
            ),
            go_away=types.LiveServerGoAway(time_left="12.5s"),
        )
    )
    assert [(c.id, c.name, c.args) for c in m.tool_calls] == [("c1", "go_to", {"place": "home"})]
    assert m.cancelled == ("c0",) and m.handle == "h2" and m.go_away_s == 12.5
    assert m.activity is None
    not_resumable = convert(
        types.LiveServerMessage(
            session_resumption_update=types.LiveServerSessionResumptionUpdate(
                new_handle="h3", resumable=False
            )
        )
    )
    assert not_resumable.handle is None


def test_usage_is_split_by_modality_and_thoughts_count_as_text_out() -> None:
    usage = usage_of(
        types.UsageMetadata(
            prompt_token_count=1000,
            prompt_tokens_details=[
                types.ModalityTokenCount(modality=types.MediaModality.AUDIO, token_count=700),
                types.ModalityTokenCount(modality=types.MediaModality.TEXT, token_count=300),
            ],
            response_token_count=200,
            response_tokens_details=[
                types.ModalityTokenCount(modality=types.MediaModality.AUDIO, token_count=150)
            ],
            thoughts_token_count=20,
            tool_use_prompt_token_count=5,
        )
    )
    assert (usage.audio_in, usage.text_in, usage.audio_out, usage.text_out) == (700, 305, 150, 70)
    m = convert(types.LiveServerMessage(usage_metadata=types.UsageMetadata(prompt_token_count=9)))
    assert m.usage is not None and m.usage.text_in == 9 and m.usage.reports == 1


def test_the_servers_vad_is_read_as_the_persons_speech_starting_and_ending() -> None:
    kinds = {
        types.VoiceActivityType.ACTIVITY_START: "start",
        types.VoiceActivityType.ACTIVITY_END: "end",
    }
    for kind, expected in kinds.items():
        m = convert(
            types.LiveServerMessage(voice_activity=types.VoiceActivity(voice_activity_type=kind))
        )
        assert m.activity == expected
