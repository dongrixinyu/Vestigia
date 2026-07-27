from __future__ import annotations

from vestigia.identify import ModelFingerprint
from vestigia.workflow import load_fingerprint, save_fingerprint


def test_saved_fingerprint_can_be_loaded_without_http_or_mock_libraries(tmp_path) -> None:
    fingerprint = ModelFingerprint(
        model="reference-model",
        provider="openai_compatible",
        prompt="Reply with one word.",
        system=None,
        temperature=0.1,
        max_tokens=16,
        request_configuration={"extra_body": {"top_p": 0.9}},
        field="parsed.text",
        values=('"yes"',),
        distribution={'"yes"': 1.0},
        text_length={"stability": {"reliable": True}},
        stability={"reliable": True},
    )
    output = tmp_path / "reference-fingerprint.json"

    save_fingerprint(fingerprint, output)
    loaded = load_fingerprint(output)

    assert loaded == fingerprint
