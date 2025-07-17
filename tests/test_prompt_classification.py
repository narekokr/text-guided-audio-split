import pytest
from llm_backend.interpreter import classify_prompt

@pytest.mark.parametrize("prompt,expected_type,expected_keys,expected_stems", [
    (
        "Make vocals louder and add reverb to vocals",
        "remix",
        ["volumes", "reverb"],
        None
    ),
    (
        "Shift vocals up by 2 semitones",
        "remix",
        ["volumes", "pitch_shift"],
        None
    ),
    (
        "Give me vocals and drums stems only",
        "separation",
        ["stems"],
        ["vocals", "drums"]
    ),
    (
        "I love this song",
        "clarification",
        ["reason"],
        None
    ),

    (
        "Extract vocals and trumpet",
        "separation",
        ["stems"],
        ["vocals"]  # trumpet ignored
    ),
    (
        "asdfghjk",
        "clarification",
        ["reason"],
        None
    ),
    (
        "",
        "clarification",
        ["reason"],
        None
    ),
],
ids=[
    "remix_vocals_reverb",
    "remix_pitch_shift",
    "separation_vocals_drums",
    "clarification_love_song",
    "separation_vocals_trumpet",
    "clarification_asdfghjk",
    "clarification_empty"
])

def test_classify_prompt_variants(prompt, expected_type, expected_keys, expected_stems):
    result = classify_prompt(prompt)
    assert result["type"] == expected_type

    if expected_type == "remix":
        instructions = result.get("instructions", {})
        for key in expected_keys:
            assert key in instructions

        assert set(instructions["volumes"].keys()) == {"vocals", "drums", "bass", "other"}

    elif expected_type == "separation":
        assert "stems" in result
        if expected_stems is not None:
            assert set(result["stems"]) == set(expected_stems)

    elif expected_type == "clarification":
        assert "reason" in result
        # Allow more reason types that the LLM might return
        valid_reasons = ["unclear_intent", "general_question", "out_of_scope", "nonsense", "unclear", "invalid"]
        assert result["reason"] in valid_reasons, f"Got unexpected reason: {result['reason']}"


def test_debug_pitch_shift():
    prompt = "Shift vocals up by 2 semitones"
    result = classify_prompt(prompt)

    print(f"\nDEBUG - Full result: {result}")
    print(f"DEBUG - Type: {result.get('type')}")
    if "instructions" in result:
        instructions = result["instructions"]
    assert result["type"] == "remix"
    instructions = result.get("instructions", {})
    assert "volumes" in instructions
    assert "pitch_shift" in instructions

def test_global_reverb_classification():
    result = classify_prompt("Add reverb to the whole mix")
    assert result["type"] == "remix"
    instructions = result.get("instructions", {})
    assert "global_reverb" in instructions
    assert instructions["global_reverb"] > 0

def test_clarification_classification():
    result = classify_prompt("What can you do?")
    assert result["type"] == "clarification"
    assert "reason" in result

def test_precision_reverb_only():
    result = classify_prompt("Add reverb to vocals")
    assert result["type"] == "remix"
    instructions = result.get("instructions", {})

    assert "reverb" in instructions
    assert "vocals" in instructions["reverb"]

    if "compression" in instructions:
        compression_values = list(instructions["compression"].values())
        assert not any(compression_values) or all(v is None for v in compression_values)
