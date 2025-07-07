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
        "separation",
        ["stems"],
        []
    ),

    (
        "Extract vocals and trumpet",
        "separation",
        ["stems"],
        ["vocals"]  # trumpet ignored
    ),
    (
        "asdfghjk",
        "separation",
        ["stems"],
        []  # default fallback
    ),
    (
        "",
        "separation",
        ["stems"],
        []  # default fallback for empty prompt
    ),
],
ids=[ #for identifying which specific test case failed
 "remix_vocals_reverb",
 "remix_pitch_shift",
 "separation_vocals_drums",
 "separation_unclear",
 "separation_unknown_stem_trumpet",
 "separation_nonsense",
 "separation_empty"
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


def test_debug_pitch_shift():
    """Debug test to see exactly what classify_prompt returns"""
    prompt = "Shift vocals up by 2 semitones"
    result = classify_prompt(prompt)

    print(f"\nDEBUG - Full result: {result}")
    print(f"DEBUG - Type: {result.get('type')}")
    print(f"DEBUG - Instructions: {result.get('instructions', {})}")

    if "instructions" in result:
        instructions = result["instructions"]
        print(f"DEBUG - Reverb: {instructions.get('reverb', {})}")
        print(f"DEBUG - Pitch shift: {instructions.get('pitch_shift', {})}")
        print(f"DEBUG - Volumes: {instructions.get('volumes', {})}")

    # Your original assertions
    assert result["type"] == "remix"
    instructions = result.get("instructions", {})
    assert "volumes" in instructions
    assert "pitch_shift" in instructions

"""
TODO
Unknown/unsupported stems
Empty or nonsense prompts
Fallback robustness with mocked invalid JSON
Edge-case inputs with emojis or non-ASCII
"""

#run by:  pytest tests/test_prompt_classification.py -v
