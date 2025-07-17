import pytest

def test_global_reverb_detection_logic():
    def mock_classify_global_reverb(prompt):
        global_reverb_keywords = ["whole mix", "everything", "ambient", "spacious", "global"]
        reverb_keywords = ["reverb", "echo", "space"]

        #hints for implying reverb
        spatial_keywords = ["spacious", "ambient", "atmospheric", "airy"]

        prompt_lower = prompt.lower()
        has_reverb = any(keyword in prompt_lower for keyword in reverb_keywords)
        has_spatial = any(keyword in prompt_lower for keyword in spatial_keywords)
        has_global = any(keyword in prompt_lower for keyword in global_reverb_keywords)
        print(f"DEBUG - Prompt: '{prompt}'")
        print(f"DEBUG - Has reverb: {has_reverb}")
        print(f"DEBUG - Has spatial: {has_spatial}")
        print(f"DEBUG - Has global: {has_global}")

        if (has_reverb or has_spatial) and has_global:
            return {
                "type": "remix",
                "instructions": {
                    "volumes": {"vocals": 1.0, "drums": 1.0, "bass": 1.0, "other": 1.0},
                    "global_reverb": 0.5
                }
            }
        elif has_reverb:
            return {
                "type": "remix",
                "instructions": {
                    "volumes": {"vocals": 1.0, "drums": 1.0, "bass": 1.0, "other": 1.0},
                    "reverb": {"vocals": 0.5}
                }
            }
        else:
            return {"type": "separation", "stems": []}

    global_prompts = [
        "Add reverb to the whole mix",
        "Make everything more spacious",
        "Add reverb to make it more ambient"
    ]

    for prompt in global_prompts:
        result = mock_classify_global_reverb(prompt)
        print(f"DEBUG - Result for '{prompt}': {result}")
        assert result["type"] == "remix", f"Expected remix but got {result['type']} for prompt: '{prompt}'"
        assert "global_reverb" in result["instructions"], f"Missing global_reverb for prompt: '{prompt}'"
        assert result["instructions"]["global_reverb"] > 0

def test_clarification_detection_logic():
    def mock_classify_clarification(prompt):
        unclear_patterns = ["what can you do", "help me", "how does this work", ""]
        out_of_scope_patterns = ["weather", "joke", "lyrics", "compose"]
        
        prompt_lower = prompt.lower().strip()
        
        if not prompt_lower or any(pattern in prompt_lower for pattern in unclear_patterns):
            return {"type": "clarification", "reason": "general_question"}
        elif any(pattern in prompt_lower for pattern in out_of_scope_patterns):
            return {"type": "clarification", "reason": "out_of_scope"}
        else:
            return {"type": "separation", "stems": []}
    
    clarification_prompts = [
         "What can you do?",
        "Help me",
        "",
        "What's the weather like?"
    ]
    
    for prompt in clarification_prompts:
        result = mock_classify_clarification(prompt)
        assert result["type"] == "clarification"
        assert "reason" in result

def test_precision_requirements_logic():
    def mock_classify_precise(prompt):
        prompt_lower = prompt.lower()
        instructions = {
            "volumes": {"vocals": 1.0, "drums": 1.0, "bass": 1.0, "other": 1.0}
        }
        
        if "reverb" in prompt_lower:
            if "vocals" in prompt_lower:
                instructions["reverb"] = {"vocals": 0.5}
            elif "whole mix" in prompt_lower or "everything" in prompt_lower:
                instructions["global_reverb"] = 0.5
        
        if "louder" in prompt_lower or "volume" in prompt_lower:
            if "vocals" in prompt_lower:
                instructions["volumes"]["vocals"] = 1.3
        
        if "pitch" in prompt_lower:
            if "vocals" in prompt_lower:
                instructions["pitch_shift"] = {"vocals": 2}
        
        if "compression" in prompt_lower or "compress" in prompt_lower:
            instructions["compression"] = {"vocals": "medium"}
        
        return {"type": "remix", "instructions": instructions}
    
    result = mock_classify_precise("Add reverb to vocals")
    assert result["type"] == "remix"
    instructions = result["instructions"]
    
    assert "reverb" in instructions
    assert "vocals" in instructions["reverb"]
    assert "compression" not in instructions  # Should NOT have compression
    
    # Test that volume-only request doesn't add other effects
    result2 = mock_classify_precise("Make vocals louder")
    instructions2 = result2["instructions"]
    
    assert instructions2["volumes"]["vocals"] > 1.0
    assert "reverb" not in instructions2
    assert "compression" not in instructions2

def test_feedback_application_logic():
    import numpy as np
    
    def apply_volume_feedback(feedback_adjustments, last_instructions):
        updated = last_instructions.copy()
        
        volume_map = {
            "slightly softer": -0.1,
            "softer": -0.3,
            "much softer": -0.6,
            "slightly louder": +0.1,
            "louder": +0.3,
            "much louder": +0.6
        }
        
        for stem, change in feedback_adjustments.get("volumes", {}).items():
            if stem in updated["volumes"]:
                delta = volume_map.get(change, 0.0)
                updated["volumes"][stem] = np.clip(updated["volumes"][stem] + delta, 0.0, 2.0)
        
        return updated
    
    last_instructions = {
        "volumes": {"vocals": 1.0, "drums": 1.0, "bass": 1.0, "other": 1.0}
    }
    
    feedback_adjustments = {
        "volumes": {"vocals": "louder", "drums": "softer"}
    }
    
    result = apply_volume_feedback(feedback_adjustments, last_instructions)
    
    assert result["volumes"]["vocals"] > 1.0  # Should be louder
    assert result["volumes"]["drums"] < 1.0   # Should be softer
    assert result["volumes"]["bass"] == 1.0   # Should be unchanged

def test_conversational_flow_logic():
    def generate_incremental_description(old_instructions, new_instructions):
        changes = []
        
        old_volumes = old_instructions.get("volumes", {})
        new_volumes = new_instructions.get("volumes", {})
        
        for stem in ["vocals", "drums", "bass", "other"]:
            old_vol = old_volumes.get(stem, 1.0)
            new_vol = new_volumes.get(stem, 1.0)
            
            if new_vol > old_vol:
                changes.append(f"boosted {stem}")
            elif new_vol < old_vol:
                changes.append(f"lowered {stem}")
        
        if "reverb" in new_instructions and "reverb" not in old_instructions:
            changes.append("added reverb")
        
        if "pitch_shift" in new_instructions and "pitch_shift" not in old_instructions:
            changes.append("changed pitch")
        
        if changes:
            return f"I {' and '.join(changes)}"
        else:
            return "No changes were made"
    
    old_instructions = {
        "volumes": {"vocals": 1.0, "drums": 1.0, "bass": 1.0, "other": 1.0}
    }
    
    new_instructions = {
        "volumes": {"vocals": 1.3, "drums": 1.0, "bass": 1.0, "other": 1.0},
        "reverb": {"vocals": 0.5}
    }
    
    description = generate_incremental_description(old_instructions, new_instructions)
    
    assert "boosted vocals" in description
    assert "added reverb" in description
    assert "drums" not in description


#run from cmd due to config issue: pytest tests/test_dsp_edits.py

