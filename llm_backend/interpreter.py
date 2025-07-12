import json
from transformers import pipeline
import logging
from demucs.demucs.pretrained import get_model
import openai

from openai.types.chat import (
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)

with open("api_key.txt", "r") as file:
    api_key = file.read().strip()

client = openai.OpenAI(api_key=api_key)

model = get_model(name="mdx_extra_q")
logger = logging.getLogger(__name__)
VALID_STEMS = {"vocals", "drums", "bass", "other"}

pipe = pipeline("text2text-generation", model="google/flan-t5-large")


def extract_stem_list(prompt: str) -> list[str]:
    logger.info(f"prompt: {prompt}")
    instruction = (
        "From the following user request, extract only the valid stems (vocals, drums, bass, other). "
        "Return them as a comma-separated list and nothing else."
    )

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                ChatCompletionSystemMessageParam(
                    role="system",
                    content="You are a helpful assistant that extracts music stems."
                ),
                ChatCompletionUserMessageParam(
                    role="user",
                    content=f"{instruction}\n\nUser request: {prompt}"
                )
            ],
            max_tokens=10,
            temperature=0
        )
        logger.info(f"Model response: {response}")

        content = response.choices[0].message.content
        logger.info(f"Model response: {content}")

        valid_stems = [s.strip() for s in content.lower().split(",") if s.strip() in VALID_STEMS]
        logger.info(f"Filtered stems: {valid_stems}")
        return valid_stems
    except Exception as e:
        logger.error(f"OpenAI error: {e}")
        return []

def classify_prompt(prompt: str) -> dict:
    system_prompt = """
        You are a music assistant. Based on user prompt, classify it as either:
        1. Separation → if user asks to extract stems like 'give me vocals and drums'
        2. Remix → if user gives volume/adjustment hints like 'make vocals louder, mellow the drums'
        Return a strict JSON structure as specified below.

        For separation:
        {
            "type": "separation", 
            "stems": ["vocals", "drums"]
        }
        
        - Valid stems are: vocals, drums, bass, other.
        - If user requests unsupported stems (e.g. trumpet), ignore them.
        - If none are valid, return: {"type": "separation", "stems": []}

        {
          "type": "remix",
          "instructions": {
            "volumes": {
              "vocals": 1.2,
              "drums": 0.7,
              "bass": 1.0,
              "other": 1.0
            },
            "reverb": {
              "vocals": 0.5
            },
            "pitch_shift": {
              "vocals": 2
            },
            "compression": {
              "vocals": "low"
            }
          }
        }

        Instructions:
        - **Instructions.volumes**: Always include all four stems with float multipliers (default 1.0 if not mentioned).
        - **Instructions.reverb**: Value between 0.0 (none) to 1.0 (max).
        - **Instructions.pitch_shift**: Integer in semitones (+ for up, - for down).
        - **Instructions.compression**: Choose among "low", "medium", "high".
        - Valid stems are: vocals, drums, bass, other. If the user asks for anything else (e.g. trumpet, guitar), return only valid stems and ignore the rest.
        - If the user requests an unsupported stem (e.g. "give me trumpet"), return: {"type": "separation", "stems": []}
        - For remixing, always include **all four stems** and only adjust volumes based on the prompt. If no volume is mentioned, use default 1.0.
        - If the user’s intent is unclear, default to {"type": "separation", "stems": []}
        For volumes:
            - If user says "slightly louder" or "a bit louder": set to 1.1
            - If user says "louder": set to 1.3
            - If user says "much louder": set to 1.6
            - If user says "extremely louder" or "max volume": set to 2.0
            - "slightly softer" or "a bit softer": 0.9
            - "softer": 0.7
            - "much softer": 0.5
            - "mute": 0.0
        For pitch_shift:
            - If user says "raise pitch by X semitones" or "increase pitch by X", set to +X.
            - If user says "lower pitch by X semitones" or "decrease pitch by X", set to -X.
            - If unspecified, set to 0.

        For reverb:
            - If user says "slight reverb" or "a bit of reverb": 0.2
            - If user says "reverb" or "add reverb": 0.5
            - If user says "heavy reverb" or "a lot of reverb": 0.8
            - If user says "maximum reverb" or "max reverb": 1.0
            - If unspecified, set to 0.0.
        
        For compression:
            - If user says "light compression" or "slight compression": "low"
            - If user says "compression" or "add compression": "medium"
            - If user says "strong compression" or "heavy compression": "high"
            - If unspecified, set to "medium".
        
        For filter:
            - Detect user requests like "add a low-pass filter at 4kHz" or "apply high-pass at 120Hz".
            - Return JSON with "filter" key, e.g.:
              "filter": {
                "vocals": {
                  "type": "lowpass",
                  "cutoff": 4000
                }
              }
            
            For band-pass (e.g. "telephone effect"):
            - Return "type": "bandpass", with "low_cutoff" and "high_cutoff".
            
            For eq:
            - Parse phrases like "boost 3kHz by 5dB with width 1.0" or "cut 100Hz by -3dB".
            - Return JSON with frequency (Hz), width (Q), and gain_db (dB).
            
        """

    chat = [
        ChatCompletionSystemMessageParam(role="system", content=system_prompt),
        ChatCompletionUserMessageParam(role="user", content=prompt)
    ]

    completion = client.chat.completions.create(
        model="gpt-4",
        messages=chat,
        temperature=0.5,
    )
    response = completion.choices[0].message.content

    try:
        return json.loads(response)
    except json.JSONDecodeError:
        return {"type": "separation", "stems": []}  # Fallback

def parse_feedback(feedback_text: str) -> dict:
    system_prompt = """
        You are a music DSP feedback interpreter assistant.
        Based on user feedback, extract intended adjustments in structured JSON.
        Return JSON like:

        {
          "volumes": {
            "vocals": "louder",
            "drums": "softer"
          },
          "reverb": {
            "vocals": "more",
            "drums": "less"
          },
          "pitch_shift": {
            "vocals": "+2",
            "drums": "-1"
          },
          "compression": {
            "vocals": "high",
            "drums": "low"
          }
        }

        Guidelines:
        - Only include stems that are explicitly mentioned in feedback.
        - For **volumes**, use one of: "slightly softer", "softer", "much softer", "mute", "slightly louder", "louder", "much louder".
        - For **pitch_shift**, return semitone adjustments with '+' or '-' (e.g. '+2' or '-1').
        - For **reverb**, use: "less", "more".
        - For **compression**, use: "low", "medium", "high".
        - If feedback does not mention an effect for a stem, omit that effect.
        - If nothing is detected, return an empty JSON object {}.

        Return only valid JSON as above. No explanations.
    """

    chat = [
        ChatCompletionSystemMessageParam(role="system", content=system_prompt),
        ChatCompletionUserMessageParam(role="user", content=feedback_text)
    ]

    completion = client.chat.completions.create(
        model="gpt-4",
        messages=chat,
        temperature=0.5
    )
    response = completion.choices[0].message.content

    try:
        return json.loads(response)
    except json.JSONDecodeError:
        return {}


def describe_audio_edit(task_type: str, instructions: dict = None, extracted_stems: list[str] = None) -> str:
    system_prompt = """
       You are an assistant summarizing audio processing actions. Given the task type ("separation" or "remix"), and optional parameters, return a short, natural sentence describing what was done.

       Guidelines:
       - If task is "separation", describe which stems were extracted (e.g., vocals, drums).
       - If task is "remix", describe only the meaningful DSP adjustments applied (volume, pitch, reverb, compression, filters, EQ).
       - Use musical and user-friendly language.
       - Ignore unchanged/default stems (e.g., volumes of 1.0 or effects not applied).
       - Keep it short and clear (1-2 sentences).
       - Don’t say "the user asked for" — speak as if you applied it.
       """

    user_prompt = {
        "task_type": task_type,
        "instructions": instructions,
        "extracted_stems": extracted_stems
    }

    chat = [
        ChatCompletionSystemMessageParam(role="system", content=system_prompt),
        ChatCompletionUserMessageParam(role="user", content=json.dumps(user_prompt))
    ]

    completion = client.chat.completions.create(
        model="gpt-4",
        messages=chat,
        temperature=0.5,
    )
    response = completion.choices[0].message.content.strip()

    return response