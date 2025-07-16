import json
from transformers import pipeline
import logging
from demucs.demucs.pretrained import get_model
from demucs.demucs.hdemucs import HDemucs
import openai
import torch

from openai.types.chat import (
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)

with open("api_key.txt", "r") as file:
    api_key = file.read().strip()

client = openai.OpenAI(api_key=api_key)
sources = ["stem"]

#model = get_model(name="mdx_extra_q")
model = HDemucs(sources = sources)
checkpoint = torch.load("outputs/xps/97d170e1/best.th", map_location="cpu",weights_only=False)
model.load_state_dict(checkpoint['state'])
model.eval()

logger = logging.getLogger(__name__)
pipe = pipeline("text2text-generation", model="google/flan-t5-large")


def classify_prompt(prompt: str) -> dict:
    system_prompt = """
        You are a music assistant. Based on user prompt, classify it as either:
        1. Separation → if user asks to extract stems like 'give me violins and drums'
        2. Remix → if user gives volume/adjustment hints like 'make violins louder, mellow the drums'
        For both tasks, you are strongly encouraged to choose stems from the following list, but you are allowed to include a stem that 
        is not on the list if one of the user requested stems is not remotely relevant to any stem on the list. 
        
        Grand Piano,Electric Piano,Harpsichord,Clavinet,Celesta,Glockenspiel,Music Box,Vibraphone,Marimba,Xylophone,Tubular Bells,Dulcimer,Organ 
        Accordion,Harmonica,Tango Accordion,Acoustic Guitar,Electric Guitar,Distortion Guitar,Acoustic Bass,Electric Bass,Synth Bass,Violin,Viola 
        Cello,Contrabass,Orchestral Harp,Timpani,String Ensemble,Synth Strings,Trumpet,Trombone,Tuba,French Horn,Brass Section,Synth Brass,Soprano Sax 
        Alto Sax,Tenor Sax,Baritone Sax,Oboe,English Horn,Bassoon,Clarinet,Piccolo,Flute,Recorder,Pan Flute,Whistle,Agogo,Steel Drums,Woodblock,
        Melodic Tom,Synth Drum,Drums

        Return a strict JSON structure as specified below.

        For separation:
        {
            "type": "separation", 
            "stems": ["Violin", "Drums"]
        }
        
        For remix: 
        {
          "type": "remix",
          "instructions": {
            "volumes": {
              "Violin": 1.2,
              "Drums": 0.7,
              "Bass": 1.0,
            },
            "reverb": {
              "Violin": 0.5
            },
            "pitch_shift": {
              "Violin": 2
            },
            "compression": {
              "Violin": "low"
            }
          }
        }

        Instructions:
        - **Instructions.reverb**: Value between 0.0 (none) to 1.0 (max).
        - **Instructions.pitch_shift**: Integer in semitones (+ for up, - for down).
        - **Instructions.compression**: Choose among "low", "medium", "high".
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
       You are an assistant summarizing audio processing actions. Given the task type ("separation" or "remix"), and optional parameters, 
       return a short, natural sentence describing what was done.

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