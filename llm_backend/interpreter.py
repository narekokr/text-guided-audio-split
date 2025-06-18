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
        Return JSON structure similar according to the specifications below:
        For separation:
        {"type": "separation", "stems": ["vocals", "drums"]}
        
        For remix:
        {"type": "remix", "volumes": {"vocals": 1.2, "drums": 0.7, "bass": 1.0, "other": 1.0}}
        
        Instructions:
        - Valid stems are: vocals, drums, bass, other. If the user asks for anything else (e.g. trumpet, guitar), return only valid stems and ignore the rest.
        - If the user requests an unsupported stem (e.g. "give me trumpet"), return: {"type": "separation", "stems": []}
        - For remixing, always include **all four stems** and only adjust volumes based on the prompt. If no volume is mentioned, use default 1.0.
        - If the user’s intent is unclear, default to {"type": "separation", "stems": []}
        """


    chat = [
        ChatCompletionSystemMessageParam(role="system", content=system_prompt),
        ChatCompletionUserMessageParam(role="user", content=prompt)
    ]

    completion = client.chat.completions.create(
        model="gpt-4",
        messages=chat,
        temperature=0,
    )
    response = completion.choices[0].message.content

    try:
        return json.loads(response)
    except json.JSONDecodeError:
        return {"type": "separation", "stems": []}  # Fallback
