import logging
import openai

from openai.types.chat import (
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)

# These will change after fine-tuning
VALID_STEMS = {"vocals", "drums", "bass", "other"}

# Read your OpenAI key (update path if needed)
with open("api_key.txt", "r") as file:
    api_key = file.read().strip()

client = openai.OpenAI(api_key=api_key)
logger = logging.getLogger(__name__)

def interpret_prompt(prompt: str) -> list[str]:
    """
    Uses OpenAI ChatGPT model to interpret a user prompt and extract desired stems.

    Args:
        prompt (str): e.g., "separate vocals and drums"
    Returns:
        list[str]: e.g., ["vocals", "drums"]
    """
    logger.info(f"User prompt: {prompt}")
    instruction = (
        "From the following user request, extract only the valid stems "
        "(vocals, drums, bass, other). "
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
            max_tokens=16,
            temperature=0
        )
        content = response.choices[0].message.content
        logger.info(f"Raw model response: {content}")

        # Parse and filter
        stems = [s.strip().lower() for s in content.split(",")]
        valid_stems = [s for s in stems if s in VALID_STEMS]
        logger.info(f"Filtered valid stems: {valid_stems}")
        return valid_stems
    except Exception as e:
        logger.error(f"OpenAI error: {e}")
        return []