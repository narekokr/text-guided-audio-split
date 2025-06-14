import torchaudio
from transformers import pipeline
import logging
from demucs.demucs.pretrained import get_model
import openai

from openai.types.chat import (
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)

api_key = "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"
client = openai.OpenAI(api_key=api_key)

#os.environ["OPENAI_API_KEY"]

model = get_model(name="mdx_extra_q")
logger = logging.getLogger(__name__)
VALID_STEMS = {"vocals", "drums", "bass", "other"}

#pretrained model mdx_extra_q is always trained to output those 4 stems

"""
#Note:
Input is natural language text and output that we want is also short text response, hence we need text2text task and flan-t5 is trained specifically for such instruction-following behavior
"""
# Load model once at module level
pipe = pipeline("text2text-generation", model="google/flan-t5-large")


def interpret_prompt(prompt: str) -> list[str]:
    """
       Uses OpenAI ChatGPT model to interpret a user prompt and extract desired stems.

       Args:
           prompt (str): e.g., "separate vocals and drums"

       Returns:
           list[str]: e.g., ["vocals", "drums"]
    """
    logger.info(f"prompt: {prompt}")

    instruction = (
        "You are a music AI system. Given a user request, return a list of stems "
        "to extract. Valid stems are: vocals, drums, bass, other. "
        f"Prompt: '{prompt}'\n"
        "Return only a comma-separated list of valid stems from the prompt"
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
                    content=(
                        "From the following user request, extract only the valid stems (vocals, drums, bass, other). "
                        "Return them as a comma-separated list and nothing else.\n\n"
                        "User request: separate vocals and drums"
                    )
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

"""

    #only generates up to 10 tokens in new response
    #this limits verbosity as we don't need long paragraphs, just a short stem list
    #so 10 is safe upper limit for current scope, and can be increased if necessary
    raw_stems = pipe(instruction, max_new_tokens=10)[0]["generated_text"]
    logger.info(f"Model response: {raw_stems}")
    valid_stems = [s.strip() for s in raw_stems.lower().split(",") if s.strip() in VALID_STEMS]

    logger.info(f"Filtered stems: {valid_stems}")
    #Passing anything else (like "guitar", "piano") will raise errors or produce silence. So filtering protects the system for 1st iteration level. Adjust this later
    return valid_stems
"""