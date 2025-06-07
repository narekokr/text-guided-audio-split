import torchaudio
from transformers import pipeline
import logging
from demucs.demucs.pretrained import get_model
from demucs.demucs.apply import apply_model

model = get_model(name="mdx_extra_q")
#waveform, sr = torchaudio.load("your_audio.wav")

logger = logging.getLogger(__name__)
VALID_STEMS = {"vocals", "drums", "bass", "other"}

#pretrained model mdx_extra_q is always trained to output those 4 stems

#sources = apply_model(model, waveform, device="cpu")
# sources is a dict like: { 'vocals': tensor, 'drums': tensor, ... }

def interpret():
    return 0

"""
#Note:
Input is natural language text and output that we want is also short text response, hence we need text2text task and flan-t5 is trained specifically for such instruction-following behavior
"""
# Load model once at module level
pipe = pipeline("text2text-generation", model="google/flan-t5-large")


def interpret_prompt(prompt: str) -> list[str]:
    """
       Uses Hugging Face model to interpret a user prompt and extract desired stems.

       Args:
           prompt (str): e.g., "separate vocals and drums"

       Returns:
           list[str]: e.g., ["vocals", "drums"]
    """
    logger.info(f"Received prompt: {prompt}")

    instruction = (
        f"List the musical stems to extract based on this: '{prompt}'. "
        f"Return only a comma-separated list like: vocals, drums"
    )

    #only generates up to 10 tokens in new response
    #this limits verbosity as we don't need long paragraphs, just a short stem list
    #so 10 is safe upper limit for current scope, and can be increased if necessary
    raw_stems = pipe(instruction, max_new_tokens=10)[0]["generated_text"]
    response = [stem for stem in raw_stems if stem in VALID_STEMS]

    logger.info(f"Filtered stems: {response}")
    valid_stems = [s.strip() for s in response.split(",") if s.strip()]

    #Passing anything else (like "guitar", "piano") will raise errors or produce silence. So filtering protects the system for 1st iteration level. Adjust this later
    return valid_stems

