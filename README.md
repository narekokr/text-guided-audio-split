# Text-Guided Audio Stem Separation with Demucs & Hugging Face

This stage of the project demonstrates how to perform **text-guided audio source separation** using:

- **Flan-T5** from Hugging Face to interpret user prompts like `"extract vocals and drums"` and convert them into target stems.
- **Demucs** (`mdx_extra_q` model) for high-quality music source separation into standard stems: `vocals`, `drums`, `bass`, and `other`.
- **FastAPI** to serve a `/separate` endpoint that accepts an audio file and a prompt, performs inference, and returns filtered audio stems in Base64 format.
- **FastAPI** to serve a `/chat` endpoint that orchestrates communication

> ✅ A validation layer ensures only supported stems are passed to Demucs (`vocals`, `drums`, `bass`, `other`), guarding against incorrect or unsafe prompt interpretations.

---

## 🚀 How It Works

1. **User uploads audio** and provides a **natural language prompt** (e.g. `"only separate vocals"`).
2. The prompt is passed to `interpreter.py`, which returns a **clean list of desired stems**.
3. Demucs separates the audio into 4 standard stems.
4. Only the stems mentioned in the prompt are returned in the response.
5. Output stems are sent back via API in downloadable formats.

---

## 🧪 One-Shot Prompt Example

Example input:

```text
Prompt: "Extract vocals and bass only"


## For Fine-Tuning update the following:

1. Dataset path in conf/config.yaml
2. CLAP checkpoint path in demucs/api.py
3. CLAP checkpoint path in audio_utils/separator.py
4. Dataset path, CLAP checkpoint path and Save Path in fine_tuning/preprocess_triplets.py


## TO DO:

1. Remix instructions / Simple DSP
2. CLAP similarity check to determine instruments in the mix
3. Relax the fixed set of Valid Stems in audio_utils/separator.py
